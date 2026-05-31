"""Learned motion-embedding matcher — drop-in for asl.matcher.Matcher.

Scores a live window against a target sign's prototypes by cosine of their motion
embeddings, mapped to [0, 1]. Same (window, target) -> strength interface as
Matcher, so session/app swap one for the other. The encoder is injected as a
callable so this is unit-testable WITHOUT onnx (tests pass a plain python encode
fn + synthetic prototypes); `from_files` builds the onnxruntime-backed encoder for
production.

A gloss carries MULTIPLE prototypes (k=2-3 "phase" prototypes from enroll.py, plus
optional per-user calibration vectors). They are OR'd: a window matches a gloss if
it is close to ANY of that gloss's prototypes, so strength = MAX (cos+1)/2 over the
gloss's prototypes and best_distance = MIN (1 - cos).
"""
import numpy as np


class EmbeddingMatcher:
    def __init__(self, encode, prototypes, min_motion=0.0):
        """
        Args:
            encode: callable (window (T, 84) float32) -> L2-normed embedding (emb_dim,).
            prototypes: {gloss: protos}, where `protos` is either a single L2-normed
                embedding (emb_dim,) or a stack of them (k, emb_dim). Both forms are
                stored as a (k, emb_dim) matrix so a gloss can hold several phase /
                calibration prototypes.
            min_motion: temporal-motion floor (mean per-coordinate std over the
                window's time axis). A window below it has effectively NO hand motion
                — a no-hands frame (MediaPipe zero-fills an undetected hand) or hands
                held still — which this MOTION encoder maps to a near-universal ~1.0
                cosine, causing false confirms. Such windows score worst. 0.0 (the
                default) disables the gate, preserving the pure-cosine contract for
                direct/unit use; `from_files` switches it on for production.
        """
        self._encode = encode
        self._protos = {g: self._as_matrix(p) for g, p in prototypes.items()}
        self._min_motion = float(min_motion)
        # Stacked, L2-normed prototype matrix for SINGLE-ENCODE ranking: every
        # prototype row across all glosses, with `_row_gloss[i]` mapping row i to its
        # gloss index in `_gloss_list`. rank() encodes the window once and does one
        # matmul against this, instead of re-encoding per gloss (1308x).
        self._gloss_list = list(self._protos)
        rows = []
        row_gloss = []
        for gi, g in enumerate(self._gloss_list):
            mat = self._protos[g]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0  # leave a zero row ~zero (ranks worst)
            rows.append((mat / norms).astype(np.float32))
            row_gloss.extend([gi] * mat.shape[0])
        self._proto_matrix = (
            np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), np.float32)
        )
        self._row_gloss = np.asarray(row_gloss, dtype=np.intp)

    def _embed(self, window: np.ndarray):
        """Validate + encode a window into an L2-normed embedding. Raises ValueError
        on a non-2-D window (a flat frame or a (B,T,84) batch is not a single motion
        window). Returns None for a no-motion window (motion floor) or a degenerate
        (zero-norm) embedding — callers map None to the worst score."""
        win = np.asarray(window, dtype=np.float32)
        if win.ndim != 2:
            raise ValueError(
                f"window must be a 2-D (T, 84) sequence, got ndim={win.ndim}; "
                "the embedding matcher scores motion over time, not a single frame "
                "or a batch"
            )
        # Motion floor: a window with (near) zero temporal variation has no hand
        # motion to embed — a no-hands frame or hands held still — and this encoder
        # maps such a constant to a ~1.0 cosine against most prototypes.
        if self._min_motion > 0.0 and (
            win.shape[0] < 2 or float(np.mean(np.std(win, axis=0))) < self._min_motion
        ):
            return None
        emb = np.asarray(self._encode(win), dtype=np.float32).reshape(-1)
        en = np.linalg.norm(emb)
        if en < 1e-12:
            return None
        return emb / en  # L2-normed, so a cosine is a plain dot product

    @staticmethod
    def _as_matrix(p) -> np.ndarray:
        """Coerce a gloss's prototype(s) to a 2-D (k, emb_dim) float32 matrix.

        Accepts a single (emb_dim,) vector or an already-stacked (k, emb_dim) array
        so callers (and the legacy one-prototype-per-gloss store) compose unchanged."""
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2:
            raise ValueError(
                f"prototype for a gloss must be 1-D (emb_dim,) or 2-D (k, emb_dim), "
                f"got ndim={arr.ndim}"
            )
        return arr

    def _cosines(self, window: np.ndarray, target: str) -> np.ndarray:
        """Cosines of the window's embedding against EACH of `target`'s prototypes.

        Returns an empty array for a no-motion window or a degenerate (zero-norm)
        embedding, which the callers map to the worst score rather than a neutral one.
        """
        protos = self._protos[target]  # KeyError on unknown target (validated first)
        emb = self._embed(window)      # ValueError on bad ndim; None if degenerate
        if emb is None:
            return np.empty(0, dtype=np.float32)
        # Drop any zero-norm prototype rows; renormalize the rest (the export may
        # leave them not-quite-unit). emb is already unit, so this is a plain cosine.
        pn = np.linalg.norm(protos, axis=1)
        keep = pn >= 1e-12
        if not np.any(keep):
            return np.empty(0, dtype=np.float32)
        protos, pn = protos[keep], pn[keep]
        # Clamp to [-1, 1]: float32 rounding can push a near-unit cosine just past the
        # bounds, which would make strength leave [0, 1] or best_distance go negative.
        return np.clip((protos @ emb) / pn, -1.0, 1.0).astype(np.float32)

    def strength(self, window: np.ndarray, target: str) -> float:
        """Match strength in [0, 1]: MAX over `target`'s prototypes of (cosine + 1)/2.

        Taking the max means the best-matching phase prototype wins, so a mid-sign
        window matches whichever phase it lands in. A degenerate (zero) embedding
        scores 0.0 (worst) rather than the neutral 0.5 a cosine of 0 would give."""
        cosines = self._cosines(window, target)
        if cosines.size == 0:
            return 0.0
        return float((cosines.max() + 1.0) / 2.0)

    def best_distance(self, window: np.ndarray, target: str) -> float:
        """Smallest cosine distance (1 - cosine) to any of `target`'s prototypes, in
        [0, 2] (parity with Matcher.best_distance, a non-negative distance). A
        degenerate (zero) embedding gets the worst distance 2.0, consistent with
        strength()==0.0."""
        cosines = self._cosines(window, target)
        if cosines.size == 0:
            return 2.0
        return float(1.0 - cosines.max())

    def rank(self, window: np.ndarray, k: int = 3) -> list:
        """The k closest glosses by best-per-gloss cosine distance. ONE entry per
        gloss, ranked by its nearest prototype. Encodes the window ONCE (then a
        single matmul over all prototype rows) so it is cheap enough for the
        per-confirm open-set rank gate, not only the debug HUD. Returns [] for a
        no-motion / degenerate window (nothing to rank)."""
        emb = self._embed(window)  # ValueError on bad ndim; None if degenerate
        if emb is None or self._proto_matrix.shape[0] == 0:
            return []
        cos = self._proto_matrix @ emb                       # (P,) one per proto row
        best = np.full(len(self._gloss_list), -np.inf, dtype=np.float32)
        np.maximum.at(best, self._row_gloss, cos)            # best phase per gloss
        k = min(k, len(self._gloss_list))
        top = np.argpartition(best, -k)[-k:]
        top = top[np.argsort(best[top])[::-1]]               # sort the top-k desc
        return [{"gloss": self._gloss_list[i], "distance": round(float(1.0 - best[i]), 3)}
                for i in top]

    @classmethod
    def from_files(cls, onnx_path, prototypes_path, min_motion=0.01):
        """Build an onnxruntime-backed matcher from an exported encoder + an .npz
        of prototypes (parallel arrays `glosses` and `protos`, ONE ROW PER PROTOTYPE
        so a gloss may repeat for its k phase prototypes).

        `min_motion` (on by default here) gates out no-motion windows — see __init__.
        0.01 sits ~1000x above a constant window's 0.0 yet far below a real sign's
        temporal std (WLASL clips: min ~0.006, median ~0.79), so it rejects no-hands
        / held-still windows while passing genuine signing."""
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        def encode(window: np.ndarray) -> np.ndarray:
            w = np.asarray(window, dtype=np.float32)
            if w.ndim == 2:
                w = w[None, ...]  # (1, T, 84)
            out = sess.run(None, {input_name: w})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)

        data = np.load(prototypes_path, allow_pickle=True)
        glosses = [str(g) for g in data["glosses"]]
        protos_mat = np.asarray(data["protos"], dtype=np.float32)
        # `glosses` and `protos` are parallel-indexed (row i is a prototype for
        # glosses[i]); guard against corruption that would mismatch a row to a gloss.
        if protos_mat.shape[0] != len(glosses):
            raise ValueError(
                f"prototypes file mismatch: {len(glosses)} glosses but "
                f"{protos_mat.shape[0]} prototype rows"
            )
        # Group rows by gloss so a gloss carries its k phase prototypes together.
        grouped: dict[str, list] = {}
        for g, row in zip(glosses, protos_mat):
            grouped.setdefault(g, []).append(row)
        protos = {g: np.stack(rows, axis=0) for g, rows in grouped.items()}
        return cls(encode, protos, min_motion=min_motion)
