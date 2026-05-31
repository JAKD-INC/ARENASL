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
    def __init__(self, encode, prototypes):
        """
        Args:
            encode: callable (window (T, 84) float32) -> L2-normed embedding (emb_dim,).
            prototypes: {gloss: protos}, where `protos` is either a single L2-normed
                embedding (emb_dim,) or a stack of them (k, emb_dim). Both forms are
                stored as a (k, emb_dim) matrix so a gloss can hold several phase /
                calibration prototypes.
        """
        self._encode = encode
        self._protos = {g: self._as_matrix(p) for g, p in prototypes.items()}

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

        Returns an empty array for a degenerate (zero-norm) embedding, which the
        callers map to the worst score rather than a neutral one.
        """
        # Require exactly a 2-D (T, 84) window. A flat (84,) frame is not a motion
        # window; a 3-D (B, T, 84) batch would bypass from_files' batch-wrapping
        # (which only prepends a batch axis when ndim == 2) and silently feed an
        # extra axis to the encoder, yielding a meaningless score. Guard both.
        win = np.asarray(window, dtype=np.float32)
        if win.ndim != 2:
            raise ValueError(
                f"window must be a 2-D (T, 84) sequence, got ndim={win.ndim}; "
                "the embedding matcher scores motion over time, not a single frame "
                "or a batch"
            )
        protos = self._protos[target]  # KeyError on unknown target
        emb = np.asarray(self._encode(win), dtype=np.float32).reshape(-1)
        en = np.linalg.norm(emb)
        if en < 1e-12:
            return np.empty(0, dtype=np.float32)
        # Renormalize defensively: encoder output and prototypes are L2-normed, but a
        # not-quite-unit injected encode (or float32 export rounding) would otherwise
        # push cosines past +-1.0. Drop any zero-norm prototype rows from the pool.
        pn = np.linalg.norm(protos, axis=1)
        keep = pn >= 1e-12
        if not np.any(keep):
            return np.empty(0, dtype=np.float32)
        protos, pn = protos[keep], pn[keep]
        # Clamp to [-1, 1]: float32 rounding can push a near-unit cosine just past the
        # bounds, which would make strength leave [0, 1] or best_distance go negative.
        return np.clip((protos @ emb) / (pn * en), -1.0, 1.0).astype(np.float32)

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
        """Debug: the k closest glosses by best-per-gloss cosine distance (like
        Matcher.rank). One entry per gloss, ranked by its nearest prototype."""
        scored = sorted((self.best_distance(window, g), g) for g in self._protos)
        return [{"gloss": g, "distance": round(d, 3)} for d, g in scored[:k]]

    @classmethod
    def from_files(cls, onnx_path, prototypes_path):
        """Build an onnxruntime-backed matcher from an exported encoder + an .npz
        of prototypes (parallel arrays `glosses` and `protos`, ONE ROW PER PROTOTYPE
        so a gloss may repeat for its k phase prototypes)."""
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
        return cls(encode, protos)
