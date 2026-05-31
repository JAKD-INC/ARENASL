"""Learned motion-embedding matcher — drop-in for asl.matcher.Matcher.

Scores a live window against a target sign's prototype by cosine of their motion
embeddings, mapped to [0, 1]. Same (window, target) -> strength interface as
Matcher, so session/app swap one for the other. The encoder is injected as a
callable so this is unit-testable WITHOUT onnx (tests pass a plain python encode
fn + synthetic prototypes); `from_files` builds the onnxruntime-backed encoder for
production.
"""
import numpy as np


class EmbeddingMatcher:
    def __init__(self, encode, prototypes):
        """
        Args:
            encode: callable (window (T, 84) float32) -> L2-normed embedding (emb_dim,).
            prototypes: {gloss: prototype (emb_dim,) np.ndarray}, L2-normalized.
        """
        self._encode = encode
        self._protos = {g: np.asarray(p, dtype=np.float32) for g, p in prototypes.items()}

    def _cos(self, window: np.ndarray, target: str):
        """Cosine of the window's embedding against `target`'s prototype.

        Returns None for a degenerate (zero-norm) embedding or prototype, which the
        callers map to the worst score rather than a neutral one.
        """
        proto = self._protos[target]  # KeyError on unknown target
        emb = np.asarray(self._encode(np.asarray(window, dtype=np.float32)), dtype=np.float32)
        # Encoder output and prototypes are L2-normed; renormalize defensively so a
        # not-quite-unit injected encode still yields a true cosine in [-1, 1].
        en, pn = np.linalg.norm(emb), np.linalg.norm(proto)
        if en < 1e-12 or pn < 1e-12:
            return None
        # Clamp to [-1, 1]: float32 rounding (notably from onnxruntime exports) can
        # push a near-unit cosine just past +-1.0, which would otherwise make
        # strength fall outside [0, 1] or best_distance go negative.
        return float(np.clip(np.dot(emb, proto) / (en * pn), -1.0, 1.0))

    def strength(self, window: np.ndarray, target: str) -> float:
        """Match strength in [0, 1]: (cosine + 1) / 2. A degenerate (zero) embedding
        scores 0.0 (worst) rather than the neutral 0.5 a cosine of 0 would give."""
        cos = self._cos(window, target)
        if cos is None:
            return 0.0
        return (cos + 1.0) / 2.0

    def best_distance(self, window: np.ndarray, target: str) -> float:
        """Cosine distance 1 - cosine, in [0, 2] (parity with Matcher.best_distance,
        which exposes a non-negative distance). A degenerate (zero) embedding gets
        the worst distance 2.0, consistent with strength()==0.0."""
        cos = self._cos(window, target)
        if cos is None:
            return 2.0
        return 1.0 - cos

    def rank(self, window: np.ndarray, k: int = 3) -> list:
        """Debug: the k closest glosses by cosine distance (like Matcher.rank)."""
        scored = sorted((self.best_distance(window, g), g) for g in self._protos)
        return [{"gloss": g, "distance": round(d, 3)} for d, g in scored[:k]]

    @classmethod
    def from_files(cls, onnx_path, prototypes_path):
        """Build an onnxruntime-backed matcher from an exported encoder + an .npz
        of prototypes (arrays `glosses` and `protos`)."""
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
        # save_prototypes writes `glosses` and `protos` parallel-indexed; guard
        # against corruption that would silently mismatch a gloss to a prototype.
        if protos_mat.shape[0] != len(glosses):
            raise ValueError(
                f"prototypes file mismatch: {len(glosses)} glosses but "
                f"{protos_mat.shape[0]} prototype rows"
            )
        protos = {g: protos_mat[i] for i, g in enumerate(glosses)}
        return cls(encode, protos)
