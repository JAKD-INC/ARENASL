"""Enrollment: turn reference clips into per-gloss prototype embeddings.

Open-vocab: adding a sign to the game means enrolling its reference clips, no
retraining. Each clip -> embedding via the trained encoder; per gloss the
prototype is the L2-normalized mean of its clip embeddings, so cosine to the
prototype is the live match strength.
"""
import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def enroll(encode, refs):
    """Build prototypes from reference clips.

    Args:
        encode: callable (window (T, 84) float32) -> L2-normed embedding (emb_dim,).
        refs: {gloss: [seq (T, 84), ...]}.

    Returns:
        {gloss: prototype (emb_dim,) float32}, each the L2-normalized mean of its
        per-clip embeddings.
    """
    protos = {}
    for gloss, clips in refs.items():
        if not clips:
            continue
        embs = [np.asarray(encode(np.asarray(c, dtype=np.float32)), dtype=np.float32)
                for c in clips]
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        protos[gloss] = _l2norm(mean)
    return protos


def save_prototypes(path, protos):
    """Write prototypes to an .npz with parallel arrays `glosses` (str) and
    `protos` (G, emb_dim) float32, ordered to match."""
    glosses = sorted(protos)
    if glosses:
        matrix = np.stack([np.asarray(protos[g], dtype=np.float32) for g in glosses], axis=0)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    np.savez(path, glosses=np.array(glosses), protos=matrix.astype(np.float32))
    return path
