"""Enrollment: turn reference clips into per-gloss PHASE prototypes.

Open-vocab: adding a sign to the game means enrolling its reference clips, no
retraining. Live scores a sliding *window* (fixed length L, see
`dataset.WINDOW_SIZE`) by cosine to a prototype, so enrollment must build its
prototypes from the SAME windowed embeddings the live loop produces — not from a
single whole-clip embedding (the audit's "train/enroll on the live window length"
fix).

For each gloss:
  * each clip is sampled into several fixed-L windows (matching live), every
    window embedded by the trained encoder -> a cloud of windowed embeddings;
  * that cloud is clustered into k (2-3) PHASE prototypes (a sign passes through
    distinct hand configurations, so a mid-sign window matches whichever phase it
    landed in). `EmbeddingMatcher` takes the MAX cosine over a gloss's phase
    prototypes, so k prototypes raise recall without smearing distinct phases into
    one averaged blob.
  * prototypes are built by a robust medoid / trimmed-mean so a single bad clip or
    detector dropout does not drag a prototype off the manifold.

Data floor (audit fix #5 — ~43% of WLASL clips have rotted): a gloss needs at
least `data_floor` (default 8) USABLE clips for a trustworthy prototype; below
that we log and keep going, and a gloss with fewer than `min_clips` (default 3)
is REFUSED outright (dropped) because 1-2 clips cannot give a stable prototype.
"""
import logging

import numpy as np

try:  # canonical fixed-L windowing lives in dataset (shared with train.py)
    from train.dataset import WINDOW_SIZE, sample_window
except ImportError:  # pragma: no cover - resilience while dataset is co-edited
    WINDOW_SIZE = 16

    def sample_window(seq, L, rng):
        """Fallback fixed-L window sampler matching the shared contract:
        (seq (T,F), L:int, rng) -> (<=L, F). A clip shorter than L is returned
        whole (the warmup-fill T<L case); otherwise a random L-length slice."""
        seq = np.asarray(seq, dtype=np.float32)
        T = seq.shape[0]
        if T <= L:
            return seq
        start = int(rng.integers(0, T - L + 1))
        return seq[start:start + L]


log = logging.getLogger(__name__)


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return (v / n).astype(np.float32)


def _windowed_embeddings(encode, clip, *, window, n_windows, rng):
    """Embed `n_windows` fixed-L windows sampled from one clip -> (n, emb_dim).

    A clip shorter than the window still yields its (warmup-fill) embedding; the
    repeated random starts give the per-clip motion cloud the phase clustering
    works over.
    """
    clip = np.asarray(clip, dtype=np.float32)
    embs = []
    for _ in range(max(1, n_windows)):
        w = np.asarray(sample_window(clip, window, rng), dtype=np.float32)
        if w.shape[0] == 0:
            continue
        emb = np.asarray(encode(w), dtype=np.float32).reshape(-1)
        embs.append(emb)
    if not embs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(embs, axis=0)


def _trimmed_mean_unit(embs: np.ndarray, trim: float = 0.2) -> np.ndarray:
    """Medoid-anchored trimmed mean of unit embeddings -> one L2-normed prototype.

    Anchor on the medoid (the member with the highest summed cosine to the rest,
    i.e. the most central real embedding), drop the `trim` fraction farthest from
    it, then average the survivors. This is robust to a few outlier windows
    (detector dropouts, a clip that caught the wrong sign phase) in a way a plain
    mean is not.
    """
    embs = np.asarray(embs, dtype=np.float32)
    n = embs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    unit = np.stack([_l2norm(e) for e in embs], axis=0)
    if n <= 2:
        return _l2norm(unit.mean(axis=0))
    # Medoid: highest total cosine similarity to the others.
    sims = unit @ unit.T
    medoid = int(np.argmax(sims.sum(axis=1)))
    cos_to_medoid = unit @ unit[medoid]
    keep = max(1, int(round(n * (1.0 - trim))))
    order = np.argsort(-cos_to_medoid)  # most-central first
    chosen = order[:keep]
    return _l2norm(unit[chosen].mean(axis=0))


def _phase_prototypes(embs: np.ndarray, k: int, rng) -> list:
    """Cluster a gloss's windowed embeddings into <=k PHASE prototypes.

    A small spherical k-means (cosine) on the unit embeddings; each cluster's
    robust trimmed mean is one phase prototype. Falls back to a single prototype
    when there are too few embeddings to support k clusters. Empty clusters are
    dropped, so the result has 1..k prototypes.
    """
    embs = np.asarray(embs, dtype=np.float32)
    n = embs.shape[0]
    if n == 0:
        return []
    unit = np.stack([_l2norm(e) for e in embs], axis=0)
    k = int(max(1, k))
    if k == 1 or n <= k:
        # Not enough windows to meaningfully split into phases.
        return [_trimmed_mean_unit(unit)]

    # Spread (k-means++ style, cosine) init so the seeds land in DIFFERENT phases
    # instead of clumping in one — a plain random init can pick two co-located
    # windows and collapse a phase. First center random; each next center is the
    # window farthest (lowest max cosine) from the chosen ones.
    first = int(rng.integers(0, n))
    chosen = [first]
    for _ in range(1, k):
        max_cos = np.max(unit @ unit[chosen].T, axis=1)
        nxt = int(np.argmin(max_cos))
        if nxt in chosen:  # degenerate (all identical) — stop seeding
            break
        chosen.append(nxt)
    centers = unit[chosen].copy()
    k = centers.shape[0]

    # Spherical k-means: assign by cosine, recenter as the (renormalized) mean.
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(25):
        new_assign = np.argmax(unit @ centers.T, axis=1)
        if np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for c in range(k):
            members = unit[assign == c]
            if members.shape[0] > 0:
                centers[c] = _l2norm(members.mean(axis=0))

    protos = []
    for c in range(k):
        members = unit[assign == c]
        if members.shape[0] == 0:
            continue
        protos.append(_trimmed_mean_unit(members))
    return protos or [_trimmed_mean_unit(unit)]


def enroll(encode, refs, *, k: int = 3, window: int = WINDOW_SIZE,
           n_windows: int = 8, min_clips: int = 3, data_floor: int = 8,
           rng=None):
    """Build k phase prototypes per gloss from WINDOWED reference-clip embeddings.

    Args:
        encode: callable (window (<=L, 84) float32) -> L2-normed embedding (emb_dim,).
        refs: {gloss: [clip (T, 84), ...]}.
        k: target phase prototypes per gloss (2-3); fewer are returned for short
            clip pools.
        window: fixed window length L to sample (defaults to the live WINDOW_SIZE).
        n_windows: random fixed-L windows to sample per clip.
        min_clips: a gloss with fewer USABLE clips than this is REFUSED (dropped).
        data_floor: below this many usable clips the gloss is kept but logged as
            under-supported.
        rng: optional np.random.Generator for reproducible window sampling.

    Returns:
        {gloss: [prototype (emb_dim,) float32, ...]} — 1..k L2-normed phase
        prototypes per kept gloss. A gloss below `min_clips` usable clips is
        absent from the result.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for gloss, clips in refs.items():
        usable = [np.asarray(c, dtype=np.float32) for c in (clips or [])
                  if np.asarray(c).ndim == 2 and np.asarray(c).shape[0] >= 1]
        n_usable = len(usable)
        if n_usable < min_clips:
            log.warning("enroll: REFUSING gloss %r — only %d usable clip(s), "
                        "need >= %d", gloss, n_usable, min_clips)
            continue
        if n_usable < data_floor:
            log.warning("enroll: gloss %r under-supported — %d usable clip(s) "
                        "(< data floor %d); prototype may be noisy",
                        gloss, n_usable, data_floor)
        embs = [_windowed_embeddings(encode, c, window=window,
                                     n_windows=n_windows, rng=rng)
                for c in usable]
        embs = [e for e in embs if e.shape[0] > 0]
        if not embs:
            log.warning("enroll: gloss %r produced no embeddings; dropping", gloss)
            continue
        cloud = np.concatenate(embs, axis=0)
        protos = _phase_prototypes(cloud, k, rng)
        if protos:
            out[gloss] = protos
    return out


def _iter_prototype_rows(protos):
    """Normalize either return shape into (gloss, vector) rows, one per prototype.

    Accepts the phase-prototype shape {gloss: [vec, ...]} as well as the legacy
    single-vector shape {gloss: vec}, so callers that still build a single
    prototype keep working.
    """
    for gloss, val in protos.items():
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1:
            yield gloss, arr
        else:
            for row in arr:
                yield gloss, np.asarray(row, dtype=np.float32)


def save_prototypes(path, protos):
    """Write prototypes to an .npz with parallel arrays `glosses` (str, ONE ROW
    PER PROTOTYPE — a gloss repeats once per phase prototype) and `protos`
    (N, emb_dim) float32. `EmbeddingMatcher` groups the rows back by gloss.

    Rows are ordered by (gloss, prototype index) so the file is deterministic.
    """
    rows = sorted(_iter_prototype_rows(protos), key=lambda gp: gp[0])
    if rows:
        glosses = [g for g, _ in rows]
        matrix = np.stack([_l2norm(v) for _, v in rows], axis=0)
    else:
        glosses = []
        matrix = np.zeros((0, 0), dtype=np.float32)
    np.savez(path, glosses=np.array(glosses), protos=matrix.astype(np.float32))
    return path


if __name__ == "__main__":
    # Enroll prototypes from a landmark cache + a trained encoder:
    #   python -m train.enroll --cache data/cache.npz --onnx data/encoder.onnx \
    #       --out data/prototypes.npz
    # Uses the SAME cache as training (sequences are already (T,84) hand-xy), so no
    # re-extraction. Set asl_matcher_mode=embedding + asl_prototypes_path to use it.
    import argparse
    from collections import defaultdict

    import onnxruntime as ort

    from train.train import _load_cache

    p = argparse.ArgumentParser(description="Enroll prototypes from cache + encoder.")
    p.add_argument("--cache", required=True, help=".npz from `python -m train.dataset`")
    p.add_argument("--onnx", default="data/encoder.onnx")
    p.add_argument("--out", default="data/prototypes.npz")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    a = p.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def _encode(w):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[None, ...]  # (1, T, 84)
        return np.asarray(sess.run(None, {iname: w})[0], dtype=np.float32).reshape(-1)

    refs = defaultdict(list)
    for s in _load_cache(a.cache):
        refs[s["gloss"]].append(s["seq"])
    protos = enroll(_encode, dict(refs), k=a.k, window=a.window)
    save_prototypes(a.out, protos)
    print(f"enrolled {len(protos)} glosses -> {a.out}")
