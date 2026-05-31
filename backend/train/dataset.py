"""Training-data assembly for the motion encoder.

A *sample* is a dict {"gloss": str, "signer_id": hashable, "seq": (T, 84) float32}.
`signer_split` keeps every signer entirely on one side of the train/val split so
validation measures cross-signer generalization (the thing that matters live).
`SequenceDataset` yields (window (<=window, 84) float32, label int) pairs,
augmenting on the train side. `build_cache` turns WLASL clips into samples via the
existing extraction pipeline and writes an .npz (not run in tests).
"""
import numpy as np
from torch.utils.data import Dataset

from asl.schema import match_features
from train.augment import augment, temporal_crop


def signer_split(samples, val_frac: float = 0.2):
    """Split samples into (train, val) by signer_id so no signer is in both.

    Signers are assigned to val until ~val_frac of *samples* are held out;
    assignment is deterministic (sorted by signer id) so runs are reproducible.

    Both sides are guaranteed non-empty whenever that is possible:
      * With >=2 distinct signers, at least one whole signer lands on each side
        (the no-signer-on-both-sides contract is preserved).
      * With exactly 1 signer there is no way to honor both invariants, so that
        signer's samples are split across the two sides as a last resort (the
        no-both-sides contract is unsatisfiable with a single signer).
    Empty input returns ([], []).
    """
    by_signer: dict = {}
    for s in samples:
        by_signer.setdefault(s["signer_id"], []).append(s)
    if not samples:
        return [], []
    # Deterministic order; signers with id None are treated as their own bucket.
    order = sorted(by_signer, key=lambda x: (x is None, str(x)))

    # Single signer: both invariants cannot both hold, so fall back to a
    # sample-level split that keeps each side non-empty when there is more than
    # one sample. With a lone sample even that is impossible, so keep it on the
    # train side (never hand back an empty train set).
    if len(order) == 1:
        group = by_signer[order[0]]
        if len(group) == 1:
            return list(group), []
        n_val = max(1, int(round(len(group) * val_frac)))
        n_val = min(n_val, len(group) - 1)
        return list(group[n_val:]), list(group[:n_val])

    total = len(samples)
    target_val = int(round(total * val_frac))
    train, val = [], []
    val_signers: list = []
    n_val = 0
    for signer in order:
        group = by_signer[signer]
        # Send a signer to val while we are still short of the target and doing so
        # would not swallow every signer (always leave at least one for train).
        if n_val < target_val and len(val_signers) < len(order) - 1:
            val.extend(group)
            val_signers.append(signer)
            n_val += len(group)
        else:
            train.extend(group)
    # Guarantee both sides non-empty: if nothing reached val, hold out the
    # smallest signer group.
    if not val:
        smallest = min(order, key=lambda sg: len(by_signer[sg]))
        moved = by_signer[smallest]
        val.extend(moved)
        train = [s for s in train if s["signer_id"] != smallest]
    return train, val


class SequenceDataset(Dataset):
    """Yields (window (<=window, 84) float32, label int).

    Labels are assigned from the sorted set of glosses present in `samples`, so
    the same gloss maps to the same index regardless of split. On the train side
    each item is augmented and then cropped to at most `window` frames; on val the
    raw sequence is used (capped at `window`).
    """

    def __init__(self, samples, train: bool, window: int = 64, seed: int = 0):
        self.samples = list(samples)
        self.train = train
        self.window = int(window)
        self.glosses = sorted({s["gloss"] for s in self.samples})
        self.label_of = {g: i for i, g in enumerate(self.glosses)}
        self._rng = np.random.default_rng(seed)

    @property
    def num_classes(self) -> int:
        return len(self.glosses)

    def __len__(self) -> int:
        return len(self.samples)

    def _to_window(self, seq: np.ndarray) -> np.ndarray:
        """match_features is idempotent on already-84-wide seqs but enforces the
        contract; then cap/crop to `window` frames."""
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim == 2 and seq.shape[1] == 84:
            feat = seq
        else:
            feat = match_features(seq).astype(np.float32)
        T = feat.shape[0]
        if self.train:
            feat = augment(feat, self._rng)
            T = feat.shape[0]
        if T > self.window:
            feat = temporal_crop(feat, self.window, self._rng)
        if feat.shape[0] == 0:
            feat = np.zeros((1, 84), dtype=np.float32)
        return feat.astype(np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        window = self._to_window(s["seq"])
        label = self.label_of[s["gloss"]]
        return window, label


def build_cache(glosses, clip_paths, out_path, per_gloss=None):
    """Extract (T, 84) samples from WLASL clips and write them to an .npz.

    `clip_paths` maps gloss -> list of (signer_id, video_path). Uses
    build.extract.extract_frames + features.normalize_frame + match_features so
    training/enroll/live all share identical features. NOT run in unit tests.
    """
    import json
    from asl.features import normalize_frame
    from build.extract import extract_frames

    samples = []
    for gloss in glosses:
        entries = clip_paths.get(gloss, [])
        if per_gloss is not None:
            entries = entries[:per_gloss]
        for signer_id, path in entries:
            frames = extract_frames(path)
            normed = []
            for f in frames:
                try:
                    normed.append(normalize_frame(f).flatten())
                except ValueError:
                    continue
            if len(normed) < 2:
                continue
            seq = match_features(np.array(normed, dtype=np.float32)).astype(np.float32)
            samples.append({"gloss": gloss, "signer_id": signer_id, "seq": seq})

    # Store as object arrays of ragged sequences plus parallel label/signer arrays.
    np.savez(
        out_path,
        seqs=np.array([s["seq"] for s in samples], dtype=object),
        glosses=np.array([s["gloss"] for s in samples]),
        signer_ids=np.array([str(s["signer_id"]) for s in samples]),
        meta=json.dumps({"feature_dim": 84}),
    )
    return samples
