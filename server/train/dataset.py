"""Training-data assembly for the motion encoder.

A *sample* is a dict {"gloss": str, "signer_id": hashable, "seq": (T, 84) float32}.
`signer_split` keeps every signer entirely on one side of the train/val split so
validation measures cross-signer generalization (the thing that matters live) and
ASSERTS no signer leaks across the split. `sample_window` draws a fixed-L window
with a random start (matching the live 16-frame sliding window); `SequenceDataset`
trains on those windows (plus some shorter T<L warmup-fill windows). `build_cache`
turns WLASL clips into samples via the existing extraction pipeline, joining
WLASL_v0.3.json on video id to attach a real signer_id, and writes an .npz (not run
in tests). `build_library_cache` is the glue from `build.build_library`'s gloss
catalogue to `build_cache`.
"""
import numpy as np
from torch.utils.data import Dataset

from asl.schema import match_features
from train.augment import augment

# Live feeds the encoder a 16-frame sliding window; training/enroll must sample
# windows of the SAME length (random start) so the encoder never sees a tempo or
# coverage at train time that it won't see live.
WINDOW_SIZE = 16

# How often the train side draws a *short* (T<L) window, mimicking the live
# warmup-fill phase where the buffer hasn't filled to L frames yet.
SHORT_WINDOW_PROB = 0.2

# dxli94 WLASL metadata carries a `signer_id` per video; Voxel51/WLASL does not.
# Join by video id to recover cross-signer structure for the validation split.
WLASL_V03_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)


def sample_window(seq: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a contiguous window of up to L frames from `seq` with a random start.

    Mirrors the live sliding window: live feeds the encoder fixed-L windows, so
    training and enrollment must sample fixed-L windows too. If T <= L the whole
    sequence is returned (this is also how a real T<L warmup-fill window looks).
    Always returns at least one frame.
    """
    seq = np.asarray(seq, dtype=np.float32)
    L = max(1, int(L))
    T = seq.shape[0]
    if T <= L:
        return seq.copy()
    start = int(rng.integers(0, T - L + 1))
    return seq[start:start + L].astype(np.float32)


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
    # Contract: with >=2 signers no signer may appear on both sides; assert it so a
    # regression that leaks signers (re-introducing the DTW blindness) fails loudly.
    train_signers = {s["signer_id"] for s in train}
    val_signers = {s["signer_id"] for s in val}
    assert train_signers.isdisjoint(val_signers), (
        "signer_split leaked a signer across train/val: "
        f"{train_signers & val_signers}"
    )
    return train, val


class SequenceDataset(Dataset):
    """Yields (window (<=window, 84) float32, label int).

    Labels are assigned from the sorted set of glosses present in `samples`, so
    the same gloss maps to the same index regardless of split. Both sides sample a
    fixed-L window (random start) so the encoder trains on the SAME window length
    it sees live; on the train side the sequence is also augmented, and a fraction
    of items are drawn as shorter T<L windows that mimic the live warmup-fill
    phase. On val the window is sampled deterministically (no augmentation).
    """

    def __init__(self, samples, train: bool, window: int = WINDOW_SIZE, seed: int = 0):
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
        """Reduce to the 84-wide hand-xy features, (optionally) augment, then draw
        a fixed-L window with a random start (matching live), occasionally a
        shorter warmup-fill window on the train side."""
        seq = np.asarray(seq, dtype=np.float32)
        # match_features is idempotent on already-84-wide seqs but enforces the
        # contract for raw (T, 147) sequences.
        if seq.ndim == 2 and seq.shape[1] == 84:
            feat = seq
        else:
            feat = match_features(seq).astype(np.float32)
        L = self.window
        if self.train:
            feat = augment(feat, self._rng)
            # Some windows mimic the live warmup phase, where the rolling buffer
            # has fewer than L frames; sampling a shorter window teaches the
            # encoder those partial-fill windows.
            if self._rng.random() < SHORT_WINDOW_PROB:
                L = int(self._rng.integers(2, self.window + 1))
        feat = sample_window(feat, L, self._rng)
        if feat.shape[0] == 0:
            feat = np.zeros((1, 84), dtype=np.float32)
        return feat.astype(np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        window = self._to_window(s["seq"])
        label = self.label_of[s["gloss"]]
        return window, label


def _signer_id_by_video(cache_dir=None) -> dict:
    """Map WLASL video_id (str) -> signer_id by fetching dxli94's WLASL_v0.3.json.

    Voxel51/WLASL carries no signer_id, so the planned signer-split silently
    degrades to a random split (the blindness that killed DTW). dxli94's metadata
    has a real signer_id per instance, keyed by the same video id Voxel51 names
    its clips by. NOT run in unit tests (it hits the network).
    """
    import json
    import os
    from urllib.request import urlopen

    cache_dir = cache_dir or os.environ.get("WLASL_META_DIR", ".")
    local = os.path.join(cache_dir, "WLASL_v0.3.json")
    if os.path.exists(local):
        with open(local, "r") as fh:
            entries = json.load(fh)
    else:
        with urlopen(WLASL_V03_URL) as resp:  # noqa: S310 (trusted constant URL)
            entries = json.loads(resp.read().decode("utf-8"))
        try:
            with open(local, "w") as fh:
                json.dump(entries, fh)
        except OSError:
            pass  # caching is best-effort; the join works without it
    out = {}
    for entry in entries:
        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            if vid is not None and "signer_id" in inst:
                out[str(vid)] = inst["signer_id"]
    return out


def _extract_one(args):
    """Worker: extract one clip -> (gloss, signer_id, (T,84) seq) or None. Runs in
    its own process with its own MediaPipe instance (spawn-safe; imports lazily)."""
    gloss, video_id, path, signer_id = args
    try:
        from asl.features import normalize_frame
        from build.extract import extract_frames

        normed = []
        for f in extract_frames(path):
            try:
                normed.append(normalize_frame(f).flatten())
            except ValueError:
                continue
        if len(normed) < 2:
            return None
        seq = match_features(np.array(normed, dtype=np.float32)).astype(np.float32)
        return (gloss, str(signer_id), seq)
    except Exception:  # a bad clip must not sink the whole build
        return None


def build_cache(glosses, clip_paths, out_path, per_gloss=None, signer_by_video=None,
                workers=None):
    """Extract (T, 84) samples from WLASL clips and write them to an .npz.

    `clip_paths` maps gloss -> list of (video_id, video_path). The real signer_id
    is recovered by joining dxli94's WLASL_v0.3.json on video_id (fetched once via
    `_signer_id_by_video`, or passed in as `signer_by_video` for reuse/testing); a
    video with no match falls back to its own video_id as a unique signer so it is
    never silently merged with another signer. Extraction is parallelized across
    `workers` processes (default min(cpu,8)) in crash-tolerant batches so a full
    ~2000-gloss build survives OOMs/bad clips. NOT run in unit tests.
    """
    import json
    import multiprocessing as mp
    import os
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    if signer_by_video is None:
        signer_by_video = _signer_id_by_video()

    tasks = []
    for gloss in glosses:
        entries = clip_paths.get(gloss, [])
        if per_gloss is not None:
            entries = entries[:per_gloss]
        for video_id, path in entries:
            # Join on video id; fall back to the (unique) video id itself so an
            # unmatched clip is its own signer rather than collapsing into one bucket.
            signer_id = signer_by_video.get(str(video_id), f"vid:{video_id}")
            tasks.append((gloss, video_id, path, signer_id))

    total = len(tasks)
    samples = []          # {gloss, signer_id, seq, video_id}
    done_ids = set()

    # RESUME: if an interrupted run left a cache carrying video_ids, keep its samples
    # and skip those clips so the (expensive) MediaPipe extraction isn't redone.
    if os.path.exists(out_path):
        try:
            prev = np.load(out_path, allow_pickle=True)
            if "video_ids" in prev:
                for seq, g, sid, vid in zip(prev["seqs"], prev["glosses"],
                                            prev["signer_ids"], prev["video_ids"]):
                    samples.append({"gloss": str(g), "signer_id": str(sid),
                                    "seq": np.asarray(seq, dtype=np.float32),
                                    "video_id": str(vid)})
                    done_ids.add(str(vid))
                print(f"resuming: {len(done_ids)} clips already extracted, skipping them")
        except Exception:
            samples, done_ids = [], set()  # corrupt/legacy cache -> start fresh

    pending = [t for t in tasks if str(t[1]) not in done_ids]

    def _flush():
        np.savez(
            out_path,
            seqs=np.array([s["seq"] for s in samples], dtype=object),
            glosses=np.array([s["gloss"] for s in samples]),
            signer_ids=np.array([str(s["signer_id"]) for s in samples]),
            video_ids=np.array([str(s["video_id"]) for s in samples]),
            meta=json.dumps({"feature_dim": 84}),
        )

    def _collect(task, r):
        if r is not None:
            g, sid, seq = r
            samples.append({"gloss": g, "signer_id": sid, "seq": seq,
                            "video_id": str(task[1])})

    if not workers or workers <= 1:
        # Serial / in-process: keeps the in-process extraction monkeypatch working in
        # unit tests, and is the simple path for small runs.
        for t in pending:
            _collect(t, _extract_one(t))
        _flush()
    else:
        # Parallel for real (multi-thousand-clip) runs: crash-tolerant batches with a
        # recycling spawn pool, CHECKPOINTED after every batch so a crash/Ctrl-C only
        # loses the current batch (MediaPipe holds native state that won't fork).
        print(f"extracting {len(pending)} clips ({len(done_ids)} cached) "
              f"with {workers} workers...")
        ctx = mp.get_context("spawn")
        pool_kw = {"max_workers": workers, "mp_context": ctx}
        if sys.version_info >= (3, 11):  # max_tasks_per_child is 3.11+ only
            pool_kw["max_tasks_per_child"] = 10
        batch = max(workers * 8, 16)
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            try:
                with ProcessPoolExecutor(**pool_kw) as ex:
                    futmap = {ex.submit(_extract_one, t): t for t in chunk}
                    for fut in as_completed(futmap):
                        try:
                            _collect(futmap[fut], fut.result())
                        except Exception:
                            pass
            except BrokenProcessPool:
                print(f"  ! batch crashed near {start}/{len(pending)} (likely OOM — "
                      f"lower --workers); skipping it")
            _flush()  # checkpoint -> resumable
            print(f"  [{min(start + batch, len(pending))}/{len(pending)} new | "
                  f"{len(samples)}/{total} total] checkpointed -> {out_path}")
    return samples


def build_library_cache(glosses, out_path, per_gloss=8, workers=None, clips_dir=None):
    """Glue from build.build_library's gloss catalogue to build_cache.

    build_library knows how to enumerate WLASL clips (Voxel51/WLASL samples.json:
    gloss -> [filepath]) and how to fetch a clip, but nothing wired that catalogue
    into build_cache. This resolves each gloss to (video_id, local_clip_path)
    entries (downloading via hf_hub_download, video_id = the filepath stem, which
    is what build_library names templates by AND the key WLASL_v0.3.json joins on)
    and hands them to build_cache. NOT run in unit tests (network + MediaPipe).

    When `clips_dir` is set, the FIRST usable downloaded clip per gloss is also
    copied there as `<gloss>.mp4` (the lowercase single-token gloss == the HUD
    slug) so be-server can serve a reference example clip for that gloss. With
    `clips_dir=None` (the default) nothing is copied — the cache build is unchanged.
    """
    import json
    import shutil
    from collections import defaultdict
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from build.build_library import REPO

    samples = json.loads(
        Path(hf_hub_download(REPO, "samples.json", repo_type="dataset")).read_text()
    )["samples"]
    by_gloss: dict = defaultdict(list)
    for s in samples:
        by_gloss[s["gloss"]["label"]].append(s["filepath"])

    if glosses is None:
        glosses = sorted(by_gloss)

    if clips_dir is not None:
        Path(clips_dir).mkdir(parents=True, exist_ok=True)

    clip_paths: dict = {}
    for gloss in glosses:
        entries = []
        for fp in by_gloss.get(gloss, [])[:per_gloss]:
            try:
                clip = hf_hub_download(REPO, fp, repo_type="dataset")
            except Exception as exc:  # a bad clip must not sink the whole build
                print(f"  skip {fp}: {exc}")
                continue
            entries.append((Path(fp).stem, clip))
        clip_paths[gloss] = entries
        # Copy the first usable downloaded clip as the gloss's reference example so
        # be-server can serve GET /clips/<gloss>.mp4 (the HUD's example video).
        if clips_dir is not None and entries:
            dest = Path(clips_dir) / f"{gloss.lower()}.mp4"
            try:
                shutil.copy(entries[0][1], dest)
            except OSError as exc:  # a copy failure must not sink the build
                print(f"  skip clip copy for {gloss}: {exc}")

    return build_cache(glosses, clip_paths, out_path, per_gloss=per_gloss, workers=workers)


if __name__ == "__main__":
    # Clips -> landmark-sequence cache: `python -m train.dataset --out cache.npz`.
    # Feeds train.py (`--cache`). Network + MediaPipe; run in the Docker image.
    import argparse

    p = argparse.ArgumentParser(description="Build the WLASL landmark-sequence cache.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--glosses", nargs="*", default=None, help="explicit gloss subset")
    g.add_argument("--all", action="store_true", help="every gloss in the dataset")
    p.add_argument("--out", default="data/cache.npz")
    p.add_argument("--per-gloss", type=int, default=8)
    p.add_argument("--clips-dir", default="data/clips",
                   help="copy the first usable clip per gloss here as <gloss>.mp4 "
                        "(be-server serves these as /clips/<gloss>.mp4); '' to skip")
    import os as _os
    p.add_argument("--workers", type=int, default=min(_os.cpu_count() or 4, 8),
                   help="parallel extraction workers (default min(cpu,8); ~0.5-1GB each)")
    a = p.parse_args()
    build_library_cache(None if a.all else a.glosses, a.out,
                        per_gloss=a.per_gloss, workers=a.workers,
                        clips_dir=(a.clips_dir or None))
