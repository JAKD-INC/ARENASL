"""Build reference templates + display clips from WLASL (HuggingFace).

Clip extraction is parallelized across processes — each worker downloads its
clip and runs MediaPipe independently. The pool is run in memory-bounded batches
and tolerates worker crashes (a dead worker skips its batch instead of aborting
the whole build), so a full ~2000-gloss run survives OOMs and bad clips.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from asl.features import normalize_frame
from build.extract import extract_frames

REPO = "Voxel51/WLASL"

# A starter subset of visually-distinct, common signs. Override with --glosses
# (or --all for the entire WLASL vocabulary present in the dataset).
DEFAULT_GLOSSES = [
    "book", "computer", "drink", "go", "help", "eat", "water", "name",
    "learn", "teacher", "family", "friend", "happy", "work", "play",
]


def _normalize_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """Normalize each frame (drop frames with degenerate shoulders), flatten."""
    out = []
    for f in frames:
        try:
            out.append(normalize_frame(f).flatten())
        except ValueError:
            continue  # degenerate shoulders -> unusable frame
    return np.array(out)


def _process_clip(args):
    """Worker: download one clip, extract + normalize. Returns (gloss, fp, seq,
    clip_path) with seq=None if it yielded too few usable frames. Runs in its
    own process with its own MediaPipe instance."""
    gloss, filepath = args
    try:
        clip = hf_hub_download(REPO, filepath, repo_type="dataset")
        seq = _normalize_sequence(extract_frames(clip))
    except Exception as exc:  # a bad clip must not sink the whole build
        return (gloss, filepath, None, f"error: {exc}")
    if len(seq) < 2:
        return (gloss, filepath, None, "too few usable frames")
    return (gloss, filepath, seq, clip)


def build(glosses, out_dir, clips_dir, per_gloss, workers):
    out_dir, clips_dir = Path(out_dir), Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(
        Path(hf_hub_download(REPO, "samples.json", repo_type="dataset")).read_text()
    )["samples"]
    by_gloss: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_gloss[s["gloss"]["label"]].append(s["filepath"])

    if glosses is None:  # --all
        glosses = sorted(by_gloss)

    tasks = [
        (gloss, fp)
        for gloss in glosses
        for fp in by_gloss.get(gloss, [])[:per_gloss]
    ]
    missing = [g for g in glosses if not by_gloss.get(g)]
    if missing:
        print(f"WARN: {len(missing)} gloss(es) absent from dataset: {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))
    total = len(tasks)
    print(f"Extracting {total} clips across {len(glosses)} glosses "
          f"with {workers} workers (~0.5-1GB RAM each)...")

    # spawn (not fork): MediaPipe holds native/GL state that doesn't survive fork.
    ctx = mp.get_context("spawn")
    have_display = set()       # glosses with a display clip copied
    written = 0
    # Run in batches with a fresh, recycling pool per batch. This bounds memory
    # and isolates worker crashes: an OOM/segfault aborts only the current batch.
    batch_size = max(workers * 8, 16)
    for start in range(0, total, batch_size):
        batch = tasks[start:start + batch_size]
        try:
            _pool_kw = {"max_workers": workers, "mp_context": ctx}
            if sys.version_info >= (3, 11):  # max_tasks_per_child is 3.11+ only
                _pool_kw["max_tasks_per_child"] = 10
            with ProcessPoolExecutor(**_pool_kw) as ex:
                futures = {ex.submit(_process_clip, t): t for t in batch}
                for fut in as_completed(futures):
                    try:
                        gloss, fp, seq, info = fut.result()
                    except Exception as exc:
                        print(f"  worker died on {futures[fut][1]}: {exc}")
                        continue
                    if seq is None:
                        continue
                    # Name by video id (deterministic) so re-runs overwrite rather
                    # than create duplicate templates. load_templates groups on the
                    # text before the last '__', i.e. the gloss.
                    vid = Path(fp).stem
                    np.save(out_dir / f"{gloss}__{vid}.npy", seq)
                    written += 1
                    if gloss not in have_display:  # first usable clip -> overlay clip
                        shutil.copy(info, clips_dir / f"{gloss}.mp4")
                        have_display.add(gloss)
        except BrokenProcessPool:
            print(f"  ! batch crashed near {start}/{total} (likely OOM — lower "
                  f"--workers). Skipping it and continuing.")
        print(f"  [{min(start + batch_size, total)}/{total}] "
              f"{written} templates, {len(have_display)} glosses ready")

    print(f"Done: {written} templates across {len(have_display)} glosses.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--glosses", nargs="*", default=None,
                   help="explicit gloss subset (default: a 15-sign starter set)")
    g.add_argument("--all", action="store_true",
                   help="build every gloss present in the dataset (~2000)")
    p.add_argument("--out", default="data/templates")
    p.add_argument("--clips", default="../public/clips")
    p.add_argument("--per-gloss", type=int, default=8)
    # Capped at 16: each worker holds a full MediaPipe stack (~0.5-1GB), so
    # oversubscribing cores OOMs. Batches + crash-tolerance keep a too-high count
    # from aborting the run, but stay within RAM. Override with --workers.
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16))
    a = p.parse_args()
    if a.all:
        chosen = None
    elif a.glosses is not None:
        chosen = a.glosses
    else:
        chosen = DEFAULT_GLOSSES
    build(chosen, a.out, a.clips, a.per_gloss, a.workers)
