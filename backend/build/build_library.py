"""Build reference templates + display clips from WLASL (HuggingFace).

Clip extraction is parallelized across processes — each worker downloads its
clip and runs MediaPipe independently, so a full ~2000-gloss build uses every
core instead of plodding one clip at a time.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    print(f"Extracting {len(tasks)} clips across {len(glosses)} glosses "
          f"with {workers} workers...")

    kept = defaultdict(int)        # gloss -> templates written so far
    have_display = set()           # glosses with a display clip copied
    done = 0
    # spawn (not fork): MediaPipe holds native/GL state that doesn't survive fork.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(_process_clip, t) for t in tasks]
        for fut in as_completed(futures):
            gloss, fp, seq, info = fut.result()
            done += 1
            if seq is None:
                continue
            idx = kept[gloss]
            np.save(out_dir / f"{gloss}__{idx}.npy", seq)
            kept[gloss] += 1
            if gloss not in have_display:  # first usable clip -> overlay clip
                shutil.copy(info, clips_dir / f"{gloss}.mp4")
                have_display.add(gloss)
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {len(have_display)} glosses ready")

    usable = sum(1 for g in glosses if kept[g])
    print(f"Done: {sum(kept.values())} templates across {usable}/{len(glosses)} glosses.")


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
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = p.parse_args()
    if a.all:
        chosen = None
    elif a.glosses is not None:
        chosen = a.glosses
    else:
        chosen = DEFAULT_GLOSSES
    build(chosen, a.out, a.clips, a.per_gloss, a.workers)
