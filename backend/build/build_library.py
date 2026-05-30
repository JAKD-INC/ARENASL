"""Build reference templates + display clips from a WLASL gloss subset (HF)."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
from asl.features import normalize_frame
from build.extract import extract_frames

REPO = "Voxel51/WLASL"

# A starter subset of visually-distinct, common signs. Override with --glosses.
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


def build(glosses, out_dir, clips_dir, per_gloss):
    out_dir, clips_dir = Path(out_dir), Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    samples_path = hf_hub_download(REPO, "samples.json", repo_type="dataset")
    samples = json.loads(Path(samples_path).read_text())["samples"]
    by_gloss: dict[str, list[str]] = {}
    for s in samples:
        g = s["gloss"]["label"]
        if g in glosses:
            by_gloss.setdefault(g, []).append(s["filepath"])

    for gloss in glosses:
        filepaths = by_gloss.get(gloss, [])[:per_gloss]
        if not filepaths:
            print(f"WARN: no clips for gloss {gloss!r}; skipping")
            continue
        kept = 0
        for idx, fp in enumerate(filepaths):
            clip = hf_hub_download(REPO, fp, repo_type="dataset")
            seq = _normalize_sequence(extract_frames(clip))
            if len(seq) < 2:
                print(f"WARN: {gloss} clip {idx} yielded <2 usable frames; skipping")
                continue
            np.save(out_dir / f"{gloss}__{kept}.npy", seq)
            if kept == 0:  # first usable clip becomes the overlay display clip
                shutil.copy(clip, clips_dir / f"{gloss}.mp4")
            kept += 1
            print(f"{gloss}: clip {idx} -> {len(seq)} frames (template {kept})")
        if kept == 0:
            print(f"WARN: gloss {gloss!r} produced no usable templates")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glosses", nargs="*", default=DEFAULT_GLOSSES)
    p.add_argument("--out", default="data/templates")
    p.add_argument("--clips", default="../public/clips")
    p.add_argument("--per-gloss", type=int, default=8)
    a = p.parse_args()
    build(a.glosses, a.out, a.clips, a.per_gloss)
