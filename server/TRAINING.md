# Training the motion-embedding recognizer (on your GPU)

The game ships with DTW recognition by default. The **learned** path (per-window
motion embedding → cosine to per-gloss prototypes) is opt-in and needs three
artifacts in `data/`: a landmark **cache**, a trained **`encoder.onnx`**, and
**`prototypes.npz`**. Build them once, then point the server at them.

This run needs MediaPipe (system GL libs), a CUDA `torch`, and compute — so it runs
on your machine (RTX 3060), not in CI. Two paths; the native one is simplest on WSL.

## Pipeline (what the three commands do)
1. `train.dataset` → download a WLASL gloss subset from HuggingFace, run MediaPipe,
   write `(gloss, signer_id, (T,84) hand-xy sequence)` samples to `data/cache.npz`.
2. `train.train` → train the bi-GRU encoder on fixed-L windows (signer-split val),
   export `data/encoder.onnx`. **Uses CUDA automatically** (`--device` to override).
3. `train.enroll` → embed the cache's clips with the encoder → `data/prototypes.npz`
   (k phase prototypes per gloss; data floor ≥8 usable clips, refuses <3).

## Option A — native WSL venv (recommended; uses the 3060 directly)
```bash
sudo apt-get update && sudo apt-get install -y libgles2 libegl1 libgl1 libglib2.0-0
cd server
python3 -m venv .venv-train
.venv-train/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv-train/bin/pip install numpy "dtaidistance>=2.4" onnx onnxruntime \
    mediapipe opencv-python-headless huggingface_hub
# MediaPipe models (extract.py reads them from <repo>/public/models):
mkdir -p ../public/models
curl -fL -o ../public/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -fL -o ../public/models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

# Start SMALL to validate the gate before scaling vocab:
PYTHONPATH=. .venv-train/bin/python -m train.dataset --glosses book drink help eat water --per-gloss 12 --out data/cache.npz
PYTHONPATH=. .venv-train/bin/python -m train.train  --cache data/cache.npz --onnx data/encoder.onnx --epochs 40
PYTHONPATH=. .venv-train/bin/python -m train.enroll --cache data/cache.npz --onnx data/encoder.onnx --out data/prototypes.npz
```
`train.train` prints `training on device=cuda` and `val top1/top5` — that's the
first signal. The HONEST gate (per the design audit) is the next step: does it track
*your* live signing?

## Option B — Docker + GPU (needs nvidia-container-toolkit on the host)
```bash
docker build -f server/Dockerfile.train -t arenasl-train .
docker run --rm --gpus all -v "$PWD/data:/work/data" arenasl-train \
    python -m train.dataset --glosses book drink help eat water --per-gloss 12 --out data/cache.npz
docker run --rm --gpus all -v "$PWD/data:/work/data" arenasl-train \
    python -m train.train  --cache data/cache.npz --onnx data/encoder.onnx --epochs 40
docker run --rm --gpus all -v "$PWD/data:/work/data" arenasl-train \
    python -m train.enroll --cache data/cache.npz --onnx data/encoder.onnx --out data/prototypes.npz
```

## Validate live (the real gate)
Point the server at the artifacts and enable the learned path, then sign into the app
and watch the HUD (`strength`, `dist`, `top-3`):
```bash
ASL_MATCHER_MODE=embedding ASL_FEATURE_MODE=hands \
ASL_ENCODER_PATH=data/encoder.onnx ASL_PROTOTYPES_PATH=data/prototypes.npz \
ASL_RANK_EVERY=5 \
<run the server>
```
Success bar (audit): when you perform the prompted gloss, it should rise to the top-3
and its strength should clearly separate from idle/other signs. If it doesn't, the
fix is **hybrid per-user enrollment** (Phase 2): enroll prototypes from *your own*
calibration clips instead of WLASL strangers — that's the single biggest lever, and
`train.enroll` already works from any cache, so it'll plug straight in.

## Scaling
Once the small set validates, grow the vocab (`--glosses ... --per-gloss 8` or
`--all`) and re-run all three steps. Keep an eye on per-gloss clip counts — WLASL's
long tail has few usable clips; `enroll` refuses glosses with <3.
