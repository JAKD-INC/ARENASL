#!/usr/bin/env bash
# Get + train the FULL WLASL vocabulary end to end: download every gloss, extract
# MediaPipe landmarks (parallel), train the encoder on the GPU, enroll prototypes.
#
# Run from the repo (it cd's into server/ itself):
#   bash server/train/run_full.sh
#
# Tunables (env vars):
#   WORKERS    parallel extraction procs (default = CPU count; ~0.5-1GB RAM each)
#   PER_GLOSS  clips per gloss to extract (default 12)
#   EPOCHS     training epochs (default 60)
#   GLOSSES    space-separated subset instead of --all (e.g. GLOSSES="book eat go")
#   OUT        cache path (default data/cache_full.npz)
#
# Prereqs: server/.venv with cu121 torch + mediapipe + deps (see server/TRAINING.md),
# the system GL libs, and MediaPipe models in ../public/models. This is the LONG run
# (~16k clips) — expect a download- and CPU-bound extraction phase, then fast GPU
# training. Re-running skips nothing; delete data/cache_full.npz to force re-extract.
set -euo pipefail

cd "$(dirname "$0")/.."          # -> server/
PY=.venv/bin/python
WORKERS="${WORKERS:-$(nproc)}"
PER_GLOSS="${PER_GLOSS:-12}"
EPOCHS="${EPOCHS:-60}"
OUT="${OUT:-data/cache_full.npz}"
ENC=data/encoder.onnx
PROTO=data/prototypes.npz

[ -x "$PY" ] || { echo "ERROR: $PWD/$PY not found — create the venv per server/TRAINING.md"; exit 1; }
mkdir -p data

# HF token (raises rate limits for the ~16k-clip download): from .env or the env.
if [ -z "${HF_TOKEN:-}" ] && [ -f ../.env ]; then
  set -a; . ../.env; set +a
fi
[ -n "${HF_TOKEN:-}" ] && echo "[run_full] using HF_TOKEN (rate limits raised)" \
                       || echo "[run_full] no HF_TOKEN — public dataset works but may rate-limit"

# gloss selection: --all unless GLOSSES is set
if [ -n "${GLOSSES:-}" ]; then GSEL=(--glosses ${GLOSSES}); else GSEL=(--all); fi

echo "[run_full] 1/3 extract  (workers=$WORKERS per_gloss=$PER_GLOSS) -> $OUT"
if [ -f "$OUT" ]; then
  echo "  $OUT exists; skipping extraction (delete it to re-extract)"
else
  time PYTHONPATH=. "$PY" -m train.dataset "${GSEL[@]}" --per-gloss "$PER_GLOSS" --workers "$WORKERS" --out "$OUT"
fi

echo "[run_full] 2/3 train    (epochs=$EPOCHS, GPU if available) -> $ENC"
time PYTHONPATH=. "$PY" -m train.train --cache "$OUT" --onnx "$ENC" --epochs "$EPOCHS"

echo "[run_full] 3/3 enroll   -> $PROTO"
time PYTHONPATH=. "$PY" -m train.enroll --cache "$OUT" --onnx "$ENC" --out "$PROTO"

echo "[run_full] DONE. Artifacts:"
ls -la "$OUT" "$ENC" "$PROTO"
echo "[run_full] The server auto-uses these (asl_matcher_mode=auto). Run it with:"
echo "    JWT_SECRET=dev TURN_SECRET=dev PYTHONPATH=. $PY -m uvicorn app.main:app --port 8001"
