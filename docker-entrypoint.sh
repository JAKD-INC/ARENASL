#!/bin/sh
set -e

# On first run the templates volume is empty, so `docker compose up app` would
# crash (the server loads templates at import). Build a default WLASL subset
# once so the app is self-sufficient; the volume persists, so later starts are
# fast. The standalone `builder` service (its command starts with `python`, not
# `uvicorn`) bypasses this and builds explicitly with whatever args it was given.
if [ "$1" = "uvicorn" ] && ! ls /data/templates/*.npy >/dev/null 2>&1; then
  echo "[entrypoint] No WLASL templates in /data/templates."
  echo "[entrypoint] Building the default subset (first run; downloads from HuggingFace, ~minutes)..."
  # Tunable without overriding the command: ASL_BUILD_GLOSSES="book drink help"
  # and ASL_BUILD_PER_GLOSS=4 (defaults: the builder's 15-gloss subset, 8 each).
  python -m build.build_library --out /data/templates --clips /data/clips \
    --per-gloss "${ASL_BUILD_PER_GLOSS:-8}" \
    ${ASL_BUILD_GLOSSES:+--glosses ${ASL_BUILD_GLOSSES}}
fi

exec "$@"
