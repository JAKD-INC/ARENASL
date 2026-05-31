# syntax=docker/dockerfile:1
# Our game server (lobby/matchmaking/duel/replay + vendored ASL recognition).
# Linux base + a from-source build gives dtaidistance's compiled C DTW (its PyPI
# wheel is pure-Python on every platform). Build context is the repo root
# (see docker-compose `be-server`).

FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Toolchain for dtaidistance's C extension: build-essential = gcc,
# libgomp1 = OpenMP runtime (present on Linux; absent from mac clang).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Compile dtaidistance's fast C DTW FIRST, independent of our source, so it stays
# cached across code changes. Assert it loaded (fail the build, not ship slow).
# Install onnxruntime here too (server-side EmbeddingMatcher inference); keeping it
# in this pre-source layer means it stays cached across code changes.
RUN pip install setuptools wheel Cython "numpy>=1.26" "onnxruntime>=1.18" \
 && pip install --no-deps --no-build-isolation --no-binary dtaidistance "dtaidistance>=2.4" \
 && python -c "import numpy as np; from dtaidistance import dtw_ndim; dtw_ndim.distance_fast(np.zeros((2,1)), np.zeros((2,1))); print('dtaidistance C extension OK')" \
 && python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__, 'OK')"

# Project deps + editable install. Code changes only re-run from here; the C
# layer above stays cached, so iteration is fast.
COPY server/pyproject.toml /app/pyproject.toml
COPY server/app /app/app
COPY server/asl /app/asl
COPY server/train /app/train
COPY server/build /app/build
COPY server/tests /app/tests
RUN pip install -e ".[dev]"

# Defaults; secrets + volume paths are supplied by docker-compose.
ENV SIGNS_DATASET_PATH=/app/app/data/signs.json \
    ASL_TEMPLATES_DIR=/data/templates \
    DB_URL=sqlite:////data/arena.db \
    REPLAY_DIR=/data/replays

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "10"]
