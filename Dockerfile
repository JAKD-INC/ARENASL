# syntax=docker/dockerfile:1

# ---- Stage 1: build the frontend (Vite -> dist/) ----
FROM node:20-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN mkdir -p public && npm run build   # outputs /app/dist

# ---- Stage 2: python runtime (server + WLASL builder) ----
FROM python:3.11-slim-bookworm AS runtime

# MediaPipe's native runtime dlopens libGLESv2.so.2 / libEGL.so.1; OpenCV needs
# libGL + glib. Without these, create_from_options fails with a libGLESv2 error.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgles2 libegl1 libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements.txt backend/requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-build.txt

# Bake the MediaPipe models into the image (so nothing is downloaded at runtime).
RUN mkdir -p /app/public/models \
 && curl -fL -o /app/public/models/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
 && curl -fL -o /app/public/models/pose_landmarker_lite.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

COPY backend/ /app/backend/
COPY --from=frontend /app/dist /app/dist
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Templates + reference clips live on a mounted volume (written by the builder);
# models + built frontend are baked into the image.
ENV ASL_TEMPLATES_DIR=/data/templates \
    ASL_CLIPS_DIR=/data/clips \
    ASL_MODELS_DIR=/app/public/models \
    ASL_DIST_DIR=/app/dist \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
