import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from asl.embedding_matcher import EmbeddingMatcher
from asl.library import load_templates
from asl.matcher import Matcher
from asl.schema import match_features
from asl.session import Session
from server.connection import handle_message
from server.prompts import prompt_stream

TEMPLATES_DIR = os.environ.get("ASL_TEMPLATES_DIR", "data/templates")
# Trained motion-embedding artifacts (Plan 3). When BOTH exist, the app swaps the
# DTW Matcher for the learned EmbeddingMatcher behind the same strength interface;
# otherwise it falls back to DTW so the server stays runnable pre-training.
ENCODER_PATH = os.environ.get("ASL_ENCODER", "data/encoder.onnx")
PROTOTYPES_PATH = os.environ.get("ASL_PROTOTYPES", "data/prototypes.npz")

# Calibration inherited from Plan 1's settled values (tune against real data).
# Calibration is env-tunable so you can dial it in live (restart, no rebuild):
#   ASL_SCALE=6 ASL_GET_THRESHOLD=0.5 docker compose up -d --force-recreate app
# Watch the HUD's `dist` (raw DTW distance) and set SCALE near a good match's
# distance so strength = exp(-dist/scale) spans 0..1 instead of pinning at 0.
SCALE = float(os.environ.get("ASL_SCALE", "10"))
GET_THRESHOLD = float(os.environ.get("ASL_GET_THRESHOLD", "0.5"))
CONFIRM_DROP = float(os.environ.get("ASL_CONFIRM_DROP", "0.8"))
# Frames the sign must stay above threshold to confirm by holding (no dip/move-on
# needed). ~10 frames ≈ 0.4s. Lower = quicker accept.
CONFIRM_HOLD = int(os.environ.get("ASL_CONFIRM_HOLD", "10"))
MISS_BUDGET = None  # no auto-miss/timer — advance only on a correct sign
# Small window so strength tracks RECENT motion and reacts fast. (Frame-based, so
# at low fps it spans more time — keep it short.)
WINDOW_SIZE = int(os.environ.get("ASL_WINDOW_SIZE", "16"))
# Cascade guard: frames the window must refill after a get before any confirm path
# fires (the embedding must reflect the NEW sign first). Defaults to one full window.
WARMUP_FRAMES = int(os.environ.get("ASL_WARMUP", str(WINDOW_SIZE)))


def _build_matcher():
    """Prefer the trained EmbeddingMatcher when its artifacts exist; otherwise fall
    back to the DTW Matcher over the reference templates. Returns (matcher, vocab)."""
    if os.path.isfile(ENCODER_PATH) and os.path.isfile(PROTOTYPES_PATH):
        matcher = EmbeddingMatcher.from_files(ENCODER_PATH, PROTOTYPES_PATH)
        return matcher, sorted(matcher._protos)
    # Reduce templates to the hands-only xy match features (same as live frames).
    templates = {g: [match_features(t) for t in seqs]
                 for g, seqs in load_templates(TEMPLATES_DIR).items()}
    return Matcher(templates, scale=SCALE), sorted(templates)


_matcher, _vocab = _build_matcher()

app = FastAPI()


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    session = Session(
        _matcher, prompt_stream(_vocab, seed=0),
        get_threshold=GET_THRESHOLD, confirm_drop=CONFIRM_DROP,
        miss_budget=MISS_BUDGET, window_size=WINDOW_SIZE, lookahead=3,
        confirm_hold=CONFIRM_HOLD, warmup_frames=WARMUP_FRAMES,
    )
    try:
        while True:
            msg = await websocket.receive_json()
            out = handle_message(session, msg)
            if out is not None:
                await websocket.send_json(out)
    except WebSocketDisconnect:
        pass


def _mount(env_var: str, route: str, *, html: bool = False) -> None:
    """Mount a static dir if its env var points at an existing directory.

    Skipped (so unit tests run without these) unless the dir exists. The "/"
    SPA mount is added last so /ws and /clips, /models take precedence."""
    directory = os.environ.get(env_var)
    if directory and os.path.isdir(directory):
        name = route.strip("/") or "root"
        app.mount(route, StaticFiles(directory=directory, html=html), name=name)


_mount("ASL_CLIPS_DIR", "/clips")          # reference clips (built into the volume)
_mount("ASL_MODELS_DIR", "/models")        # MediaPipe .task models
_mount("ASL_DIST_DIR", "/", html=True)     # built frontend (SPA) — must be last
