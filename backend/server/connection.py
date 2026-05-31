from typing import Optional
from asl.features import normalize_frame
from asl.schema import assemble_frame, match_features
from asl.session import State


def state_to_dict(state: State) -> dict:
    return {
        "current": state.current,
        "queue": state.queue,
        "strength": state.strength,
        "score": state.score,
        "event": state.event,
        "confirmed": state.confirmed,
        "distance": state.distance,
    }


def handle_message(session, msg: dict) -> Optional[dict]:
    """Decode a landmark message, push it through the session, and return the
    encoded State dict. Returns None (send nothing) for frames with no pose or
    degenerate shoulders."""
    try:
        t = float(msg["t"])
        frame = assemble_frame(msg.get("pose"), msg.get("handLeft"), msg.get("handRight"))
        features = match_features(normalize_frame(frame).flatten())  # hands-only xy
    except (KeyError, TypeError, ValueError):
        return None  # malformed/incomplete frame -> send nothing
    state = session.push(features, t)
    return state_to_dict(state)
