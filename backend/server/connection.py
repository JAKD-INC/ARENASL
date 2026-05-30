from typing import Optional
from asl.features import normalize_frame
from asl.schema import assemble_frame
from asl.session import State


def state_to_dict(state: State) -> dict:
    return {
        "current": state.current,
        "queue": state.queue,
        "strength": state.strength,
        "score": state.score,
        "event": state.event,
        "confirmed": state.confirmed,
    }


def handle_message(session, msg: dict) -> Optional[dict]:
    """Decode a landmark message, push it through the session, and return the
    encoded State dict. Returns None (send nothing) for frames with no pose or
    degenerate shoulders."""
    try:
        t = float(msg["t"])
        frame = assemble_frame(msg.get("pose"), msg.get("handLeft"), msg.get("handRight"))
        normalized = normalize_frame(frame).flatten()
    except (KeyError, TypeError, ValueError):
        return None  # malformed/incomplete frame -> send nothing
    state = session.push(normalized, t)
    return state_to_dict(state)
