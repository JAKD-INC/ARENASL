import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient


def _write_templates(tmp_path):
    # Two trivial 1-D-ish templates so the app can load a Matcher.
    seq = np.zeros((3, 49 * 3))
    np.save(tmp_path / "book__0.npy", seq)
    np.save(tmp_path / "drink__0.npy", seq + 5.0)


def _client(tmp_path, monkeypatch):
    _write_templates(tmp_path)
    monkeypatch.setenv("ASL_TEMPLATES_DIR", str(tmp_path))
    import importlib, server.app
    importlib.reload(server.app)            # re-load templates from the env dir
    return TestClient(server.app.app)


def _msg():
    pose = [[0.0, 0.0, 0.0] for _ in range(33)]
    pose[12] = [1.0, 0.0, 0.0]
    return {"t": 0.0, "pose": pose, "handLeft": None, "handRight": None}


def test_ws_returns_state_with_current_prompt(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(_msg())
        state = ws.receive_json()
        assert state["current"] in {"book", "drink"}
        assert "score" in state and "strength" in state
