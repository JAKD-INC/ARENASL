import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient


def _write_templates(tmp_path):
    # Two trivial 1-D-ish templates so the app can load a Matcher.
    seq = np.zeros((3, 49 * 3))
    np.save(tmp_path / "book__0.npy", seq)
    np.save(tmp_path / "drink__0.npy", seq + 5.0)


def _reload_app(tmp_path, monkeypatch):
    _write_templates(tmp_path)
    monkeypatch.setenv("ASL_TEMPLATES_DIR", str(tmp_path))
    # Point the model artifacts at paths that don't exist so the matcher selection
    # is deterministic (independent of cwd / any real data/ dir).
    monkeypatch.setenv("ASL_ENCODER", str(tmp_path / "missing-encoder.onnx"))
    monkeypatch.setenv("ASL_PROTOTYPES", str(tmp_path / "missing-prototypes.npz"))
    import importlib, server.app
    importlib.reload(server.app)            # re-load templates from the env dir
    return server.app


def _client(tmp_path, monkeypatch):
    return TestClient(_reload_app(tmp_path, monkeypatch).app)


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


def test_falls_back_to_dtw_matcher_when_no_model_files(tmp_path, monkeypatch):
    # No encoder.onnx / prototypes.npz on disk -> keep the DTW Matcher (current
    # behavior), vocab taken from the reference templates.
    from asl.matcher import Matcher
    mod = _reload_app(tmp_path, monkeypatch)
    assert isinstance(mod._matcher, Matcher)
    assert mod._vocab == ["book", "drink"]
