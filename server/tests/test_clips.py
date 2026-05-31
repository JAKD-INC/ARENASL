"""Reference example clips are served as static files at /clips/<gloss>.mp4.

The mount is added at app creation only when the configured asl_clips_dir
exists (so tests/CI without built clips still start). Here we point the setting
at a tmp dir holding a dummy clip and re-run the mount helper, then assert the
file is served. Routes are restored afterward so the shared app stays isolated."""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from app import main
from app.config import get_settings
from app.main import app


def test_clips_served_from_configured_dir(tmp_path, monkeypatch):
    clip = tmp_path / "book.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42dummy-mp4-bytes"
    clip.write_bytes(payload)

    monkeypatch.setenv("ASL_CLIPS_DIR", str(tmp_path))
    get_settings.cache_clear()

    routes_before = list(app.router.routes)
    try:
        assert main.mount_clips(app) is True
        with TestClient(app) as c:
            resp = c.get("/clips/book.mp4")
        assert resp.status_code == 200, resp.text
        assert resp.content == payload
    finally:
        app.router.routes[:] = routes_before
        get_settings.cache_clear()


def test_clips_not_mounted_when_dir_missing(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("ASL_CLIPS_DIR", str(missing))
    get_settings.cache_clear()
    try:
        assert main.mount_clips(app) is False
    finally:
        get_settings.cache_clear()


def test_default_clips_dir_setting():
    get_settings.cache_clear()
    try:
        assert get_settings().asl_clips_dir == "./data/clips"
    finally:
        get_settings.cache_clear()
