from __future__ import annotations

import pytest

from app.auth.security import hash_password, verify_password
from app.elo import STARTING_ELO, Experience
from tests.conftest import register


def test_password_hash_roundtrip():
    h = hash_password("hunter2hunter2")
    assert h != "hunter2hunter2"
    assert verify_password("hunter2hunter2", h)
    assert not verify_password("wrong-password", h)


def test_register_returns_token_and_me_works(client):
    token = register(client, email="a@example.com", display_name="Ada")
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["display_name"] == "Ada"
    assert body["elo"] == STARTING_ELO[Experience.intermediate]
    assert isinstance(body["player_id"], int)


@pytest.mark.parametrize("experience", list(Experience))
def test_experience_seeds_starting_elo(client, experience):
    token = register(client, email=f"{experience.value}@example.com", experience=experience.value)
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.json()["elo"] == STARTING_ELO[experience]


def test_duplicate_email_is_rejected(client):
    register(client, email="dup@example.com")
    resp = client.post(
        "/auth/register",
        json={
            "email": "dup@example.com",
            "password": "supersecret123",
            "display_name": "Other",
            "experience": "beginner",
        },
    )
    assert resp.status_code == 409


def test_email_is_normalized_case_insensitively(client):
    register(client, email="Mixed@Example.com")
    resp = client.post(
        "/auth/register",
        json={
            "email": "mixed@example.com",
            "password": "supersecret123",
            "display_name": "Other",
            "experience": "beginner",
        },
    )
    assert resp.status_code == 409


def test_login_success_and_failure(client):
    register(client, email="login@example.com", password="rightpassword1")
    ok = client.post("/auth/token", data={"username": "login@example.com", "password": "rightpassword1"})
    assert ok.status_code == 200
    assert ok.json()["token_type"] == "bearer"

    bad = client.post("/auth/token", data={"username": "login@example.com", "password": "wrongpassword"})
    assert bad.status_code == 401

    missing = client.post("/auth/token", data={"username": "nobody@example.com", "password": "whatever12"})
    assert missing.status_code == 401


def test_me_requires_a_valid_token(client):
    assert client.get("/auth/me").status_code == 401
    bad = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert bad.status_code == 401


def test_signs_endpoint_serves_dataset(client):
    resp = client.get("/signs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert len(body["entries"]) > 0
    assert {"sign_id", "word", "difficulty"} <= body["entries"][0].keys()
