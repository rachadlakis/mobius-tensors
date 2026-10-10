"""The dashboard: login in front of the API, /status, and the page."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from argus.engine.dashboard import (COOKIE, camera_status, check_host, server_token,  # noqa: E402
                                    session_value)
from argus.engine.server import create_app  # noqa: E402
from argus.engine.store import EventStore  # noqa: E402

TOKEN = "s3cret-token-for-tests"


def app_with(tmp_path, token: str | None = None):
    config = {"store": {"path": str(tmp_path / "db.sqlite")}, "cameras": {"gate": {}, "yard": {}}}
    if token:
        config["server"] = {"token": token}
    return create_app(config)


def test_without_a_token_everything_is_open_and_the_page_is_served(tmp_path):
    client = TestClient(app_with(tmp_path))
    page = client.get("/")
    assert page.status_code == 200 and "<title>Argus</title>" in page.text and "/status" in page.text
    assert client.get("/events").status_code == 200
    status = client.get("/status").json()
    assert status["login"] is False and [c["camera"] for c in status["cameras"]] == ["gate", "yard"]


def test_with_a_token_every_route_needs_it(tmp_path):
    client = TestClient(app_with(tmp_path, TOKEN))
    for path in ("/events", "/status", "/health", "/media/x.jpg", "/docs", "/openapi.json"):
        assert client.get(path).status_code == 401, path
    login_page = client.get("/")
    assert login_page.status_code == 200 and 'action="/login"' in login_page.text
    assert client.get("/favicon.svg").status_code == 200  # public: the login page shows it

    auth = {"Authorization": f"Bearer {TOKEN}"}  # scripts
    assert client.get("/events", headers=auth).status_code == 200
    assert client.get("/events", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_login_sets_a_cookie_that_is_not_the_token(tmp_path):
    client = TestClient(app_with(tmp_path, TOKEN))
    bad = client.post("/login", data={"token": "nope"}, follow_redirects=False)
    assert bad.status_code == 401 and "Wrong token" in bad.text and COOKIE not in client.cookies

    ok = client.post("/login", data={"token": TOKEN}, follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    cookie = ok.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and TOKEN not in cookie
    assert client.cookies[COOKIE] == session_value(TOKEN)
    assert client.get("/status").json()["login"] is True
    assert "id=\"cameras\"" in client.get("/").text  # the dashboard, not the login page

    client.post("/logout", follow_redirects=False)
    client.cookies.clear()
    assert client.get("/status").status_code == 401


def test_camera_status(tmp_path):
    store = EventStore(tmp_path / "db.sqlite")
    now = datetime(2026, 10, 10, 15, 0).timestamp()
    emb = np.ones(4, np.float32) / 2
    store.add_sightings([
        {"camera": "gate", "time": now - 30, "kind": "object", "track_id": 1, "label": "person",
         "embedding": emb},
        {"camera": "gate", "time": now - 40, "kind": "object", "track_id": 1, "label": "person",
         "embedding": emb},
        {"camera": "gate", "time": now - 50, "kind": "object", "track_id": 2, "label": "car",
         "embedding": emb},
        {"camera": "yard", "time": now - 86400, "kind": "scene", "embedding": emb},  # yesterday
    ])
    store.add_event("gate", now - 20, {"type": "down", "rule": "person_down", "label": "person",
                                       "track_id": 1}, None)
    held = store.add_event("gate", now - 10, {"type": "enter", "rule": "yard", "label": "person",
                                              "track_id": 1}, None)
    store.set_verification(held, "no", "A shadow.")
    store.add_event("yard", now - 600, {"type": "tamper", "rule": "blocked", "label": "camera"},
                    None)
    s = camera_status(store, ["gate", "yard", "office"], now=now)
    by = {c["camera"]: c for c in s["cameras"]}
    assert by["gate"]["online"] and by["gate"]["seconds_ago"] == 10
    assert by["gate"]["tracked_today"] == {"person": 1, "car": 1}
    assert by["gate"]["alerts_today"] == {"total": 2, "held_back": 1}
    assert by["gate"]["tamper"] is None
    assert not by["yard"]["online"] and by["yard"]["tamper"]["rule"] == "blocked"
    assert by["office"]["last_seen"] is None and not by["office"]["online"]  # configured, silent
    store.close()


def test_host_check_and_token_from_env(monkeypatch):
    check_host("127.0.0.1", None)
    check_host("0.0.0.0", TOKEN)
    with pytest.raises(SystemExit, match="without a login"):
        check_host("0.0.0.0", None)
    assert server_token(None) is None and server_token({"token": "  "}) is None
    monkeypatch.setenv("ARGUS_TEST_TOKEN", TOKEN)
    assert server_token({"token": "${ARGUS_TEST_TOKEN}"}) == TOKEN
    monkeypatch.delenv("ARGUS_TEST_TOKEN")
    with pytest.raises(ValueError, match="not set"):  # a missing secret is never "no login"
        server_token({"token": "${ARGUS_TEST_TOKEN}"})


def test_wrong_logins_are_slowed_down(tmp_path):
    client = TestClient(app_with(tmp_path, TOKEN))
    start = time.monotonic()
    client.post("/login", data={"token": "guess"}, follow_redirects=False)
    assert time.monotonic() - start >= 0.9
