"""The dashboard: one web page on the API server, for the people who watch a site, plus a login
in front of the whole API.

    server:
      token: ${ARGUS_TOKEN}     # in .env; empty = no login (only safe on 127.0.0.1)

    GET  /          the dashboard (the login page until signed in)
    POST /login     form field `token` -> a session cookie; POST /logout drops it
    GET  /status    per camera: last activity, online, objects tracked and alerts today, tamper

The page itself (dashboard.html) is plain HTML + JS over the API: live alert feed with
snapshots and check verdicts, camera cards, text search and plate lookup. No build step.

With a token, every request needs it: the session cookie (the page), or
`Authorization: Bearer <token>` (scripts). Make one with
`python -c "import secrets; print(secrets.token_urlsafe(24))"`. The cookie is HttpOnly and
SameSite=Strict; over a network, serve it through HTTPS (a reverse proxy) so it can't be read
on the way.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from argus.engine.cameras import expand_env

if TYPE_CHECKING:
    from argus.engine.store import EventStore

PAGE = Path(__file__).with_name("dashboard.html")
COOKIE = "argus_session"
PUBLIC = {"/login", "/logout", "/favicon.svg"}
ONLINE_SECONDS = 120     # a camera with nothing stored for this long shows as offline
TAMPER_SECONDS = 3600    # a tamper alert this recent shows on the camera's card
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" '
           'height="64" rx="14" fill="#0b1230"/><g transform="translate(2.46 2.46) scale(0.4923)">'
           '<path d="M8 60 Q60 8 112 60 Q60 112 8 60 Z" fill="none" stroke="#2864ff" '
           'stroke-width="12" stroke-linejoin="round"/><circle cx="60" cy="60" r="17" '
           'fill="#6fd8f7"/></g></svg>')

LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Argus</title>
<link rel="icon" href="/favicon.svg"><style>
:root{color-scheme:light dark;--bg:#f5f7fb;--card:#fff;--ink:#0b1230;--muted:#5b6478;--line:#dde3ee;--blue:#2864ff}
@media (prefers-color-scheme:dark){:root{--bg:#070c20;--card:#0f1733;--ink:#eef2ff;--muted:#9aa4c0;--line:#25304f}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
font:15px/1.4 "Segoe UI",system-ui,sans-serif}
form{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px;width:min(340px,90vw)}
h1{font-size:20px;margin:0 0 4px;display:flex;gap:10px;align-items:center}h1 img{width:30px}
p{color:var(--muted);margin:0 0 18px}input,button{width:100%;box-sizing:border-box;font:inherit;
padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
button{margin-top:12px;background:var(--blue);border-color:var(--blue);color:#fff;cursor:pointer}
.err{color:#d6453d;margin-top:10px}</style></head><body><form method="post" action="/login">
<h1><img src="/favicon.svg" alt="">Argus</h1><p>Enter the site's access token.</p>
<input type="password" name="token" autocomplete="current-password" autofocus required
aria-label="Access token">{error}<button type="submit">Sign in</button></form></body></html>"""


def session_value(token: str) -> str:
    """What the cookie holds: derived from the token, so the token itself is never stored."""
    return hmac.new(token.encode(), b"argus-dashboard-session", hashlib.sha256).hexdigest()


def server_token(config: dict[str, Any] | None) -> str | None:
    token = expand_env(str((config or {}).get("token") or "")).strip()
    return token or None


def check_host(host: str, token: str | None) -> None:
    """Refuse to serve beyond this machine without a login."""
    if host not in LOCAL_HOSTS and not token:
        raise SystemExit(f"--host {host} would expose the API without a login: set server.token "
                         "in the config (e.g. ${ARGUS_TOKEN} in .env), or keep 127.0.0.1")


def camera_status(store: EventStore, cameras: list[str], now: float | None = None
                  ) -> dict[str, Any]:
    """Per camera: last thing stored, online, objects tracked and alerts today, recent tamper."""
    now = time.time() if now is None else now
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0,
                                                   microsecond=0).timestamp()
    last: dict[str, float] = {}
    tracked: dict[str, dict[str, int]] = {}
    alerts: dict[str, dict[str, int]] = {}
    tamper: dict[str, dict[str, Any]] = {}
    with store._lock:  # same package: read with the store's own connection and lock
        db = store._db
        for table in ("sightings", "poses", "events", "plates"):
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
            if not exists:
                continue
            for camera, t in db.execute(f"SELECT camera, MAX(time) FROM {table} GROUP BY camera"):
                if t is not None:
                    last[camera] = max(last.get(camera, 0.0), t)
        for camera, label, n in db.execute(
                "SELECT camera, label, COUNT(DISTINCT track_id) FROM sightings WHERE kind = "
                "'object' AND time >= ? AND track_id IS NOT NULL GROUP BY camera, label",
                (midnight,)):
            tracked.setdefault(camera, {})[label] = n
        for camera, n, held in db.execute(
                "SELECT camera, COUNT(*), SUM(verified = 'no') FROM events WHERE time >= ? "
                "GROUP BY camera", (midnight,)):
            alerts[camera] = {"total": n, "held_back": int(held or 0)}
        for camera, rule, t in db.execute(
                "SELECT camera, rule, MAX(time) FROM events WHERE type = 'tamper' AND time >= ? "
                "GROUP BY camera", (now - TAMPER_SECONDS,)):
            tamper[camera] = {"rule": rule, "time": t,
                              "iso_time": datetime.fromtimestamp(t).isoformat(timespec="seconds")}
    names = sorted(set(cameras) | set(last))
    out = []
    for name in names:
        seen = last.get(name)
        counts = dict(sorted(tracked.get(name, {}).items(), key=lambda kv: -kv[1]))
        out.append({
            "camera": name,
            "last_seen": datetime.fromtimestamp(seen).isoformat(timespec="seconds") if seen else None,
            "seconds_ago": round(now - seen) if seen else None,
            "online": bool(seen and now - seen < ONLINE_SECONDS),
            "tracked_today": counts,
            "alerts_today": alerts.get(name, {"total": 0, "held_back": 0}),
            "tamper": tamper.get(name),
        })
    return {"time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "online_seconds": ONLINE_SECONDS, "cameras": out}


def add_dashboard(app: FastAPI, store: EventStore, cameras: list[str],
                  token: str | None) -> None:
    """The page, /status, and (with a token) a login in front of every route."""
    session = session_value(token) if token else None

    def signed_in(request: Request) -> bool:
        if session is None:
            return True
        cookie = request.cookies.get(COOKIE, "")
        if cookie and hmac.compare_digest(cookie, session):
            return True
        auth = request.headers.get("authorization", "")
        return auth.lower().startswith("bearer ") and token is not None and \
            hmac.compare_digest(auth[7:].strip(), token)

    @app.middleware("http")
    async def require_login(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if session is None or path in PUBLIC or signed_in(request):
            return await call_next(request)
        if path == "/" or path == "/dashboard":
            return HTMLResponse(LOGIN.replace("{error}", ""))
        return JSONResponse({"detail": "Sign in: session cookie or Authorization: Bearer <token>"},
                            status_code=401)

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def page() -> HTMLResponse:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-store"})

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon() -> Response:
        return Response(FAVICON, media_type="image/svg+xml")

    @app.post("/login", include_in_schema=False)
    async def login(token_field: str = Form("", alias="token")) -> Response:
        if session is None:
            return RedirectResponse("/", status_code=303)
        if not (token and secrets.compare_digest(token_field.strip(), token)):
            await asyncio.sleep(1.0)  # slows down guessing
            return HTMLResponse(LOGIN.replace("{error}", '<div class="err">Wrong token.</div>'),
                                status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, session, httponly=True, samesite="strict",
                            max_age=30 * 24 * 3600)
        return response

    @app.post("/logout", include_in_schema=False)
    def logout() -> Response:
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(COOKIE)
        return response

    @app.get("/status")
    def status() -> dict[str, Any]:
        """Per camera: last activity (online if something was stored in the last 2 minutes),
        objects tracked and alerts today (with how many the check held back), and a tamper
        alert in the last hour."""
        return {**camera_status(store, cameras), "login": session is not None}
