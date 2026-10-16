"""The Today page: the console's home ("Histor UI Design/handoff"). The day so far in one
place: a written summary, the alerts that need a look, each camera's activity, the journal of
events, and the chat beside it.

    GET  /                         the page (today.html: plain HTML + JS, system fonts, offline)
    GET  /api/today                the day so far: figures, per-camera activity, events
    GET  /api/summary              one written paragraph of the day so far (chat model, cached)
    GET  /api/events/{id}          more frames of one event, and a face to look for
    POST /api/events/{id}/seen     an alert someone looked at: it leaves "Needs a look"
    POST /api/ask                  {question, event?} -> lookups made, answer, evidence
    POST /api/ask/reset            a new conversation
    POST /api/photo                a photo (body: the image file) for the face tool
    GET  /api/status               recording or not: the top bar of every page polls it
    GET  /media/{path}             a stored snapshot or thumbnail

The top bar (TOPBAR_*) is shared with the Gradio pages (Cameras, Find a person, Ask, Reports,
Test bench), mounted under /app by ui.py.
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from histor.chat import answer, model_config, system_prompt
from histor.engine.search import SearchEngine
from histor.engine.status import camera_status
from histor.llm import Backend, create_backend
from histor.report import Reporter, ReportSpec
from histor.tools import Toolbox

log = logging.getLogger("histor.today")
PAGE = Path(__file__).with_name("today.html")

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "van"}
KIND_ORDER = {"alert": 0, "warn": 1, "plate": 2, "info": 3}  # a grid cell shows the first
TAMPER = {"blocked": ("Camera covered", "The {cam} camera was covered"),
          "dark": ("Picture went dark", "The {cam} picture went dark"),
          "blurred": ("Picture blurred", "The {cam} picture went blurry"),
          "moved": ("Camera moved", "The {cam} camera was moved"),
          "frozen": ("Picture frozen", "The {cam} picture froze")}
SUMMARY_MAX_AGE = 15 * 60     # the written summary is made again after this long...
SUMMARY_MIN_AGE = 2 * 60      # ...or after a new event, but not more often than this
STATUS_MAX_AGE = 30           # model server check

TODAY_PROMPT = """Write a short summary of {period} so far, for the site's owner.
Cameras: {cameras}.
{interests}
Exact counts so far (from the store):
{facts}

Check details with your tools if needed, always with since="{since}" and until="{until}".
One paragraph of plain sentences, under {words} words: first the alerts that matter, with
camera and time, then how busy it was. No title, no list, no greeting.
Only what the tools returned: no guesses presented as facts."""

# ---------------------------------------------------------------------------- top bar

APP = "/app"  # the Gradio pages (dark, like this one)
NAV = [("Today", "/"), ("Cameras", f"{APP}/?__theme=dark&tab=cameras"),
       ("Find a person", f"{APP}/?__theme=dark&tab=find"),
       ("Ask", f"{APP}/?__theme=dark&tab=ask"), ("Reports", f"{APP}/?__theme=dark&tab=reports"),
       ("Test bench", f"{APP}/?__theme=dark&tab=bench")]
DEV_ONLY = {"Test bench"}

TOPBAR_CSS = """
.hx-top{--hx-text:#eef1f7;--hx-text-2:#c9d0df;--hx-muted:#97a1ba;--hx-dim:#69738f;
  --hx-line:rgba(151,161,186,.14);--hx-blue:#2864ff;--hx-cyan:#6fd8f7;--hx-alert:#ef4444;
  height:60px;flex:none;display:flex;align-items:center;gap:20px;padding:0 20px;
  background:rgba(8,13,31,.86);-webkit-backdrop-filter:saturate(1.4) blur(14px);
  backdrop-filter:saturate(1.4) blur(14px);border-bottom:1px solid var(--hx-line);
  box-shadow:0 8px 24px -16px rgba(0,0,0,.6);color:var(--hx-text);
  font:14px/1.4 ui-sans-serif,system-ui,-apple-system,'Segoe UI Variable Text','Segoe UI',sans-serif;
  box-sizing:border-box;position:relative;z-index:20}
.hx-top *{box-sizing:border-box}
.hx-top a,.hx-top a:hover{text-decoration:none!important}
.hx-top :focus-visible{outline:2px solid var(--hx-blue)!important;outline-offset:2px}
.hx-brand{display:flex;align-items:center;gap:10px;color:var(--hx-text)!important;flex:none;
  padding-right:20px;border-right:1px solid var(--hx-line)!important;height:28px}
.hx-brand svg,.hx-brand img{width:28px;height:28px;border-radius:8px;display:block;
  box-shadow:0 0 0 1px rgba(111,216,247,.18),0 4px 14px -4px rgba(40,100,255,.55)}
.hx-brand span{font-weight:650;font-size:16px;letter-spacing:-.01em}
.hx-nav{display:flex;align-items:center;gap:2px;padding:4px;border-radius:12px;
  background:rgba(151,161,186,.05);border:1px solid var(--hx-line);min-width:0;
  overflow-x:auto;scrollbar-width:none}
.hx-nav::-webkit-scrollbar{display:none}
.hx-nav a{position:relative;display:flex;align-items:center;gap:8px;height:34px;padding:0 12px;
  border-radius:8px;color:var(--hx-muted)!important;font-weight:500;font-size:14px;
  white-space:nowrap;border:1px solid transparent!important;
  transition:color .15s,background-color .15s,border-color .15s}
.hx-nav a svg{width:16px;height:16px;flex:none;fill:none;stroke:currentColor;stroke-width:1.8;
  stroke-linecap:round;stroke-linejoin:round;opacity:.85}
.hx-nav a:hover{color:var(--hx-text)!important;background:rgba(151,161,186,.08)}
.hx-nav a[aria-current="page"]{color:var(--hx-text)!important;font-weight:600;
  background:linear-gradient(180deg,#1d2850,#161e3d);border-color:rgba(151,161,186,.2)!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 2px 8px -2px rgba(0,0,0,.5)}
.hx-nav a[aria-current="page"] svg{color:var(--hx-cyan);opacity:1}
.hx-nav a small{font-size:9.5px;font-weight:700;letter-spacing:.06em;color:var(--hx-cyan);
  padding:1px 5px;border-radius:5px;background:rgba(111,216,247,.12)}
.hx-badge{min-width:18px;height:18px;padding:0 5px;border-radius:999px;background:var(--hx-alert);
  color:#fff;font-size:11px;font-weight:700;line-height:18px;text-align:center;
  font-variant-numeric:tabular-nums}
.hx-badge[hidden]{display:none}
.hx-nav a.hx-dev-only{display:none}
body.hx-dev .hx-nav a.hx-dev-only{display:flex}
.hx-spacer{flex:1}
.hx-rec{display:flex;align-items:center;gap:8px;height:30px;padding:0 12px;border-radius:999px;
  font-size:12.5px;font-weight:500;color:var(--hx-muted);white-space:nowrap;flex:none;
  border:1px solid var(--hx-line);background:rgba(151,161,186,.05)}
.hx-rec i{position:relative;width:8px;height:8px;border-radius:50%;background:var(--hx-dim)}
.hx-rec.on{color:#fecaca;border-color:rgba(239,68,68,.35);background:rgba(239,68,68,.08)}
.hx-rec.on i{background:var(--hx-alert)}
.hx-rec.on i::after{content:"";position:absolute;inset:-4px;border-radius:50%;
  border:2px solid var(--hx-alert);opacity:0;animation:hx-pulse 1.8s ease-out infinite}
@keyframes hx-pulse{0%{transform:scale(.5);opacity:.7}100%{transform:scale(1.4);opacity:0}}
@media (prefers-reduced-motion:reduce){.hx-rec.on i::after{animation:none}}
.hx-devswitch{display:flex;align-items:center;gap:9px;cursor:pointer;font-size:12.5px;
  font-weight:500;color:var(--hx-muted)!important;background:none!important;border:0!important;
  box-shadow:none!important;padding:4px 2px!important;min-width:0!important;
  font-family:inherit;flex:none;border-radius:8px}
.hx-devswitch:hover{color:var(--hx-text)!important}
.hx-devswitch b{width:30px;height:18px;border-radius:999px;position:relative;background:#2a3253;
  box-shadow:inset 0 0 0 1px rgba(151,161,186,.2);transition:background .2s}
.hx-devswitch b::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;
  border-radius:50%;background:#eef1f7;box-shadow:0 1px 3px rgba(0,0,0,.4);transition:left .2s}
body.hx-dev .hx-devswitch{color:var(--hx-text-2)!important}
body.hx-dev .hx-devswitch b{background:var(--hx-blue);box-shadow:none}
body.hx-dev .hx-devswitch b::after{left:14px}
.hx-theme{width:32px;height:32px;display:grid;place-items:center;border-radius:999px;
  cursor:pointer;color:var(--hx-muted)!important;background:rgba(151,161,186,.05)!important;
  border:1px solid var(--hx-line)!important;box-shadow:none!important;padding:0!important;
  min-width:0!important;flex:none;transition:color .15s,background-color .15s}
.hx-theme:hover{color:var(--hx-text)!important;background:rgba(151,161,186,.12)!important}
.hx-theme svg{width:16px;height:16px;fill:none;stroke:currentColor;stroke-width:1.8;
  stroke-linecap:round;stroke-linejoin:round}
.hx-theme .moon,html[data-hx-theme="light"] .hx-theme .sun{display:none}
html[data-hx-theme="light"] .hx-theme .moon{display:block}
/* light: the same bar on white */
html[data-hx-theme="light"] .hx-top{--hx-text:#0b1230;--hx-text-2:#2a3253;--hx-muted:#4a5370;
  --hx-dim:#97a1ba;--hx-line:rgba(11,18,48,.10);--hx-cyan:#1a4fe6;
  background:rgba(255,255,255,.86);box-shadow:0 8px 24px -18px rgba(11,18,48,.35)}
html[data-hx-theme="light"] .hx-nav{background:rgba(11,18,48,.03)}
html[data-hx-theme="light"] .hx-nav a:hover{background:rgba(11,18,48,.05)}
html[data-hx-theme="light"] .hx-nav a[aria-current="page"]{background:#fff;
  border-color:rgba(11,18,48,.12)!important;box-shadow:0 1px 3px rgba(11,18,48,.12)}
html[data-hx-theme="light"] .hx-nav a small{background:rgba(40,100,255,.10)}
html[data-hx-theme="light"] .hx-rec.on{color:#b91c1c;background:rgba(239,68,68,.06)}
html[data-hx-theme="light"] .hx-devswitch b{background:#dde3ee}
html[data-hx-theme="light"] .hx-theme{background:rgba(11,18,48,.03)!important}
@media (max-width:1180px){.hx-rec span{display:none}.hx-rec{padding:0 10px}}
@media (max-width:1040px){.hx-nav a span{display:none}.hx-nav a{padding:0 10px}
  .hx-brand span{display:none}.hx-brand{padding-right:14px}.hx-top{gap:14px;padding:0 14px}}
@media (max-width:560px){.hx-devswitch span,.hx-rec{display:none}}
"""

# Shared by the Today page and the Gradio pages: developer switch (remembered in this
# browser), recording status, and on the Gradio pages the nav switches tabs in place.
TOPBAR_JS = """
(() => {
  const KEY = 'histor.dev';
  const get = () => { try { return localStorage.getItem(KEY) === '1'; } catch (e) { return false; } };
  const apply = on => { document.body.classList.toggle('hx-dev', on);
    document.querySelectorAll('.hx-devswitch').forEach(b => b.setAttribute('aria-checked', on));
    document.dispatchEvent(new CustomEvent('histor-dev', {detail: on})); };
  const ready = fn => document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', fn) : fn();
  ready(() => apply(get()));
  // light / dark, remembered in this browser (dark by default); Gradio's own: body.dark
  const TKEY = 'histor.theme';
  const app = location.pathname.startsWith('/app');
  const theme = () => { try { return localStorage.getItem(TKEY) || 'dark'; } catch (e) { return 'dark'; } };
  function applyTheme(t) {
    document.documentElement.dataset.hxTheme = t;
    if (app && document.body) document.body.classList.toggle('dark', t === 'dark');
    document.querySelectorAll('a[href*="__theme="]').forEach(a =>
      a.setAttribute('href', a.getAttribute('href').replace(/__theme=(dark|light)/, '__theme=' + t)));
    document.querySelectorAll('.hx-theme').forEach(b => {
      const next = t === 'dark' ? 'light' : 'dark';
      b.title = 'Switch to the ' + next + ' theme'; b.setAttribute('aria-label', b.title); });
    if (app) { const u = new URL(location.href);
      if (u.searchParams.get('__theme') !== t) { u.searchParams.set('__theme', t);
        history.replaceState(null, '', u); } }
    document.dispatchEvent(new CustomEvent('histor-theme', {detail: t}));
  }
  window.histor_theme = theme;
  applyTheme(theme());  // before the page paints: no flash of the other theme
  ready(() => {
    applyTheme(theme());
    // Gradio sets its own theme once it has loaded: keep ours for the first seconds
    if (app) { let n = 0; const t = setInterval(() => { applyTheme(theme()); if (++n > 25) clearInterval(t); }, 200); }
  });
  document.addEventListener('click', e => {
    const tb = e.target.closest('.hx-theme');
    if (tb) { const t = theme() === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(TKEY, t); } catch (err) {}
      applyTheme(t); return; }
    const sw = e.target.closest('.hx-devswitch');
    if (sw) { const on = !document.body.classList.contains('hx-dev');
      try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (err) {}
      apply(on); return; }
    const a = e.target.closest('.hx-nav a[data-tab]');
    if (a && location.pathname.startsWith('/app') && openTab(a.dataset.tab)) {
      e.preventDefault(); history.replaceState(null, '', a.getAttribute('href'));
    }
  });
  // Gradio pages: the nav drives the (hidden) Gradio tabs
  function tabButton(label) {
    return [...document.querySelectorAll('.ag-nav [role="tab"]')]
      .find(b => b.textContent.trim() === label);
  }
  function mark(label) {
    document.querySelectorAll('.hx-nav a').forEach(x => x.dataset.tab === label
      ? x.setAttribute('aria-current', 'page') : x.removeAttribute('aria-current'));
  }
  function openTab(label) {
    const b = tabButton(label);
    if (!b) return false;
    b.click(); mark(label); return true;
  }
  window.histor_openTab = openTab;
  if (location.pathname.startsWith('/app')) {
    const want = new URLSearchParams(location.search).get('tab') || 'cameras';
    // the server selects the tab (ui.tab_from_link); the bar shows which, once Gradio drew it
    let tries = 0;
    const t = setInterval(() => {
      const a = document.querySelector('.hx-nav a[href$="tab=' + want + '"]');
      if (a || ++tries > 100) { clearInterval(t); if (a) mark(a.dataset.tab); }
    }, 100);
  }
  async function status() {
    try {
      const r = await fetch('/api/status'); const s = await r.json();
      document.querySelectorAll('.hx-rec').forEach(el => {
        el.classList.toggle('on', s.on); el.querySelector('span').textContent = s.text;
        el.title = s.text; });
      document.querySelectorAll('.hx-badge').forEach(b => {
        b.hidden = !s.open; b.textContent = s.open > 99 ? '99+' : s.open;
        b.title = s.open + (s.open === 1 ? ' alert' : ' alerts') + ' to look at'; });
    } catch (e) {}
  }
  ready(() => { status(); setInterval(status, 5000); });
})();
"""


ICONS = {  # 24 x 24 line icons, drawn here (no icon font: the page loads nothing)
    "Today": '<rect x="3" y="4.5" width="18" height="16" rx="3"/><path d="M3 9.5h18M8 2.5v4M16 '
             '2.5v4"/>',
    "Cameras": '<rect x="2.5" y="6" width="13" height="12" rx="3"/><path d="m15.5 10.5 6-3.5v10'
               'l-6-3.5"/>',
    "Find a person": '<circle cx="10" cy="8" r="4"/><path d="M3 20c.8-3.6 3.6-6 7-6 1.4 0 2.7.4 '
                     '3.8 1.1"/><circle cx="17.5" cy="17.5" r="3"/><path d="m21.5 21.5-1.9-1.9"/>',
    "Ask": '<path d="M20.5 12a8.5 8.5 0 0 1-12.4 7.6L3.5 21l1.4-4.4A8.5 8.5 0 1 1 20.5 12Z"/>',
    "Reports": '<path d="M14 2.5H7A2.5 2.5 0 0 0 4.5 5v14A2.5 2.5 0 0 0 7 21.5h10a2.5 2.5 0 0 0 '
               '2.5-2.5V8Z"/><path d="M14 2.5V8h5.5M8.5 13h7M8.5 17h5"/>',
    "Test bench": '<path d="M9 2.5h6M10 2.5v6.2L4.6 18.3A2.1 2.1 0 0 0 6.4 21.5h11.2a2.1 2.1 0 0 '
                  '0 1.8-3.2L14 8.7V2.5"/><path d="M7 15h10"/>',
}


def topbar_html(active: str, logo: str) -> str:
    """The top bar; `active`: the NAV label of this page. logo: inline SVG. Today carries the
    count of alerts that need a look (TOPBAR_JS fills it)."""
    links = "".join(
        f'<a href="{href}" data-tab="{html.escape(label)}" title="{html.escape(label)}"'
        f'{" class=hx-dev-only" if label in DEV_ONLY else ""}'
        f'{" aria-current=page" if label == active else ""}>'
        f'<svg viewBox="0 0 24 24" aria-hidden="true">{ICONS.get(label, "")}</svg>'
        f'<span>{html.escape(label)}</span>'
        f'{"<small>DEV</small>" if label in DEV_ONLY else ""}'
        f'{"<b class=hx-badge hidden></b>" if label == "Today" else ""}</a>'
        for label, href in NAV)
    return (f'<header class="hx-top"><a class="hx-brand" href="/" aria-label="Histor: Today">'
            f'{logo}<span>Histor</span></a>'
            f'<nav class="hx-nav" aria-label="Main">{links}</nav><div class="hx-spacer"></div>'
            '<div class="hx-rec" role="status"><i></i><span>…</span></div>'
            '<button type="button" class="hx-theme" title="Switch theme" aria-label="Switch '
            'theme"><svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" '
            'cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 '
            '12h2M4.6 19.4 6 18M18 6l1.4-1.4"/></svg><svg class="moon" viewBox="0 0 24 24" '
            'aria-hidden="true"><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 '
            '11Z"/></svg></button>'
            '<button type="button" class="hx-devswitch" role="switch" aria-checked="false" '
            'title="Track ids, scores, models, and the test bench"><span>Developer</span><b>'
            '</b></button></header>')


# ---------------------------------------------------------------------------- events


def _plural(n: int, word: str) -> str:
    if word == "person":
        return f"{n} {'person' if n == 1 else 'people'}"
    return f"{n} {word}{'' if n == 1 else 's'}"


def _secs(value: Any) -> str:
    if value is None:
        return ""
    v = float(value)
    return f"{v:.0f} s" if v < 120 else f"{v / 60:.0f} min"


def _nice(rule: str) -> str:
    return (rule or "").replace("_", " ").strip()


def describe(e: dict[str, Any]) -> dict[str, Any]:
    """One stored event in plain words for the page: kind (alert | warn | plate | info: the
    colour), group (the filters), type, short / title / text, state."""
    t, rule, cam = e["type"], e.get("rule") or "", e["camera"]
    label = e.get("label") or "object"
    held = _secs(e.get("dwell_seconds"))
    group = "Vehicles" if label in VEHICLES else "People" if label == "person" else "Other"
    text = ""
    if t == "down":
        kind, group, typ = "alert", "People", "Person down"
        short = f"Person down for {held}" if held else "Person down"
        title = f"A person was down on the ground for {held}" if held else "A person was down"
    elif t == "hands_up":
        kind, group, typ = "alert", "People", "Hands up"
        short = f"Hands up for {held}" if held else "Hands up"
        title = f"Someone held their hands up for {held}" if held else "Someone held their hands up"
    elif t == "fire":
        kind, group, typ = "alert", "Other", "Fire"
        short = "Smoke seen" if rule == "smoke" else "Fire seen"
        title = f"{short} on the {cam} camera"
        text = "Check the camera now; the fire/smoke model raised it."
    elif t == "tamper":
        kind, group, typ = "alert", "Other", "Tamper"
        short, title = TAMPER.get(rule, (f"Tamper: {_nice(rule)}", "The {cam} camera: " +
                                         _nice(rule)))
        title = title.format(cam=cam)
    elif t == "sound":
        kind, group, typ = "alert", "Sounds", "Sound"
        short = _nice(rule).capitalize() or "Sound"
        title = f"{short} heard on the {cam} camera"
        text = f"Heard: {label}." if label != "object" else ""
    elif t == "plate":
        kind, group, typ = "plate", "Vehicles", "Watchlist plate"
        plate = e.get("plate") or "?"
        short = plate
        title = f"A {label}, {plate}" if label != "object" else f"Plate {plate}"
        text = f"The plate is on the “{_nice(rule)}” watchlist."
    elif t == "abandoned":
        kind, group, typ = "warn", "People", "Bag left"
        short = f"{label.capitalize()} left behind"
        title = f"A {label} left with nobody by it"
        text = f"Still for {held}." if held else ""
    elif t == "crowd":
        kind, typ = "warn", "Crowding"
        n = int(e.get("count") or 0)
        short = f"{_plural(n, label)} in view"
        title = f"{_plural(n, label)} in “{_nice(rule)}”"
    elif t in ("enter", "dwell"):
        kind, typ = "warn", "Area"
        short = f"Stayed in {_nice(rule)}" if t == "dwell" else f"Entered {_nice(rule)}"
        title = (f"A {label} stayed {held} in “{_nice(rule)}”" if t == "dwell" and held
                 else f"A {label} {'stayed in' if t == 'dwell' else 'entered'} “{_nice(rule)}”")
    elif t in ("exit", "cross"):
        kind, typ = "info", "Line" if t == "cross" else "Area"
        way = f" ({e['direction']})" if e.get("direction") else ""
        short = f"{'Crossed' if t == 'cross' else 'Left'} {_nice(rule)}{way}"
        title = f"A {label} {'crossed' if t == 'cross' else 'left'} “{_nice(rule)}”{way}"
    else:
        kind, typ = "info", t.capitalize()
        short, title = _nice(rule).capitalize() or typ, f"{typ}: {_nice(rule)}"
    verified = e.get("verified")
    if verified == "no":
        text = (text + " " if text else "") + "The check says it was not real" + (
            f": {e['verification']}" if e.get("verification") else ".")
    elif verified == "yes":
        text = (text + " " if text else "") + "Confirmed by the check."
    state = ""
    if kind == "alert":
        state = "held" if verified == "no" else "seen" if e.get("seen") else "open"
    when = datetime.fromtimestamp(e["time"])
    dev = [f"{t} · rule {rule or '-'} · {label}" + (
        f" · track {e['track_id']}" if e.get("track_id") is not None else ""),
        f"event {e['id']} · {e['iso_time']}"]
    if e.get("dwell_seconds") is not None:
        dev.append(f"held {float(e['dwell_seconds']):.1f} s")
    if verified:
        dev.append(f"check: {verified}" + (f" · {e['verification']}" if e.get("verification")
                                           else ""))
    return {"id": e["id"], "time": f"{when:%H:%M:%S}",
            "hour": when.hour + when.minute / 60 + when.second / 3600, "camera": cam,
            "kind": kind, "group": group, "type": typ, "short": short, "title": title,
            "text": text.strip(), "state": state, "label": label,
            "img": f"/media/{e['snapshot']}" if e.get("snapshot") else None,
            "dev": "\n".join(dev)}


def plain_summary(events: list[dict[str, Any]], visits: dict[str, int]) -> str:
    """The day in a few sentences from the counts alone (no model): shown at once, and when
    the model can't write one."""
    total = sum(visits.values())
    if not total and not events:
        return "Nothing recorded yet today."
    parts = []
    if total:
        busiest = max(visits, key=visits.get)
        on = f" on {_plural(len(visits), 'camera')}" if len(visits) > 1 else f" on {busiest}"
        parts.append(f"{total:,} visits so far{on}" + (
            f", the most on {busiest}." if len(visits) > 1 else "."))
    alerts = [e for e in events if e["kind"] == "alert"]
    if alerts:
        listed = "; ".join(f"{e['short'].lower()} at {e['time'][:5]} on {e['camera']}"
                           for e in alerts[-3:])
        more = f" and {len(alerts) - 3} more" if len(alerts) > 3 else ""
        parts.append(f"{_plural(len(alerts), 'alert')}: {listed}{more}.")
    else:
        parts.append("No alerts.")
    others = len(events) - len(alerts)
    if others:
        parts.append(f"{_plural(others, 'other event')} logged.")
    return " ".join(parts)


def call_text(record: logging.LogRecord) -> str:
    """A logged tool call (chat.answer: "-> name args") as `name(key="value", ...)`."""
    args = record.args if isinstance(record.args, tuple) else ()
    if len(args) == 2 and isinstance(args[1], dict):
        return f"{args[0]}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}"
                                         for k, v in args[1].items()) + ")"
    return record.getMessage().strip().removeprefix("-> ")


def _midnight(now: float) -> datetime:
    return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)


class Today:
    """The page's data and its chat (one conversation for the page, like the console)."""

    def __init__(self, config: dict[str, Any], engine: SearchEngine,
                 recording: Callable[[], set[str]],
                 model_state: Callable[[dict[str, Any], str | None], tuple[str, str, str]]
                 ) -> None:
        """recording: the cameras this process records right now; model_state: ui.model_state."""
        self.config = config
        self.engine = engine
        self.store = engine.store
        self.recording = recording
        self.model_state = model_state
        self.toolbox = Toolbox.from_config(engine, config)
        models = list((config.get("chat") or {}).get("models") or {})
        self.model = (config.get("chat") or {}).get("model", models[0] if models else None)
        try:
            self.reporter: Reporter | None = Reporter(config, engine)
        except ValueError as e:  # a bad `reports:` entry: counts-only summary
            log.warning("Today: no written summary (%s)", e)
            self.reporter = None
        self._chat_lock = threading.Lock()
        self._backend: Backend | None = None
        self._untold: list[str] = []
        self._summary_lock = threading.Lock()
        self._summary: dict[str, Any] = {"day": None, "events": -1, "time": 0.0, "text": None}
        self._model_line: tuple[float, dict[str, str]] = (0.0, {})

    def close(self) -> None:
        self.toolbox.close()
        if self.reporter:
            self.reporter.close()

    # ------------------------------------------------------------------ the day

    def cameras(self, *have: Any) -> list[str]:
        """The configured cameras first, then any other that recorded (a simulation)."""
        names = list(self.engine.camera_names)
        extra = set().union(*(set(h) for h in have)) - set(names)
        return names + sorted(extra)

    def day(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        start = _midnight(now)
        since, until = start.timestamp(), (start + timedelta(days=1)).timestamp()
        events = [describe(e) for e in
                  reversed(self.store.list_events(since=since, until=until, limit=2000))]
        bins = self.store.activity(since, until, 1800)
        visits = self.store.visits(since, until)
        names = self.cameras(bins, visits, {e["camera"] for e in events})
        faces = sum(self.store.face_counts(since, until).values())
        plates = len({(p["camera"], p["track_id"] if p["track_id"] is not None else p["id"])
                      for p in self.store.plate_rows(since, until)})
        lanes = []
        for name in names:
            half = bins.get(name, [0] * 48)
            lanes.append({"name": name, "visits": visits.get(name, 0), "bins": half,
                          "hours": [half[2 * h] + half[2 * h + 1] for h in range(24)]})
        status = self.status()
        return {"date": f"{start:%Y-%m-%d}", "day": f"{start.day} {start:%B}",
                "weekday": f"{start:%A}", "now": (now - since) / 3600,
                "figures": {"visits": sum(visits.values()), "faces": faces, "plates": plates,
                            "events": len(events)},
                "summary": plain_summary(events, visits), "cameras": lanes, "events": events,
                "recording": status, "model": self.model_line(),
                "faces_on": self.engine.faces is not None}

    def status(self) -> dict[str, Any]:
        """Recording: this process's recorder, and any other recorder storing right now."""
        names = list(self.engine.camera_names)
        online = {c["camera"] for c in camera_status(self.store, names)["cameras"]
                  if c["online"]}
        on = sorted(online | self.recording())
        text = "Not recording" if not on else \
            f"Recording {on[0]}" if len(on) == 1 else f"Recording {_plural(len(on), 'camera')}"
        since = _midnight(time.time()).timestamp()
        open_alerts = sum(describe(e)["state"] == "open"
                          for e in self.store.list_events(since=since, limit=2000))
        return {"on": bool(on), "text": text, "cameras": on, "open": open_alerts}

    def model_line(self) -> dict[str, str]:
        """The chat's model in a few words, for the Ask panel's header (checked every 30 s)."""
        checked, line = self._model_line
        if line and time.monotonic() - checked < STATUS_MAX_AGE:
            return line
        kind, title, _ = self.model_state(self.config, self.model)
        try:
            cfg = model_config(self.config, self.model) if self.model else {}
        except SystemExit:
            cfg = {}
        name = cfg.get("model") or self.model or "no model"
        text = (f"{name} · footage stays on site" if kind == "ok" else
                f"{name} · cloud, text only" if title.endswith("cloud model") else title)
        line = {"state": kind, "text": text}
        self._model_line = (time.monotonic(), line)
        return line

    def summary(self) -> dict[str, Any]:
        """The written paragraph of the day so far, made again after a new event or 15 min."""
        with self._summary_lock:
            now = time.time()
            start = _midnight(now)
            events = self.store.list_events(since=start.timestamp(), limit=2000)
            s = self._summary
            age = now - s["time"]
            fresh = s["day"] == f"{start:%Y-%m-%d}" and s["text"] and (
                age < SUMMARY_MIN_AGE or (age < SUMMARY_MAX_AGE and s["events"] == len(events)))
            if fresh:
                return {"text": s["text"], "source": s["source"]}
            text, source = None, "counts"
            if self.reporter is not None:
                spec = ReportSpec("today", interests=self.reporter.custom_spec().interests,
                                  words=70)
                try:
                    made = self.reporter.compose(spec, start, datetime.fromtimestamp(now),
                                                 custom=True, prompt=TODAY_PROMPT)
                    text, source = made["summary"], "model"
                except Exception:  # never break the page for its summary
                    log.exception("Today: summary failed")
            if not text:
                day = self.day(now)
                text, source = day["summary"], "counts"
            self._summary = {"day": f"{start:%Y-%m-%d}", "events": len(events), "time": now,
                             "text": " ".join(text.split()), "source": source}
            return {"text": self._summary["text"], "source": source}

    def event(self, event_id: int) -> dict[str, Any]:
        """More of one event: frames around it (the snapshot first), a face to search for."""
        found = [e for e in self.store.list_events(since=time.time() - 8 * 86400, limit=5000)
                 if e["id"] == event_id]
        if not found:
            raise HTTPException(404, "No such event")
        e = found[0]
        frames = [e["snapshot"]] if e.get("snapshot") else []
        frames += self.store.thumbs_near(e["camera"], e["time"], e.get("track_id"),
                                         limit=3 - len(frames))
        if len(frames) < 3:  # the track's own crops are not enough: the scene around it
            frames += [r for r in self.store.thumbs_near(e["camera"], e["time"], limit=3)
                       if r not in frames][:3 - len(frames)]
        face = None
        if self.engine.faces is not None and e.get("label") in (None, "person"):
            near = self.store.thumbs_near(e["camera"], e["time"], e.get("track_id"),
                                          window=120, limit=1, table="faces")
            face = near[0] if near else e.get("snapshot")
        return {"id": event_id, "frames": [f"/media/{r}" for r in frames], "face": face}

    # ------------------------------------------------------------------ ask

    def ask(self, question: str, event: int | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise HTTPException(400, "Empty question")
        calls: list[str] = []

        class Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                calls.append(call_text(record))

        tool_log = logging.getLogger("histor.today.tools")
        tool_log.propagate = False
        tool_log.setLevel(logging.INFO)
        handler = Collect()
        started = time.monotonic()
        with self._chat_lock:
            tool_log.addHandler(handler)
            tb = self.toolbox
            prefix = []
            if event is not None:
                about = [e for e in self.store.list_events(since=time.time() - 8 * 86400,
                                                            limit=5000) if e["id"] == event]
                if about:
                    d = describe(about[0])
                    prefix.append(f"[About this event: {about[0]['iso_time']} on "
                                  f"{d['camera']}: {d['title']} (type {about[0]['type']}, "
                                  f"rule {about[0].get('rule')})]")
            if self._untold:
                prefix.append(f"[Attached for face search: {', '.join(self._untold)}]")
            error = False
            try:
                if self._backend is None:
                    if not self.model:
                        raise RuntimeError("No chat model configured (chat.models).")
                    self._backend = create_backend(model_config(self.config, self.model),
                                                   system_prompt(tb), tb.tools)
                self._untold = []
                tb.evidence.clear()
                reply = answer(self._backend, tb, "\n".join([*prefix, question]), tool_log)
            except (RuntimeError, SystemExit) as e:  # model server down, missing SDK or key
                reply, error = f"Error: {e}", True
                self._backend = None
            finally:
                tool_log.removeHandler(handler)
            evidence = [f"/media/{rel}" for rel in dict.fromkeys(tb.evidence)
                        if self.store.media_path(rel)][:9]
        return {"calls": calls, "seconds": round(time.monotonic() - started, 1),
                "answer": reply, "evidence": evidence, "error": error}

    def reset(self) -> None:
        with self._chat_lock:
            self._backend = None
            self._untold = []

    def photo(self, data: bytes) -> dict[str, Any]:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if image is None:
            return {"ok": False, "message": "Not an image."}
        with self._chat_lock:
            name, message = self.toolbox.add_photo([image])
            if name:
                self._untold.append(name)
        return {"ok": bool(name), "name": name, "message": message}


def add_today_routes(server: FastAPI, today: Today, logo: str) -> None:
    page = PAGE.read_text(encoding="utf-8").replace("{{TOPBAR_CSS}}", TOPBAR_CSS).replace(
        "{{TOPBAR_JS}}", TOPBAR_JS).replace("{{TOPBAR}}", topbar_html("Today", logo))

    @server.get("/favicon.svg")
    def favicon() -> Response:
        return Response(logo, media_type="image/svg+xml")

    @server.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @server.get("/api/today")
    def api_today() -> dict[str, Any]:
        return today.day()

    @server.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        return today.summary()

    @server.get("/api/status")
    def api_status() -> dict[str, Any]:
        return today.status()

    @server.get("/api/events/{event_id}")
    def api_event(event_id: int) -> dict[str, Any]:
        return today.event(event_id)

    @server.post("/api/events/{event_id}/seen")
    def api_seen(event_id: int) -> dict[str, Any]:
        today.store.mark_seen(event_id)
        return {"ok": True}

    @server.post("/api/ask")
    async def api_ask(request: Request) -> dict[str, Any]:
        body = await request.json()
        event = body.get("event")
        # the model takes seconds to minutes: off the event loop
        return await anyio.to_thread.run_sync(
            today.ask, str(body.get("question") or ""), int(event) if event is not None else None)

    @server.post("/api/ask/reset")
    def api_reset() -> dict[str, Any]:
        today.reset()
        return {"ok": True}

    @server.post("/api/photo")
    async def api_photo(request: Request) -> dict[str, Any]:
        return today.photo(await request.body())

    @server.get("/media/{rel:path}")
    def media(rel: str) -> Response:
        path = today.store.media_path(rel)
        if path is None:
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})
