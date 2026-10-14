# Histor — Today screen (design handoff)

`Histor Today.html` is a self-contained, clickable mockup (open it in a browser). It replaces the
current Gradio home in `src/histor/ui.py`. Data in it is fake; wire it to the real store.

## Target
- Keep everything offline: system fonts only, no CDN (same rule as `engine/theme.py`).
- Suggested build: a FastAPI page (already used by `ui.py` via `gr.mount_gradio_app`) serving one
  HTML/JS page + JSON endpoints; or keep Gradio and rebuild the layout with `gr.HTML` + custom CSS.

## Layout (1440 × 900 reference, dark)
- Top bar 60px: logo (`Images/histor/web/favicon.svg`), nav Today / Live / Find a person / Ask
  (+ Test bench when Developer is on), "Recording N cameras", Developer switch.
- Main column (scrolls): date + one-paragraph written summary + figures; "Needs a look" cards
  (open alerts, Show / Mark as seen); "The day" card with filters (All, Alerts, People, Vehicles,
  Sounds) and a Strips | Grid switch; the journal (events by hour, newest first).
- Right column 420px: Ask chat (model status, lookups shown as tool calls, answer, evidence
  thumbs, example questions, input with "About <event>" context chip, Photo, Send).

## Behaviour
- Clicking a timeline mark, a grid cell with an event, or "Show" selects the event, expands it in
  the journal (3 evidence frames, Ask about this, Play clip, Find this person, Mark as seen) and
  scrolls it to the top.
- Strips: one lane per camera, 48 half-hour activity bars, event diamonds, white playhead at the
  selected event, cyan "now" line, hatched future.
- Grid: cameras × 24 hours, blue alpha = visits, diamond = event in that hour.
- Mark as seen: removes from "Needs a look", journal tag goes "Needs a look" → "Seen".
- Developer mode: track ids, scores, models, per-cell counts, Test bench nav item.

## Data sources (existing code)
- Events: `engine.store.list_events()`; plates: `store.plate_rows()`; counts: `store.counts()`.
- Thumbnails: `store.media_path(rel)`; per-camera activity: sightings bucketed by 30 min / 1 h.
- Summary paragraph: `histor.report` (Reporter) for "today so far".
- Ask: `histor.chat.answer` with `Toolbox`; show logged tool calls as in `ui.py` `send()`.

## Tokens (from engine/theme.py)
bg #080d1f · panel #0b1230 / #0e1530 / #161e3d · line rgba(151,161,186,.16)
text #eef1f7 / #c9d0df / #97a1ba / #69738f · blue #2864ff (hover #1a4fe6) · cyan #6fd8f7
alert #ef4444 · rule #f59e0b · plate #6fd8f7 · other #6089ff · ok #22c55e
