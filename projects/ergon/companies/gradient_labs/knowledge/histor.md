# Histor

Histor is a Gradient Labs product. Ask your cameras what happened, in plain language, and get
the frames as evidence. Status: working code, pilot-ready on recorded footage; live alerts to be
validated at a pilot site.

## The problem it solves

Sites record far more video than anyone watches. Footage is usually opened only after something
went wrong, and a recorder lets you jump to a time, not to a moment, so someone scrubs through
hours of video. Classic camera AI only answers "is X happening right now?" for rules set up in
advance, and its false alarms (a shadow, a coat on a chair) train guards to stop looking.

## What Histor does

- Record: every tracked object (80 everyday object types: people, vehicles, animals, bags,
  furniture...) is stored as a searchable description every couple of seconds, plus a snapshot of
  each scene every few seconds, so things with no detector name can still be found.
- People: about once a second each person's body pose becomes facts: posture (standing, sitting,
  bending, lying), hands raised, walking or running, a bag or suitcase on them, colour of top and
  trousers.
- Ask: questions in plain language with plain time ranges ("yesterday 6pm to 9pm", "last
  friday"). A language model looks things up and answers in sentences, listing the thumbnails it
  checked as evidence.
- Alert: areas and lines drawn on each camera (enter, leave, stay too long, cross, wrong way),
  crowding and queues, bags left behind, camera covered or turned, person down, hands up, number
  plates on a watchlist, sounds (smoke alarm, breaking glass, scream, siren...), and fire and
  smoke.
- Second look: optionally a local vision model checks each alert's snapshot before it is sent
  and holds back likely false alarms (it can veto an alert, never raise one).
- Deliver: webhook (into the client's systems, Slack, Teams, n8n, Zapier), email and Telegram,
  each with its own filters, hours and cooldown. A dashboard with a token login shows live alerts,
  camera status and search. WhatsApp, SMS and an on-site siren are next.
- Telegram bot: ask Histor from a phone. MCP server: other AI agents can use Histor's tools.
- Reports: hourly, daily or weekly summaries of what chosen cameras saw.

## Example questions

- Gates and car parks: "How many trucks came in this morning?", "When did AB 123 CD leave?"
- Warehouses and yards: "Was anyone at the loading bay yesterday evening?", "Did anyone fall?"
- Campuses and public spaces: "When did a bag first appear by the bench?"
- Construction sites: "Was there a white van at the gate yesterday evening?"
- Shops and offices: "How long was the queue at the till at noon?", "Was any camera covered?"

## Kinds of answer

- Counted (exact): what, when, how many, straight from the detector and tracker.
- Described (ranked): visual search by appearance, colour, clothing. Best matches, not certainties.
- Pose (measured): posture, activity and clothing colours per visit.
- Plate (exact, opt-in): plate text, forgiving look-alike characters.
- Face (scored, written approval only): a photo compared with faces on the cameras.

## Private by design

- Runs on the site's own server. Footage never leaves the building.
- No GPU needed (ONNX Runtime on CPU). A GPU can be added for more cameras per server or heavier
  models, but is not required.
- The chat model runs locally by default. A cloud model is an opt-in switch and only ever receives
  text (times, cameras, labels), never images or video.
- Faces, plates and sound are off until the client approves them. Face prints are deleted after
  7 days. Retention is a setting (30 days for sightings and events, 7 days for recordings).
- Commercially clean: code, weights and training data under permissive licences (Apache-2.0,
  MIT). Two components are marked trials only until replaced: the plate finder and one pose model.

## Fire and smoke

Fire and smoke detection has been in Histor since 2026-10-16, with our own trained model. It
complements certified smoke and heat detectors and never replaces them. Known gaps: fires that
fill the whole frame and tiny flames can be missed; fog, red lamps and sunlit clouds can cause
false alarms. Not yet tested on real camera fire footage.

## Honest limits

- Sampling: each object every 2 s, each scene every 5 s; objects under 40 px tall are not
  described.
- Visual search results are ranked guesses; the evidence frames let a person check.
- Person down has not yet been validated on real or staged falls; validate on site first.
- Plates and sounds have only been checked on synthetic input, not real cameras yet.
- One shared login token per site; no per-person accounts yet.
- Weapons detection is not offered.

## How a customer tries it

A pilot on the client's own recorded footage: we index a sample, they ask their own questions,
then decide. For live alerts, a week on their cameras gives the number that matters: false alarms
per camera per day, with and without alert checking. A partner programme (Silver, Gold, Platinum)
is drafted for introducers and CCTV installers.
