# Argus — Product Description

*After Argus Panoptes, the hundred-eyed watchman of Greek myth (πανόπτης, "all-seeing"), who never closed all his eyes at once.*

Every claim below was checked against the code in [projects/argus](../../projects/argus/) and the
object-detection modules it reuses (October 2026). Keep it that way when editing: if a feature is
on the roadmap, say so.

## One line

Argus lets you ask your cameras what happened, in plain language, and answers with the frames
as evidence.

## Short (website, brochure, LinkedIn)

Most camera AI answers one question: "is X happening right now?". Argus answers a different one:
"what happened?". It keeps a searchable memory of what your cameras see (every tracked object,
plus regular snapshots of each scene), so you can ask later: *"Was there a white van at the gate
yesterday evening?"*, *"How many cars came in today?"*. It answers in sentences and lists the
frames it checked. It can also alert you when something happens in an area you choose. It runs on
your own server, without a GPU, and by default even the language model that answers runs locally.
Your footage stays in thse building.

## Full description

### The problem

Sites record far more video than anyone watches. When something needs finding, a recorder lets
you jump to a time, not to a moment: someone scrubs through hours of footage. Alerts help only for
the rules set up in advance; today's question was usually not one of them.

### What Argus does

1. **Record.** One recorder per camera detects and tracks every object across the 80 everyday
   object types the detector knows (people, vehicles, animals, bags, furniture…). Each tracked
   object is stored as a searchable description every couple of seconds, and every few seconds
   the whole scene is stored too, so things the detector has no name for can still be found.
2. **Ask.** A question is split into *what*, *when* and *which camera*. Time ranges can be plain
   words ("yesterday 6pm to 9pm", "this morning", "last 2 hours", "last friday") or exact times.
   Sightings of the same tracked object are merged into one visit.
3. **Answer.** A language model looks things up in the store (up to six rounds of lookups per
   question), answers in sentences, and lists the thumbnails it looked at as evidence.

### Two kinds of answer, and it says which

| Kind                         | How                                                                                                                           | Good for                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Counted (exact)**    | Straight from the detector and tracker: what, when, how many. One tracked object = one visit.                                 | "How many cars came in today?", "What happened 6pm to 9pm?", "Anyone with a dog this morning?" |
| **Described (ranked)** | Visual search (CLIP) by appearance, colour, clothing, or things with no detector class. Ranked best matches, not certainties. | "A man in a red jacket near a car?", "A white van at the gate?"                                |

"Together" ("a person with a dog") means the same camera at the same time (within 15 seconds by
default), not touching in the image.

### Private by design

- Runs on the site's own server; cameras stay on the site network.
- One memory store per site, on that server. Retention is a setting (30 days by default for
  sightings; recordings 7 days by default).
- The default chat model runs locally (Qwen3 4B through Ollama, Apache-2.0, on CPU). A larger
  local model (Qwen3 30B-A3B) needs about 20 GB of RAM or a GPU.
- A cloud model (Claude) is an opt-in switch. It receives the question and the text search
  results (times, cameras, labels, thumbnail file names), never images or video.
- Face recognition is off unless the client has approved it (face prints are biometric data).

### On site

- **Hardware:** a standard server. Detection (YOLOX) and search (CLIP) run on CPU through ONNX
  Runtime; no GPU required.
- **Cameras:** any IP camera with RTSP. The detector reads the light sub stream; the full-quality
  main stream can be saved untouched in one-minute files (needs ffmpeg 5.1+), and its keyframes
  give sharp crops for search results.
- **Footprint:** describing one object takes about 55 ms on a laptop CPU; each sighting stores a
  2 KB search vector plus a small thumbnail. Recording at 1080p, 4 Mbit/s is about 43 GB per
  camera per day. On busy scenes Argus samples more sparsely rather than falling behind.
- **Model choice:** any OpenAI-compatible server (Ollama, llama.cpp, LM Studio, vLLM) is a config
  entry, no code change.

### Alerts when it matters

Argus also alerts, from the same recorder that builds the memory: alerts for the rules set in
advance, memory for every other question.

- **Areas and lines:** draw zones and tripwires on each camera. Events: an object *enters*,
  *leaves*, *stays longer than* N seconds (e.g. a car parked in a fire lane), or *crosses* a line
  (in, out or both directions).
- **Per customer:** each rule is limited to chosen object types (only `person` in the yard, only
  `truck` at the gate), and every site has its own rules in its config.
- **Anywhere in the picture:** a zone without points covers the whole frame ("a dog appears").
- **Delivery:** channels, each with its own filter: which rules, events and objects, at which
  hours ("22:00-06:00"), and a cooldown so nobody gets flooded. Webhook (into the client's own
  systems, or Slack/Teams/n8n/Zapier) and email work today, with a JPEG snapshot of the moment.
- **Ask about alerts:** every event is stored with its snapshot, whatever the channels sent, and
  the chat has an `alerts` tool: "did any alert go off last night?", "who entered the yard?".
- **Tested (2026-10-06):** Argus's own recorder on the sample street video, a whole-frame zone
  and a line: every event stored with a snapshot (102), a webhook channel filtered to the line
  with a 30 s cooldown received 3 messages, and the chat tool listed the events for "last night".
  Email is covered by unit tests (no real mail server in that run).
- **Setup:** the `zones:` and `alerts:` sections in `projects/argus/configs/default.yaml` (empty by
  default, with commented examples).
- **Next channels:** Telegram, WhatsApp (Business API, pre-approved templates), SMS and an
  on-site siren or relay; each is one small class in the shared alert code (overview in the
  object-detection README).

**Not available yet:** rules only work on the 80 detector types. Fire and smoke, or any object a
customer names that is not one of the 80, need a model trained for it (the code loads custom
models; the training step is still to build). Alerts from a description ("a white van at the
gate") would reuse the visual search, but its matches are ranked, so each site would need tuning.
Weapons are not offered: no model, and both false alarms and misses are costly.

### Where it is today

**Working now**

- Recording and indexing per camera, all 80 object types plus scene snapshots.
- Plain-language questions from the command line (`ask.py`), no language model needed.
- Chat (`chat.py`) with switchable models, local by default; evidence listed as thumbnail files.
- Main-stream recording with full-resolution crops.
- Alerts by area and line (or anywhere in the picture), per object type, sent to channels with filters (rules, events, objects, hours, cooldown); webhook and email today. The chat answers about past alerts.

**Next**

- A fixed set of test questions on recorded footage, to compare models.
- Scene captions from a small vision-language model, for questions CLIP handles badly (counting,
  "what was he doing").
- A multi-camera timeline: following one visit across cameras.
- Detectors trained per customer, such as fire and smoke.
- Alerts from a description, built on the visual search.

### Limits worth saying out loud

- "Everything" is sampled: each tracked object every 2 s, each scene every 5 s, and objects
  smaller than 40 px tall are not described. All three are tunable per site.
- Visual search results are ranked guesses; the evidence frames are there so a person can check.
- Visit counts depend on the tracker keeping an object in view; a long occlusion can split one
  visit into two.
- Small local models follow tool instructions less reliably than large ones.
- Retention purging runs inside the HTTP server (`object-detection/src/server.py`), so a site
  needs that service running for old sightings to be deleted.

### How to try it

A pilot on the client's own recorded footage: we index a sample, they ask their own questions,
then they decide.
