# Histor — Product Description

*Greek ἵστωρ: "witness, one who knows by having seen". The root of "history".*

Every claim below was checked against the code in [projects/histor](../../projects/histor/) and the
object-detection modules it reuses (October 2026). Keep it that way when editing: if a feature is
on the roadmap, say so.

## One line

Histor lets you ask your cameras what happened, in plain language, and answers with the frames
as evidence.

## Short (website, brochure, LinkedIn)

Most camera AI answers one question: "is X happening right now?". Histor answers a different one:
"what happened?". It keeps a searchable memory of what your cameras see (every tracked object,
plus regular snapshots of each scene), so you can ask later: *"Was there a white van at the gate
yesterday evening?"*, *"How many cars came in today?"*. It answers in sentences and lists the
frames it checked. It can also alert you when something happens in an area you choose. It runs on
your own server, without a GPU, and by default even the language model that answers runs locally.
Your footage stays in the building.

## Full description

### The problem

Sites record far more video than anyone watches. When something needs finding, a recorder lets
you jump to a time, not to a moment: someone scrubs through hours of footage. Alerts help only for
the rules set up in advance; today's question was usually not one of them.

### What Histor does

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

| Kind | How | Good for |
| --- | --- | --- |
| **Counted (exact)** | Straight from the detector and tracker: what, when, how many. One tracked object = one visit. | "How many cars came in today?", "What happened 6pm to 9pm?", "Anyone with a dog this morning?" |
| **Described (ranked)** | Visual search (CLIP) by appearance, colour, clothing, or things with no detector class. Ranked best matches, not certainties. | "A man in a red jacket near a car?", "A white van at the gate?" |

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
  camera per day. On busy scenes Histor samples more sparsely rather than falling behind.
- **Model choice:** any OpenAI-compatible server (Ollama, llama.cpp, LM Studio, vLLM) is a config
  entry, no code change.

### Alerts when it matters

Histor also alerts, from the same recorder that builds the memory: alerts for the rules set in
advance, memory for every other question.

- **Areas and lines:** draw zones and tripwires on each camera. Events: an object *enters*,
  *leaves*, *stays longer than* N seconds (e.g. a car parked in a fire lane), or *crosses* a line
  (in, out or both directions).
- **Per customer:** each rule is limited to chosen object types (only `person` in the yard, only
  `truck` at the gate), and every site has its own rules in its config.
- **Delivery:** webhook (into the client's own systems) or email, with a JPEG snapshot of the
  moment; a cooldown holds back repeats.
- **Tested:** zones and lines with Histor's own recorder on the sample street video (2026-10-06):
  90 events (enter, exit, dwell, cross), stored in Histor's store. Webhook and email delivery come
  from object-detection and were not part of that run.
- **Setup:** add `zones:` and `alerts:` sections to `projects/histor/configs/default.yaml` (same
  format as object-detection, see its README).

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
- Alerts by area and line, per object type, by webhook or email.

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
