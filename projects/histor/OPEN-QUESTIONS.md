<img src="../../Images/brand-kit/logo/logo-horizontal-light.png" alt="Mobius Tensors" width="200" align="right" />

# Histor — Open questions

*Histor is a Mobius Tensors product.*

The things we still need to think about or decide, collected from [DOCUMENTATION.md](DOCUMENTATION.md),
the [README](README.md) and the [Partner Program study](<docs/Partner%20Program%20—%20Study.md>)
(October 2026). Each line points to where the topic is explained. When a question is settled, move
the answer into the documentation and delete it here.

## Business

- **Price list.** We have none yet; the partner example assumes $3,000 setup + $3,840 a year for
  8 cameras. What do we charge for setup, licence per camera, opt-in features (faces, plates,
  sounds) and support? (Partner study)
- **Who runs the partner programme now that Histor moves to the partner company?** Do the Silver /
  Gold / Platinum rates (10–30 %) still hold, and who signs the agreements and pays? (Partner study)
- **Partner agreement:** needs writing and a lawyer's check; and a partner kit (brochures, one-page
  pitch, demo video, demo kit at cost). (Partner study)
- **Pilot offer:** index a sample of the client's footage, then a week on live cameras. What does a
  pilot cost, how long does it last, and what result turns it into a sale? (section 11)
- **Positioning against NVIDIA's blueprint:** "many scenarios on a CPU already on site" vs "one
  scenario on a data-centre GPU". Is that the line we sell on? (section 49)

## Legal and privacy

- **Plate finder licence (urgent).** The current weights were trained with a fork of GPL-3.0
  YOLOv9: trials only. Train our own YOLOX on Open Images, or get a legal check first? (top of
  DOCUMENTATION, section 23, roadmap #1)
- **RTMPose-t** (fast pose model): training data partly research-only. Keep it trials-only, or find
  a small COCO-only model? ViTPose-B ships, but is 10× slower. (sections 14, 47.4)
- **Face search:** written client approval is the rule. What does the approval form say, and who
  keeps it? (section 7)
- **Plates and sound:** personal data in many countries, and listening is often regulated more than
  video. Which countries / sites allow them, and what do we tell the client? (sections 7, 47.6)
- **Nemotron and Cosmos models** (NVIDIA Open Model License: revocable, ends if guardrails are
  bypassed). Do we ever ship them, or only benchmark? (sections 25.1, 48)
- **Cloud chat (Claude):** it receives times, cameras, labels, file names and the `about:` text.
  Is that acceptable per client, and how do we say it in the contract? (sections 7, 25.2, 25.3)
- **Fire alerts are not a fire alarm** (NFPA 72 / UL 268B). How do we word it in sales material
  and contracts? (section 45.1)
- **Weapons:** keep "not offered". If a client insists, the pilot plan needs staged data and a
  guard in the loop; who carries the liability? (sections 46, 48)

## Product and features

- **Fire and smoke.** Plug in our own fire & smoke model: classes, input size, training data
  licence, and the composite detector that runs it next to YOLOX. Then the CLIP smoke score, the
  persistence check and alert checking on top. (sections 44, 45.4, roadmap #3–4)
- **Threat score on CLIP vectors** ("any fights last night?"): one vote towards an alert, never an
  alert alone. Which prompts, and what threshold? (section 48)
- **New rules:** running / sudden motion, and tailgating (two people through a door line within N
  seconds; with badge events: "two people on one badge"). (section 48, roadmap #6)
- **Fights:** pose + scream sound + alert checking as candidates first; a skeleton model needs
  labelled clips we own (RWF-2000 is non-commercial). (sections 47.9, 48)
- **Objects a client names** ("forklift", "pallet"): open-vocabulary auto-labelling, then a YOLOX
  per client. Worth the effort for which clients? (section 47.5, roadmap #9)
- **Multi-camera timeline** ("where did they go next?"): start with CLIP vectors + camera layout;
  re-ID models need client approval, like faces. (section 47.7, roadmap #10)
- **Safety gear (PPE)** for industrial clients: train on SHWD / Hard Hat Workers. Is there demand?
  (section 47.8)
- **Channels:** Telegram is done (alerts and a bot, section 17.1). Next: WhatsApp, SMS or an
  on-site siren? (section 8, roadmap #7)
- **Dashboard and login.** A first dashboard (live alerts, camera cards, search) and a token login
  are in the code (`engine/dashboard.py`) and documented (section 27.1). Does the chat go into
  it too? Is one shared token enough, or do sites need a login per person? (sections 10, 27,
  roadmap #7)
- **MCP over HTTP with login**, for agents that can't start a command. (section 25.3)
- **Reports:** set them up from the UI; let the chat answer "what did last night's report say?".
  (section 9)
- **Re-index** footage recorded with the old CLIP model. (section 9)
- **Per-scene captions** with a very small vision model (SmolVLM2, Florence-2): useful, or too heavy
  on a CPU? (section 47.2)
- **Retention purging** only runs inside `server.py`: should the recorder do it too, so a site
  without the API still purges? (section 10)

## Updates on client servers

Not documented yet. Today a site is installed with `git clone` + `uv sync` (section 35), so an
update means someone logs in and runs commands. Clients can't run code and we can't visit every
site. The idea discussed (October 2026):

- **Each server checks in with our update server** once a night, connecting out over HTTPS. No
  ports are open on the client's side.
- **Nothing installs on its own.** A server installs only a release we have assigned to that site
  in our dashboard (a few test sites first, then the rest).
- **Paid sites only.** Our update server sends nothing to a site whose subscription has lapsed. A
  signed licence file on each server gives the paid-until date; after it, the server refuses
  updates and new models, even from USB. **Open:** does an unpaid site keep running on its current
  version (suggested: yes, so a break-in is still caught, which avoids liability), or stop?
- **Install next to the old version, undo if broken:** check that cameras record and models load,
  otherwise switch back and report to us. `site.yaml`, `.env`, store and recordings live outside
  the release folders.
- **Models update separately** (e.g. fire & smoke), versioned with a checksum each.
- **Offline sites:** a USB stick by post that installs itself when plugged in. Or require outbound
  internet at sale time?
- **Ready-to-run servers** shipped from one disk image, so the installer only connects cameras.
- **Dashboard of sites:** version, last check-in, cameras online, failed updates, paid-until.
- **Security:** releases signed with a key kept offline, never on the update server. The
  remote-support tunnel (Tailscale or similar) is off by default and turned on per site when
  needed, with two-factor login for named staff only, and every session logged. Each server has
  its own key. The weak point is our own signing key and staff accounts.
- **Effort:** about 6–8 weeks for one developer for everything; a minimum version (signed releases,
  update agent with undo, simple admin page with paid status set by hand, Tailscale) about 2–3
  weeks. **Check Mender first** (open-source over-the-air updates, Apache-2.0: signed updates,
  undo, device dashboard), which could cut it to ~2–3 weeks with only the licence check added.

## Protecting our code on client servers

Not planned yet (idea raised October 2026). Today a site gets the full Python source and our
trained models (`git clone`, section 35), so anyone with access to the server (client IT, an
installer, a partner) can copy them and run Histor elsewhere. We need a plan before the first
paid site. Nothing stops someone with full control of the machine for good; the goal is to make
copying costly and clearly illegal. Options to weigh:

- **Contract first:** licence agreement for clients and partners (no copying, no reverse
  engineering, no moving to other machines), with a penalty. Cheapest and needed anyway.
- **Ship no readable source:** compile the Python (Nuitka or Cython) or ship the planned C++
  port; ready-to-run servers from one disk image (see updates above).
- **Protect the models:** our own trained models (fire & smoke first) encrypted on disk and
  decrypted in memory with the site's licence key. These are the hardest part to rebuild.
- **Tie the licence to the machine:** the signed licence file (updates section) also names the
  server's hardware ID, so a copy on another machine refuses to start.
- **Limit who logs in:** client IT gets the dashboard, not a shell; installers get time-limited
  accounts; full-disk encryption so a pulled disk is unreadable; every remote session logged.
- **Watermark** builds and models per site, so a leaked copy shows where it came from.

Open: how far to go before the first sale, and which of these partners must accept in their
agreement.

## Models to choose

- **Chat model:** `qwen3:4b` (default) vs `qwen3.5:4b` (tools + vision in one), Gemma 4 E4B,
  Granite. Needs the fixed set of evaluation questions on recorded footage first. (sections 25.1,
  47.2, roadmap #8)
- **Alert-checking model:** Qwen3.5 4B vs Cosmos-Reason2-2B, in `tag` mode. (section 48)
- **Which alerts to check:** each check costs seconds on a CPU. Rare and costly ones only (person
  down, abandoned bag, area at night)? (section 18)

## To test on real sites

Everything below was checked on synthetic input, a stand-in server or the sample video only.

- **Alert checking** on a real Ollama, then a week in `tag` mode: false alarms held back vs real
  alerts wrongly held back, per event type. This is the number to quote. (section 18, roadmap #2)
- **Person down** on staged falls; false alarms per day. (sections 8, 22.5, roadmap #5)
- **Face thresholds** calibrated on CCTV footage; cameras placed at doors at head height.
  (section 21.5)
- **Plates:** read rates on real cameras (blur, night, angle). (section 23, roadmap #11)
- **Sounds:** false alarms on real camera audio. (section 24, roadmap #11)
- **Pose facts** (bending, head down, motion, carrying): unit tests only so far. (section 22.5)
- **MCP from Claude Desktop** over SSH, end to end. (section 25.3)
- **Acceptance targets:** false alarms per camera per day, agreed with the client before each
  pilot. (sections 11, 45.5, 46.3)
