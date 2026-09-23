# Ideas

Things this platform could detect or watch for, using cameras.

**Trust Level** is a rough read of how reliable the detection is today, based on published benchmarks/industry reports (not our own testing):

- 🟢 **Mature** — solid, well-established, production-ready in good conditions
- 🟡 **Needs tuning** — works, but accuracy varies a lot by setup/site and needs calibration
- 🔴 **Experimental** — high false-positive rate, not reliable enough to act on alone yet

## Safety

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Fire & Smoke Detection | | 🟡 Needs tuning — mAP 74-90%, up to ~95% on some datasets | Confused by fog/steam/dust; thin/transparent smoke often missed |
| Gun Detection | | 🔴 Experimental — 84-94% on clean benchmarks, ~73% mAP on realistic ones | Real CCTV footage is far lower quality than benchmark images |
| Shoplifting Detection | | 🔴 Experimental — 89.5% precision but only ~59% recall (zero-shot); ~85% acc (older CNNs) | Behavior-based, not object-based — highly pose/scene dependent |
| Knife Detection | | 🟡 Needs tuning — mAP 70-99% across datasets | Small object at range, motion blur; benchmarks skew toward close-up shots |
| Fall Detection | for elderly care or hospital rooms | 🟢 Mature (controlled settings) — 96-99.7% accuracy | Benchmarks use staged falls; cluttered real environments perform worse |
| Perimeter Watch | see if someone crosses a fence or line they shouldn't | 🟡 Needs tuning — near-100% for simple line-crossing rules | Degrades with weather, distance, low light; animals/foliage cause false alarms |
| Loitering Detection | someone staying too long in one spot | 🟡 Needs tuning — rule-based, no formal benchmark | Very threshold-sensitive: too short = false alarms, too long = misses |
| Fight Detection | | 🟡 Needs tuning — 77-94% accuracy (RWF-2000 benchmark) | Benchmark clips are staged; real scenes (rough play vs. fighting) are ambiguous |
| Left Bag / Package Detection | spot bags left alone | 🟡 Needs tuning — ~91-96% precision/accuracy | Clutter/occlusion in busy scenes; ambiguity in who "owns" a bag |
| Tailgating Detection | more than one person walking through a door meant for one | 🟡 Needs tuning — no independent published benchmark, vendor claims only | Crowded/rapid entries and camera angle strongly affect accuracy |
| Face Mask Check | | 🟢 Mature — 90%+ across multiple models (MAFA dataset) | Accuracy drops at side angles/occlusion, dark masks vs. skin tone |
| Crowd Size Detection | too many people in one place | 🟡 Needs tuning (moderate density); 🔴 Experimental at extremes | Accuracy collapses when very sparse (false positives) or very dense (occlusion) |
| Lonely Kid Detection | child alone in a supermarket, garden/park, etc., with no adult nearby for a set dwell time (e.g. 50 seconds, or a few minutes for a lower-alert tier) | 🔴 Experimental — no established benchmark; combines child/adult age classification (~85-90% on coarse bins) with person tracking to check for a nearby adult | Age estimation from video is unreliable at a distance; a dwell timer helps avoid false alarms from brief occlusion, but the right threshold is scene-dependent and still needs tuning |

## Traffic & Vehicles

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Car Plate Reading | | 🟢 Mature (good conditions) — 93-95%+ recognition accuracy | Degrades a lot in rain/fog/low-light or non-standard plates |
| Car Speed Check | | 🟢 Mature (well-calibrated) — median error ~0.58 km/h (BrnoCompSpeed) | Very sensitive to camera calibration — a bad setup skews every reading |
| Wrong-Way Driving | | 🟡 Needs tuning — vendor claims ~99%, but one independent study found only ~61% real-world recall | Big gap between vendor marketing numbers and field-test results |
| Running a Red Light | | 🟡 Needs tuning — 82-100% across studies, ~92% typical | Needs precise camera/trigger calibration for legal-grade evidence |
| Bad Parking | illegal or double parking | 🟡 Needs tuning — 91-95%+ accuracy | Sensitive to occlusion by other vehicles and lighting/weather |
| Vehicle Counting | how many cars pass, traffic flow | 🟢 Mature — 90-99% accuracy, beat induction loops in one field test | Degrades under heavy congestion, occlusion, bad weather |
| Crash Detection | | 🔴 Experimental — 88-97% on curated benchmarks, but zero-shot models "perform poorly" | Crashes are rare, so training data is scarce; poor generalization outside curated sets |
| Helmet Check | for motorcycle riders | 🟢 Mature — 93-95% accuracy/mAP | Occlusion, distance/resolution, other headgear resembling helmets |
| Seatbelt Check | | 🔴 Experimental — claims of 98-99% but only 56-63% mAP on unseen vehicle types | Poor generalization across camera angles/interiors; windshield glare |
| Parking Spot Checker | is a spot free or taken | 🟢 Mature — 97-98% accuracy, AUC 0.99 | Shadows, weather, and camera drift reduce accuracy over time without recalibration |
| Vehicle Type Sorting | car, truck, bus, motorcycle | 🟢 Mature (clean images) — 90-97.7% top-1 accuracy | Benchmarks assume clean, near-frontal images — real road angles are less tested |
| ArUco / AprilTag Vehicle Tracking | sticker/tag codes on vehicles for exact tracking of position and speed (good for yards, fleets, test tracks) | 🟢 Mature — deterministic marker detection (not ML), near-100% under good conditions | Sensitive to motion blur, low light, and marker occlusion |

## Retail & Business

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Queue Length Check | how long the line is | 🟡 Needs tuning — no standardized benchmark; occlusion is the main issue | Crowded checkout occlusion; telling "waiting in line" from "passing through" |
| People Counting | how many people walked in | 🟢 Mature — 95-99.5% (3D stereo); legacy 2D/thermal only 80-90% | Overlapping people/groups reduce accuracy; big gap between tech tiers |
| Where People Walk | heatmap of customer movement in a store | 🟡 Needs tuning — no standalone accuracy benchmark (derived from tracking) | Cross-camera re-identification errors compound tracking drift |
| Empty Shelf Check | | 🟡 Needs tuning — 83.7-86.3% AP in published studies | Dense/cluttered shelves, viewing angle, and lighting reduce accuracy |
| Checkout Fraud Check | catching cashier tricks like not scanning items | 🟡 Needs tuning — vendor case studies only (e.g. ~90% loss reduction), no peer-reviewed benchmark | Evidence is mostly vendor ROI claims; false positives burden staff review |

## Factory & Workplace Safety

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Safety Gear Check | hard hats, vests, gloves | 🟡 Needs tuning — wide range, 66-97% depending on detector | Distance, lighting, partial occlusion cause big swings between studies |
| No-Go Zone Check | someone entering a zone they shouldn't | 🟡 Needs tuning — ~92% frame-level accuracy, F1 ~0.92-0.93 | Needs careful zone/geofence calibration per site |
| Spill / Leak Detection | | 🔴 Experimental (RGB) / 🟡 Needs tuning (thermal, 90-99%) | RGB struggles with low contrast/reflections, generalizes poorly across sites |
| Tired Worker Detection | drowsiness | 🟡 Needs tuning — 94-97.6% accuracy (driving-specific benchmark) | Benchmark is driving-focused; general workplace poses/lighting differ a lot |

## Environment & Public Space

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Flood Water Level Check | | 🟡 Needs tuning — 84-91% precision/recall, ~21cm error (drone-based) | Water's changing appearance (turbidity, reflections, season) hurts reliability |
| Illegal Dumping / Littering | | 🔴 Experimental — wide range, 74-99.5% depending on method | Fast movements missed; garbage confused with bare soil; little real-world validation |
| Animal Intrusion | animals on farms or reserves | 🟡 Needs tuning — F1 ~96%, presence accuracy 90-97% | Similar-species confusion, small/distant animals, background clutter |
| Graffiti / Vandalism Detection | | 🔴 Experimental — wide range, 66-99% depending on class | Confuses legal murals/street art with vandalism |
| Crop Health Check | using drone footage | 🟢 Mature — 89-99%+ (established remote-sensing technique) | Varies with cloud cover, growth stage, overlapping disease symptoms |
| Smoke / Emissions Check | for factories | 🟡 Needs tuning — ~94% accuracy | Confused with steam/shadows/glare; semi-transparent smoke often missed |

## General Monitoring

| Idea | Notes | Trust Level (Benchmark) | Caveat |
| --- | --- | --- | --- |
| Motion Detection | | 🟢 Mature — modern "smart" motion detection 98-99% accuracy | Basic (non-AI) motion detection still fooled by shadows, foliage, light changes |
| Area Watch | watching a zone for people entering or being present | 🟡 Needs tuning — near-100% for simple, well-calibrated zone rules | Weather, distance, and camera angle degrade reliability, same as Perimeter Watch |

---

*Sources: synthesized from published benchmarks and papers (RWF-2000, ShanghaiTech, PKLot/CNRPark, NTHU-DDD, MAFA, PETS2006/AVSS, BrnoCompSpeed, Stanford Cars/CompCars, DCSASS, ACCIDENT benchmark, FASDD/D-Fire, OPIXray, etc.) plus vendor whitepapers, as of Sept 2026. The clearest overall pattern: simple object-presence tasks (plates, PPE, parking occupancy, vehicle type, fall detection) are the most mature; behavior/intent-inference tasks (shoplifting, loitering, fight detection, tailgating, checkout fraud) remain the least mature, mainly due to scarce real-world labeled data and inherent ambiguity in the behavior itself. Treat all of these as figures to sanity-check before relying on for our own use cases, not ground truth.*
