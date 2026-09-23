# Roadmap — Steps to Make This Happen

High-level plan tying together [Ideas.md](Ideas.md) (*what* to detect) and the [Implementation Plan](Implementation%20Plan%20—%20From%20Trained%20Model%20to%20App.md) (*how* the serving pipeline works). Current state: only `ultralytics` + `opencv-python` are set up in `server/` — everything below is still to build.

## 1. Pick the First Use Cases

Don't start with all ~40 ideas at once. Pick 1-3 from [Ideas.md](Ideas.md), favoring the 🟢 **Mature** ones (e.g. Motion Detection, Car Plate Reading, Fall Detection, PPE Check) over 🔴 **Experimental** ones (Gun, Shoplifting, Crash) — the mature ones need less data/tuning to get to something usable, and prove the pipeline end-to-end faster.

Define, per use case: target accuracy, acceptable false-positive rate, and what a "confirmed event" means.

## 2. Collect & Label Data

- Gather footage per use case — existing CCTV clips, public datasets, or synthetic/augmented data where real footage is scarce
- Label with a tool (Roboflow, CVAT, Label Studio)
- Split into train / validation / test sets

## 3. Train Models

- Train with Ultralytics YOLO (`yolo train`) per use case
- Validate against the benchmarks noted in [Ideas.md](Ideas.md)'s Trust Level column
- Iterate — augment data, tune hyperparameters, retrain — until it clears an internal accuracy bar

## 4. Package & Optimize for Serving

Pick a runtime based on the target hardware — no need to commit to one for the whole platform:

| Option | Hardware | Notes |
| --- | --- | --- |
| **OpenVINO** | CPU (+ Intel GPU/VPU) | Matches the "commodity server, no GPU required" starting point. `yolo export format=openvino` |
| **ONNX Runtime** | CPU, NVIDIA GPU (CUDA), AMD (ROCm), Intel | Hardware-agnostic middle ground — one export format, pick the execution provider per box. `yolo export format=onnx` |
| **TensorRT** | NVIDIA GPU only | Fastest option per-stream, but locks you to NVIDIA hardware. `yolo export format=engine` |
| **NVIDIA DeepStream** | NVIDIA GPU only | A full video-analytics pipeline (GPU-accelerated decode + TensorRT inference + batching), built for many camera streams per box rather than one model at a time — worth it once you're scaling GPU servers, overkill for a pilot |

Start with OpenVINO for the pilot; move a use case to TensorRT/DeepStream later if it needs GPU-level throughput (e.g. many camera streams, a heavier model, or lower latency than CPU can hit).

- Benchmark real inference speed on the target server hardware, whichever runtime is chosen

## 5. Build the Inference Service

- One always-running Python worker per camera/model: load model once, loop (grab frame → predict → publish result)
- Handle camera ingestion (RTSP via OpenCV)

## 6. Wire Up Messaging (ZeroMQ)

- Inference workers `PUB`lish raw detections; other processes `SUB`scribe
- No broker to install — just a library (`pyzmq`) each process imports

## 7. Event Logic & Storage

- A service applies business rules (confidence thresholds, N-frames-in-a-row confirmation) to turn raw detections into confirmed events
- Confirmed events + snapshots get written to a database (start with SQLite, move to PostgreSQL)

## 8. Backend API

- FastAPI service exposing REST endpoints (`GET /events`, `GET /cameras`) for history
- WebSocket endpoint subscribing to the ZeroMQ alert channel, pushing live events to connected clients

## 9. Web / Mobile App

- Web dashboard: calls REST for history, opens WebSocket for live alerts
- Mobile app: same REST API, plus push notifications (e.g. Firebase) for high-priority events

## 10. Deploy & Pilot

- Deploy on a commodity, CPU-only server (per the platform's "no GPU required to start" goal)
- Run against a real camera/site for a trial period
- Track real-world false-positive / false-negative rates — compare against the Trust Level assumptions in [Ideas.md](Ideas.md)

## 11. Iterate & Expand

- Feed pilot false positives/negatives back into training data, retrain
- Once the pipeline is proven on the first use cases, add more from [Ideas.md](Ideas.md) (agriculture, retail, workplace safety, etc.)
- Update [Ideas.md](Ideas.md)'s Trust Level column with real numbers as they come in, instead of published benchmarks

## Suggested order of work right now

1. Pick the use case (Step 1)
2. Data collection + labeling (Step 2)
3. Train + validate first model (Step 3)
4. Minimal single-camera, single-model script — no ZeroMQ/DB/API yet — just prove detection works live (Steps 4-5, scoped down)
5. Only then move to Steps 6-9, matching the "Minimal first version" section in the [Implementation Plan](Implementation%20Plan%20—%20From%20Trained%20Model%20to%20App.md)
