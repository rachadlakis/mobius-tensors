<div align="center">

<img src="Images/banner.jpeg" alt="Mobius Tensors banner" width="100%" />

# Mobius Tensors — Complete Architecture

2026-09-22 · @Someone

</div>

---

This is the single reference for **how the company, the product, and the system fit together** — business model down to server topology. It consolidates and cross-links the other two docs rather than replacing them:

- [Mobius Tensors — AI Solution Platform.md](Mobius%20Tensors%20—%20AI%20Solution%20Platform.md) — product vision, use cases, roadmap, open risks
- [Implementation Plan — From Trained Model to App.md](Implementation%20Plan%20—%20From%20Trained%20Model%20to%20App.md) — step-by-step build path, glossary
- [`server/detect.py`](server/detect.py) — the first working piece of this architecture, in code

## 1. Architecture Layers

Four layers, each depending only on the one below it — a change in how a use case works (Product layer) shouldn't require touching how servers are provisioned (Infrastructure layer), and vice versa.

```mermaid
flowchart TB
  subgraph L1["Business Layer"]
    B1[Platform company, not bespoke-project company]
    B2[One codebase sold to many clients]
  end
  subgraph L2["Product Layer"]
    P1[Use cases: Fire/Smoke, Plate, Zone Crossing, ...]
    P2[Each use case = model + ruleset plug-in]
  end
  subgraph L3["System / Platform Layer"]
    S1[Ingest → Inference → Event Logic → Alerting]
    S2[ZeroMQ bus connects every stage]
  end
  subgraph L4["Infrastructure Layer"]
    I1[Servers: CPU+OpenVINO to start, GPU when proven necessary]
    I2[Storage, networking, deployment]
  end

  L1 --> L2 --> L3 --> L4
```

## 2. Business Architecture

The company's structural bet: one shared platform serves every client, instead of a bespoke build per client.

```mermaid
mindmap
  root((Mobius Tensors))
    Product
      Platform core (shared)
      Use-case plug-ins
      Web + mobile app
    Technology
      YOLO detection
      ZeroMQ messaging
      OpenVINO / GPU inference
      FastAPI backend
    Operations
      Pilot → Production pipeline
      On-site fine-tuning per client
      Alert SLAs
    Go-to-market
      Security/monitoring clients
      New use case = new model, not new system
```

## 3. System Context (who talks to Mobius Tensors)

The platform as a black box, and everything around it that it has to interact with.

```mermaid
C4Context
  title System Context — Mobius Tensors Platform

  Person(ops, "Security Operator", "Monitors live view + alerts")
  Person(admin, "Client Admin", "Configures cameras, zones, thresholds")

  System(platform, "Mobius Tensors Platform", "Ingests video, runs detection models, raises alerts")

  System_Ext(cameras, "Client Cameras / NVR", "RTSP video streams")
  System_Ext(vms, "Client VMS / Security System", "Existing on-site security infrastructure")
  System_Ext(push, "Push Notification Service", "e.g. Firebase Cloud Messaging")

  Rel(cameras, platform, "Streams video", "RTSP")
  Rel(admin, platform, "Configures cameras/zones/thresholds", "HTTPS")
  Rel(platform, ops, "Live annotated view + alerts", "HTTPS / WebSocket")
  Rel(platform, vms, "Forwards confirmed events", "Webhook / API")
  Rel(platform, push, "Sends mobile alerts", "HTTPS")
```

## 4. Container Architecture (single site)

Zooming into the platform: the actual running pieces and how they talk to each other. This is the architecture the other two docs describe in prose — here it's the map.

```mermaid
C4Container
  title Container Diagram — single site

  Person(ops, "Security Operator")

  System_Boundary(platform, "Mobius Tensors Platform") {
    Container(ingest, "Ingest Service", "Python + OpenCV", "Grabs and decodes camera frames")
    Container(infer, "Inference Workers", "Ultralytics YOLO + OpenVINO", "One per use case/model; runs detection")
    Container(logic, "Event Logic", "Python", "Rules, thresholds, debounce")
    ContainerDb(db, "Event Store", "PostgreSQL / SQLite", "Confirmed events, snapshot paths")
    Container(api, "Backend API", "FastAPI", "REST + WebSocket + MJPEG live stream")
    Container(web, "Web Dashboard", "React", "Live view, event history, snapshots")
    Container(mobile, "Mobile App", "iOS / Android", "Alerts, push notifications")
  }

  System_Ext(cam, "Cameras", "RTSP")

  Rel(cam, ingest, "Frames", "RTSP")
  Rel(ingest, infer, "Frames", "ZeroMQ PUB/SUB")
  Rel(infer, logic, "Detections", "ZeroMQ PUSH/PULL")
  Rel(logic, db, "Writes confirmed events", "SQL")
  Rel(logic, api, "Live event stream", "ZeroMQ PUB/SUB")
  Rel(infer, api, "Annotated frames", "ZeroMQ / local")
  Rel(db, api, "Reads history", "SQL")
  Rel(api, web, "REST / WebSocket / MJPEG", "HTTPS")
  Rel(api, mobile, "REST + Push", "HTTPS")
  Rel(ops, web, "Uses")
  Rel(ops, mobile, "Uses")
```

## 5. Component Architecture — Inference Worker

The one component most of the platform's value (and risk) sits in. Every use case gets one of these, differing only in which model and confidence threshold it loads.

```mermaid
flowchart TB
  Start([Worker starts]) --> Load[Load model once<br/>OpenVINO-optimized weights]
  Load --> Loop{Frame available?}
  Loop -- yes --> Grab[Grab next frame<br/>from ZeroMQ SUB]
  Grab --> Predict[Run model.predict]
  Predict --> Draw[Draw boxes + labels<br/>on frame]
  Draw --> PubDet[Publish detection JSON<br/>ZeroMQ PUSH]
  Draw --> PubFrame[Publish annotated frame<br/>for MJPEG stream]
  PubDet --> Loop
  PubFrame --> Loop
  Loop -- no / stream ended --> Stop([Worker exits / restarts])
```

## 6. Data Architecture

**Data flow** — what happens to a frame from capture to either storage or deletion:

```mermaid
flowchart LR
  F[Raw frame] --> D[YOLO detection]
  D --> R{Meets event rule?<br/>confidence, debounce, zone}
  R -- no --> X[Discarded<br/>never stored]
  R -- yes --> E[Event created]
  E --> Snap[Snapshot saved to disk/object storage]
  E --> Row[Row written to Event Store]
  Row --> Ret{Retention policy}
  Ret -- expired --> Del[Deleted]
  Ret -- within window --> Keep[Queryable via API]
```

**Entity model** — the event store's shape, extended with `SITE`/`CLIENT` for when the platform serves more than one location:

```mermaid
erDiagram
  CLIENT ||--o{ SITE : has
  SITE ||--o{ CAMERA : has
  CAMERA ||--o{ EVENT : produces
  USE_CASE ||--o{ EVENT : classifies
  EVENT ||--o| SNAPSHOT : has
  EVENT ||--o| ALERT : triggers
  CLIENT {
    string id
    string name
  }
  SITE {
    string id
    string client_id
    string address
  }
  CAMERA {
    string id
    string site_id
    string rtsp_url
  }
  USE_CASE {
    string id
    string name
    string model_version
  }
  EVENT {
    string id
    datetime ts
    float confidence
    string status
  }
  SNAPSHOT {
    string id
    string path
  }
  ALERT {
    string id
    string channel
    string status
  }
```

## 7. Security & Network Architecture

The rule that matters most here: **ZeroMQ never crosses a network boundary the public internet can reach.** Everything internal stays internal; the Backend API is the only door.

```mermaid
flowchart TB
  subgraph Internet["Public internet"]
    WebUser[Web browser]
    MobileUser[Mobile app]
  end

  subgraph DMZ["DMZ — internet-facing"]
    API[Backend API<br/>FastAPI: REST + WebSocket + MJPEG]
  end

  subgraph Internal["Internal server network — not internet-exposed"]
    Bus[(ZeroMQ bus)]
    Ingest[Ingest]
    Infer[Inference workers]
    Logic[Event logic]
    DB[(PostgreSQL)]
  end

  subgraph CamNet["Camera network — site-local, isolated"]
    Cams[Cameras / RTSP]
  end

  WebUser -- HTTPS/WSS --> API
  MobileUser -- HTTPS --> API
  API -- SQL --> DB
  API -. subscribes .-> Bus
  Ingest --> Bus --> Infer --> Bus --> Logic --> Bus
  Logic --> DB
  Cams -- RTSP, camera-net only --> Ingest
```

## 8. Deployment Architecture — Phase 0/1 (single site, draft/pilot)

Matches the roadmap's early phases: one server, CPU-only, everything on one box.

```mermaid
flowchart TB
  subgraph Server["Single Linux server (cloud VM or on-prem)"]
    direction TB
    Ingest2[Ingest process]
    Infer2[Inference workers<br/>CPU + OpenVINO]
    Logic2[Event logic]
    DB2[(SQLite / PostgreSQL)]
    API2[Backend API]
  end
  Cams2[1-2 cameras] --> Ingest2
  Ingest2 --> Infer2 --> Logic2 --> DB2
  Logic2 --> API2
  API2 --> Client2[Client browser]
```

## 9. Deployment Architecture — Phase 3/4 (multi-site, scaled)

Where the platform goes once a client (or several) needs more than one server can handle: still one platform, more instances of it, with a thin control plane to keep them manageable centrally.

```mermaid
flowchart TB
  subgraph Cloud["Central control plane (cloud)"]
    Portal[Admin portal / multi-tenant API]
    CentralDB[(Aggregated event store)]
  end

  subgraph SiteA["Client A — Site 1 (CPU server)"]
    A1[Ingest + Infer + Logic]
  end
  subgraph SiteB["Client A — Site 2 (GPU server)"]
    B1[Ingest + Infer + Logic<br/>GPU: high camera count]
  end
  subgraph SiteC["Client B — Site 1 (CPU server)"]
    C1[Ingest + Infer + Logic]
  end

  A1 -- sync events --> Portal
  B1 -- sync events --> Portal
  C1 -- sync events --> Portal
  Portal --> CentralDB
  Portal --> WebAdmin[Client web/mobile apps]
```

## 10. Scaling Decision: CPU+OpenVINO vs GPU+TensorRT

A concrete decision framework, not a philosophy — captures the reasoning already established for the project: default to CPU, move to GPU only when a benchmark proves you need to.

```mermaid
flowchart TD
  Start2([New site / new camera count]) --> Bench[Benchmark on CPU + OpenVINO]
  Bench --> Q1{Meets required FPS/latency<br/>at target camera count?}
  Q1 -- yes --> StayCPU[Stay on CPU server<br/>lowest cost of goods]
  Q1 -- no --> Q2{Root cause: decode bottleneck<br/>or model compute?}
  Q2 -- decode --> FixDecode[Thread/parallelize decode first<br/>re-benchmark]
  FixDecode --> Bench
  Q2 -- model compute --> GPU[Move to GPU server<br/>TensorRT export]
  GPU --> Q3{Multiple concurrent models<br/>per camera needed?}
  Q3 -- yes --> GPU
  Q3 -- no --> StayCPU
```

## 11. Client Lifecycle Architecture

How a client moves through the system over time — ties the business layer back to the deployment layer.

```mermaid
flowchart LR
  Lead[Client conversation] --> Scope[Use-case scoping]
  Scope --> Survey[Camera/site survey]
  Survey --> Deploy[Deploy Phase 0/1 pipeline<br/>on client's server or ours]
  Deploy --> Pilot[Pilot: collect accuracy +<br/>false-positive data]
  Pilot --> Decision{Meets acceptance<br/>threshold?}
  Decision -- yes --> Prod[Move to production<br/>Phase 3 hardening]
  Decision -- no --> Tune[On-site fine-tuning /<br/>threshold adjustment]
  Tune --> Pilot
  Prod --> Scale[Add sites / use cases<br/>Phase 4]
```

## 12. Full System Synthesis

Everything above, in one picture — business rationale at the top, running code at the bottom.

```mermaid
flowchart TB
  Biz[["One platform, many clients<br/>(Business Layer)"]]
  UseCase[["Use case = model + ruleset<br/>(Product Layer)"]]
  Cam[Cameras / RTSP] --> Ingest3[Ingest]
  Ingest3 -- ZeroMQ --> Infer3[YOLO + OpenVINO<br/>inference workers]
  Infer3 -- ZeroMQ --> Logic3[Event logic]
  Logic3 --> DB3[(Event store)]
  Logic3 -- ZeroMQ --> API3[Backend API<br/>REST + WebSocket + MJPEG]
  DB3 --> API3
  API3 --> Web3[Web dashboard]
  API3 --> Mobile3[Mobile app]

  Biz -.-> UseCase -.-> Ingest3
```

## Related docs

- Use-case details, roadmap, and open risks → [Mobius Tensors — AI Solution Platform.md](Mobius%20Tensors%20—%20AI%20Solution%20Platform.md)
- Why each piece exists (plain-language build order + glossary) → [Implementation Plan — From Trained Model to App.md](Implementation%20Plan%20—%20From%20Trained%20Model%20to%20App.md)
- First running piece of this architecture → [`server/detect.py`](server/detect.py)
