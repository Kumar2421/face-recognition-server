# Face Service Scalability Proposal (No Triton)

## Goals

- Reduce end-to-end latency (especially p95/p99) for recognition and enrollment.
- Increase throughput on a single GPU without adding Triton.
- Preserve existing quality assessment semantics, but reorder it to be lightweight.
- Persist all *quality-passed* images and embeddings in a date-ordered way.
- Enable search in a user-provided date range (scalable via Qdrant payload filtering).

This document is grounded in the current codebase behavior:

- Inference uses **InsightFace Buffalo-L** via `insightface.app.FaceAnalysis` (see `embedders/buffalo_l.py`).
- The recognition event endpoint is `POST /v1/events/recognition` (see `_process_recognition_image` in `app.py`).
- Quality checks are implemented by `FaceQualityEvaluator` (see `quality/evaluator.py`).
- Similarity search is backed by Qdrant when enabled (`QDRANT_URL`).

## Current bottlenecks (confirmed)

### 1) Detection fallback worst-case explosion

`BuffaloLEmbedder.detect_best()` retries detection across multiple scales and rotations when no face is found. In worst case it can call `FaceAnalysis.get()` up to 32 times per image. Under load, this is the primary p99 killer.

**Decision**: disable this completely for your traffic (already-cropped faces):

- `BUFFALO_ENABLE_FALLBACK_VARIANTS=0`

### 2) GPU contention from unconstrained concurrent requests

FastAPI endpoints are `async`, but the heavy inference call is CPU/GPU bound and currently executed inline. Under load multiple requests can attempt GPU inference concurrently, causing device contention and unstable latency.

**Decision**: enforce a single GPU execution lane:

- `GPU_MAX_INFLIGHT=1`

### 3) Timing instrumentation mixes CPU, queueing, GPU async effects

Current timings are wall-clock (`time.time()`), and don’t separate:

- queue wait
- preprocess time
- model call time
- Qdrant time

We will introduce consistent breakdown timings using `time.perf_counter()`.

---

## Phase P0 (Immediate) — Reduce p99 and stabilize throughput

### A) Disable fallback variants in `detect_best`

Change `embedders/buffalo_l.py` to guard the fallback branch behind an environment flag:

- When `BUFFALO_ENABLE_FALLBACK_VARIANTS=0`, if `self.app.get(bgr)` returns no faces, raise `ValueError("no face detected")` immediately.

This alone typically yields the biggest tail-latency improvement.

### B) Lightweight quality gating (keep same evaluator)

Keep `FaceQualityEvaluator` logic, but execute it in a cheaper order:

1) **Frame-only checks (no face object required)**
   - `min_resolution`
   - `blur`
   - `brightness`

2) Only if frame-only passes:
   - run detection

3) **Face-dependent checks (requires bbox/pose/confidence)**
   - `face_ratio` (requires bbox)
   - `landmark_score` threshold
   - pose thresholds (yaw/pitch if available)

This reduces GPU work on clearly bad frames while preserving the same final accept/reject semantics.

### C) GPU Inference Manager (single-GPU scheduler)

Introduce a centralized inference manager responsible for all calls into `FaceAnalysis.get()`.

#### Design

- A background worker consumes an `asyncio.Queue` of inference jobs.
- Each job carries:
  - the input `bgr` image
  - the desired mode (`detect_best` / `detect_all`)
  - a `Future` to resolve
  - enqueue timestamp for queue-latency metrics
- Concurrency control:
  - `GPU_MAX_INFLIGHT=1` means exactly one active job at a time.

#### Worker execution model

Use `asyncio.to_thread(...)` (or a dedicated threadpool) inside the worker so the event loop stays responsive.

#### Metrics to add

Emit breakdown timings per request:

- `decode_ms`
- `quality_frame_ms`
- `queue_ms`
- `detect_embed_ms`
- `qdrant_ms`
- `total_ms`

Also track:

- `gpu_queue_depth`
- `gpu_jobs_total`
- `gpu_job_fail_total`

#### Why this increases throughput

Even without true tensor batching, this avoids GPU thrash and keeps a consistent GPU execution pipeline, which usually improves throughput under bursty load and reduces p95/p99.

---

## Phase P1 — Date-ordered embedding store (quality-passed images)

You asked for:

- “store all the quality passed images as an embedding on date order”
- search by date range
- scalable storage using Qdrant

This proposal uses:

- **Qdrant** as the embedding index (vector + payload)
- **filesystem** (or object storage later) for image persistence
- **a deterministic unique ID** for each stored image+embedding

### A) What to store

For every image that passes the quality gate (whether from recognition ingestion or enrollment), persist:

- **Image** (original BGR re-encoded to JPG) saved under a date partition
- **Embedding** vector stored in Qdrant
- **Payload fields** in Qdrant for filtering and traceability

Recommended payload schema (compatible with your current `API.md` model):

- `image_id`: string (UUID)
- `created_at`: ISO8601 string
- `created_ts`: float (unix seconds) OR int (unix ms)
- `camera`: string (optional)
- `subject_id`: string (optional; enrollment sets it, ingestion may not)
- `source`: one of `enroll | ingested | external`
- `filename`: string (optional)
- `image_path`: string (where the persisted JPG lives)
- `thumb_path`: string (optional)
- `quality`: object (optional summary fields only; avoid huge payload)

**Important**: for scalable date filtering, store `created_ts` as a numeric field and build Qdrant payload index on it.

### B) Unique ID strategy

Two safe choices:

1) **Deterministic ID (idempotent)**
   - `image_hash = sha256(original_bytes)`
   - `image_id = uuid5(NAMESPACE, source + camera + image_hash)`
   - Pros: retries won’t duplicate data
   - Cons: if the same frame is slightly recompressed, you get a new ID

2) **Random UUID**
   - `image_id = uuid4()`
   - Pros: simple
   - Cons: duplicates possible if clients retry

Recommendation: deterministic for enroll, and either deterministic or uuid4 for ingested events depending on your upstream retry behavior.

### C) Date-ordered filesystem layout

Store images under:

- `${EMBED_STORE_DIR}/images/YYYY/MM/DD/{image_id}.jpg`

Optionally also store small JSON sidecar:

- `${EMBED_STORE_DIR}/meta/YYYY/MM/DD/{image_id}.json`

This keeps directory size manageable and enables fast listing by date.

### D) Qdrant collection design

Use one collection (example): `frigate_faces_v2` or reuse existing.

Vectors:

- single vector per image

Payload indexed fields:

- `created_ts` (range filter)
- `camera` (optional filter)
- `subject_id` (optional filter)
- `source` (optional filter)

### E) Date range search behavior

Add (or extend) a search endpoint to accept:

- `since_ts` / `until_ts` (unix seconds) OR `since_iso` / `until_iso`
- optional `camera`
- optional `subject_id`

Qdrant query:

- vector search top-K
- filter:
  - `created_ts >= since_ts`
  - `created_ts <= until_ts`
  - plus optional `camera == ...`

This returns:

- `image_id`
- similarity
- payload (including `image_path` / `thumb_path`)

### F) Where to hook persistence in current pipeline

- Enrollment (`/v1/faces/add` and `/v1/faces/add_upload`):
  - if quality OK → upsert vector + persist image

- Recognition ingestion (`/v1/events/recognition`):
  - for every face that passes quality gating and produced an embedding → upsert vector + persist image
  - store `camera`, `source_path`, and `ts_val` as `created_ts`

Note: recognition ingestion currently saves accepted/rejected/no_match images under `EVENTS_DIR`. This embedding store is distinct and should be controlled via its own root dir, e.g. `EMBED_STORE_DIR=/data/embed_store`.

---

## Phase P2 — Operational safeguards

### A) Backpressure and load shedding

When GPU queue is full:

- return 429 (`Too Many Requests`) with `retry_after_ms`

Config knobs:

- `GPU_QUEUE_MAX=256` (example)
- `GPU_QUEUE_TIMEOUT_MS=1000` (example)

### B) Observability

- Keep JSON logs, but add consistent event types:
  - `inference_enqueued`
  - `inference_started`
  - `inference_finished`
  - `embedding_upserted`

- Expose Prometheus metrics for:
  - queue depth
  - job latencies
  - Qdrant latencies

---

## Configuration summary (recommended defaults)

- `GPU_MAX_INFLIGHT=1`
- `BUFFALO_ENABLE_FALLBACK_VARIANTS=0`
- `GPU_BATCH_WINDOW_MS=0` (initially; can add later)
- `GPU_QUEUE_MAX=256` (example)
- `EMBED_STORE_DIR=/data/embed_store`
- `QDRANT_COLLECTION=frigate_faces_v2` (optional new collection)

---

## Acceptance criteria

### Performance

- p99 latency decreases significantly vs current worst-case behavior.
- Under burst, service remains stable (no GPU thrash), with predictable queueing.

### Data

- Every quality-passed image produces:
  - a persisted JPG with a stable `image_id`
  - a Qdrant point with payload containing `created_ts`

### Search

- Search endpoint can filter by date range using Qdrant payload filter on `created_ts`.

