# Face Service — API Usage Guide

A practical, task-oriented guide to using the Face Recognition Server endpoints.
For the exhaustive per-endpoint reference (every field, every response shape) see
[`API.md`](../API.md) at the repo root. This document focuses on **how to actually
use the API** for the common jobs: enrolling people, recognizing them, browsing
data, and auditing events.

The service is built on **InsightFace (Buffalo-L)** for embeddings and **Qdrant**
for vector search. One embedding vector is stored per image.

---

## 1. Basics

### Base URL

| Environment | URL |
|-------------|-----|
| Local (Docker Compose) | `http://localhost:8001` |
| Production | `https://api.face.service.tools.thefusionapps.com` |

All paths also accept an optional `/api` prefix, so `/v1/faces/add` and
`/api/v1/faces/add` are equivalent.

### Authentication

Protected endpoints require an `x-api-key` header:

```bash
-H "x-api-key: YOUR_API_KEY"
```

The header is matched against `FACE_SERVICE_API_KEY` (standard data) or
`FACE_SERVICE_LEGACY_API_KEY` (an isolated "legacy" data partition). Read-only
endpoints accept the key optionally — omitting it defaults to the **standard**
partition.

> **Data segregation:** subjects, events, search history, groups and branches
> enrolled with the legacy key are invisible to standard users, and vice-versa.

### Image input

Almost every image-accepting endpoint takes one of:

- a **Base64-encoded** image string, or
- a **direct HTTP/HTTPS URL** (the server downloads it), or
- a **multipart file upload** (the `*_upload` variants).

Field names are flexible: `image_b64`, `images_b64`, and `image` are
interchangeable on most JSON endpoints.

### Quick health check

```bash
curl -s http://localhost:8001/health | python3 -m json.tool
```

Returns `ok`, subject/group counts, Qdrant status, GPU queue depth, and process
memory. Works with both `GET` and `POST`.

---

## 2. Common workflows

### Workflow A — Enroll a person, then recognize them

**Step 1: Enroll one or more images for a subject.**

```bash
curl -s -X POST "http://localhost:8001/v1/faces/add_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "subject_id=alice" \
  -F "files=@/path/to/alice1.jpg;type=image/jpeg" \
  -F "files=@/path/to/alice2.jpg;type=image/jpeg"
```

Or via JSON with URLs / base64:

```bash
curl -s -X POST "http://localhost:8001/v1/faces/add" \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"subject_id":"alice","image_urls":["https://example.com/alice.jpg"]}'
```

Response tells you how many images were embedded:

```json
{ "subject_id": "alice", "num_images": 2, "num_embedded": 2, "embedding_dim": 512, "meta": {} }
```

> If `num_embedded` is lower than `num_images`, some images failed the quality
> gate or the duplicate check. See [Troubleshooting](#5-troubleshooting).

**Step 2: Recognize a probe image** (best match + threshold + decision metadata).

```bash
curl -s -X POST "http://localhost:8001/v1/faces/recognize" \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","top_k":5,"min_similarity":0.75}'
```

```json
{ "matched": true, "subject_id": "alice", "similarity": 0.83, "results": [ ... ], "meta": { ... } }
```

Use `/v1/faces/recognize_upload` for a multipart file instead of base64.

### Workflow B — One-shot similarity search (no threshold)

When you just want the top-K nearest subjects, ranked, without a match/no-match
decision:

```bash
curl -s -X POST "http://localhost:8001/v1/faces/search" \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","top_k":5}'
```

Returns a `results` array sorted by `similarity`, plus a `query_thumb_path`.

### Workflow C — Compare two faces directly (1:1)

No enrollment needed — just score how similar two images are:

```bash
curl -s -X POST "http://localhost:8001/v1/face/compare_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file1=@/path/to/face1.jpg" \
  -F "file2=@/path/to/face2.jpg"
```

```json
{ "similarity": 0.92, "match": true, "confidence": "High", "meta": { ... } }
```

JSON equivalent (`/v1/face/compare`) accepts `image1_b64`/`image1_url` and
`image2_b64`/`image2_url`.

### Workflow D — Frigate / camera integration

Frigate calls `POST /v1/face/search` — a top-1 search that returns a single best
subject or `404` when below `FACE_SERVICE_MIN_SIMILARITY`:

```bash
curl -s -X POST "http://localhost:8001/v1/face/search" \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","camera":"front_door"}'
```

To record an attempt for later audit (saves image + thumbnail + DB row), use the
recognition events ingestion endpoint instead:

```bash
curl -s -X POST "http://localhost:8001/v1/events/recognition" \
  -F "camera=front_door" -F "top_k=5" \
  -F "file=@/path/to/frame.jpg;type=image/jpeg"
```

### Workflow E — Check image eligibility before enrolling

Validate quality (blur, pose, brightness, face size) without storing anything:

```bash
curl -s -X POST "http://localhost:8001/v1/quality/check_upload" \
  -F "file=@/path/to/image.jpg"
```

`total_quality` is `pass` / `fail`; `faces[]` holds per-face metrics and
`annotated_image` is a base64 JPEG with bounding boxes drawn.

### Workflow F — Privacy-safe multi-face extraction

For a frame with several people, get one crop per face with **all other faces
blurred**, optionally recognizing each:

```bash
curl -s -X POST "http://localhost:8001/v1/faces/privacy_extract" \
  -H "x-api-key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","recognition":true,"top_k":1}'
```

### Workflow G — Browse and manage enrolled subjects

```bash
# List subjects (with embedding counts)
curl -s "http://localhost:8001/v1/subjects?with_counts=true&limit=50"

# One subject's detail + its images
curl -s "http://localhost:8001/v1/subjects/alice"
curl -s "http://localhost:8001/v1/subjects/alice/images?limit=50"

# Find possible duplicate enrollments (subjects that look like 'alice')
curl -s "http://localhost:8001/v1/faces/cross_match/alice?top_k=10"

# Delete every vector for a subject
curl -s -X DELETE "http://localhost:8001/v1/faces/subjects/alice" \
  -H "x-api-key: YOUR_API_KEY"
```

### Workflow H — Audit recognition events & give feedback

```bash
# List events for a camera
curl -s "http://localhost:8001/v1/events/recognition?camera=front_door&limit=10"

# Fetch one event
curl -s "http://localhost:8001/v1/events/recognition/EVENT_ID"

# Label an event for accuracy tracking (tp | fp | fn | ignore)
curl -s -X POST "http://localhost:8001/v1/events/recognition/EVENT_ID/feedback" \
  -H "Content-Type: application/json" \
  -d '{"label":"tp","note":"Confirmed correct match"}'

# Aggregate stats / feedback accuracy
curl -s "http://localhost:8001/v1/events/recognition/stats?day=2026-05-14"
curl -s "http://localhost:8001/v1/events/recognition/feedback_stats"
```

### Workflow I — Visitors vs. employees cross-check

Scan stored **match** events for visitors (subject id prefixed `visitor-`/
`visiter-`) whose face is actually similar to an enrolled **employee** (prefixed
`employee-`). Useful for flagging employees who walked in as visitors. Defaults
to **today** if no date window is given.

```bash
curl -s "http://localhost:8001/v1/cross_check/visitors_vs_employees?day=2026-05-30&camera=lobby" \
  -H "x-api-key: YOUR_API_KEY"
```

```json
{
  "items": [
    {
      "employee_subject_id": "employee-emerald-e00643",
      "visitor_event_id": "42c72ad0-...",
      "visitor_subject_id": "visitor-1234",
      "similarity": 0.88,
      "top2_second": 0.61,
      "top2_margin": 0.27,
      "visitor_ts": 1778759909.49,
      "visitor_camera": "lobby",
      "visitor_image_path": "/events/accepted/lobby/....jpg",
      "visitor_thumb_path": "/thumbs/evt-....jpg"
    }
  ]
}
```

Query params: `camera`, `day`, `from_day`, `to_day`, `since_ts`, `until_ts`,
`limit` (default 500, capped by `CROSSCHECK_MAX_EVENTS`).

---

## 3. Date / time filtering

Many list and stats endpoints (recognition events, search history, cross-check,
recognize) accept a flexible date window. Precedence is:

1. `since_ts` / `until_ts` (float epoch seconds) — used if either is set.
2. `day` (e.g. `2026-05-30`) — a single calendar day.
3. `from_day` / `to_day` — an inclusive day range.

Days are resolved in the `FACE_SERVICE_TIMEZONE` timezone (default
`Asia/Kolkata`).

---

## 4. Endpoint index

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET/POST | `/health` | none | Service health, counts, GPU + memory |
| GET | `/v1/stats` | optional | Global counters |
| GET | `/metrics` | none | Prometheus metrics |
| GET | `/debug/providers` | none | ONNXRuntime / InsightFace providers |
| GET | `/ui` | none | Built-in debug web UI |
| POST | `/v1/faces/add` | key | Enroll (JSON: base64/URL) |
| POST | `/v1/faces/add_upload` | key | Enroll (multipart files/URLs) |
| POST | `/v1/faces/search` | key | Top-K similarity search (JSON) |
| POST | `/v1/faces/search_upload` | key | Top-K search (multipart) |
| POST | `/v1/faces/recognize` | key | Best match + threshold (JSON) |
| POST | `/v1/faces/recognize_upload` | key | Best match + threshold (multipart) |
| POST | `/v1/face/search` | key | Frigate top-1 search |
| POST | `/v1/face/search_upload` | key | Frigate top-1 (multipart) |
| POST | `/v1/face/compare` | key | 1:1 compare (JSON) |
| POST | `/v1/face/compare_upload` | key | 1:1 compare (multipart) |
| POST | `/v1/faces/privacy_extract` | key | Per-face crops with others blurred |
| POST | `/v1/quality/check_upload` | optional | Quality eligibility check |
| GET | `/v1/faces/cross_match/{subject_id}` | optional | Similar subjects (dup detection) |
| GET | `/v1/faces/subjects` | optional | Unique subject ids |
| DELETE | `/v1/faces/subjects/{subject_id}` | key | Delete a subject |
| GET | `/v1/subjects` | optional | Browse subjects (paginated) |
| GET | `/v1/subjects/{subject_id}` | optional | One subject's detail |
| GET | `/v1/subjects/{subject_id}/images` | optional | Subject's images |
| POST | `/v1/groups` | key | Create group |
| GET | `/v1/groups` | key | List groups |
| DELETE | `/v1/groups/{group_id}` | key | Delete group |
| POST | `/v1/branches` | key | Create branch |
| GET | `/v1/branches` | key | List branches |
| DELETE | `/v1/branches/{branch_id}` | key | Delete branch |
| POST | `/v1/events/recognition` | optional | Ingest recognition event |
| GET | `/v1/events/recognition` | optional | List events (filterable) |
| GET | `/v1/events/recognition/{event_id}` | optional | One event |
| POST | `/v1/events/recognition/{event_id}/feedback` | optional | Submit feedback |
| POST | `/v1/events/recognition/forward` | optional | Forward event to webhook |
| GET | `/v1/events/recognition/cameras` | optional | Unique camera names |
| GET | `/v1/events/recognition/stats` | optional | Aggregated event stats |
| GET | `/v1/events/recognition/feedback_stats` | optional | Feedback accuracy stats |
| GET | `/v1/search_history` | optional | List search events |
| GET | `/v1/search_history/stats` | optional | Search match/no-match stats |
| GET | `/v1/search_history/asset/image/{event_id}` | optional | Query image JPEG |
| GET | `/v1/search_history/asset/thumb/{event_id}` | optional | Query thumbnail JPEG |
| GET | `/v1/cross_check/visitors_vs_employees` | optional | Visitor↔employee cross-check |

Static asset mounts: `/thumbs/...`, `/images/...`, `/events/...`.

---

## 5. Troubleshooting

**`no face detected`** — use a clearer, front-facing image; lower
`BUFFALO_MIN_DET_SCORE`; ensure the image isn't too dark or blurry.

**`no faces embedded from provided images` (but quality passes)** — the
enrollment duplicate check likely blocked it because the face already exists
under another `subject_id`. Tune `ENROLL_DUPLICATE_CHECK_ENABLE` /
`ENROLL_DUPLICATE_MIN_SIM`.

**Recognition returns 404** — best similarity is below
`FACE_SERVICE_MIN_SIMILARITY` (or the per-request `min_similarity`). Lower the
threshold or enroll more images of the subject.

**Slow inference** — check `GET /debug/providers`; if only
`CPUExecutionProvider` shows under `insightface.session_providers`, the GPU isn't
being used.

**Qdrant errors** — ensure the `qdrant` service is up
(`docker compose up -d qdrant`) and `QDRANT_URL` is reachable from the service.

---

## 6. Related docs

- [`API.md`](../API.md) — full per-endpoint reference with every field and config var.
- [`docs/UI_GUIDE.md`](UI_GUIDE.md) — using the bundled web UI.
- [`docs/REPO_REVIEW.md`](REPO_REVIEW.md) — repository overview.
