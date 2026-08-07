# Face Service API (Swagger-style)

This service provides face embedding, enrollment, and similarity search using **InsightFace (Buffalo-L)** + **Qdrant**. It stores one vector per image and generates 256px thumbnails for gallery views.

## Interactive docs

The built-in web UI is:

- `https://api.face.service.tools.thefusionapps.com`

## Configuration

Environment variables (Docker Compose sets most of these):

- `QDRANT_URL` (example: `http://qdrant:6333`)
- `QDRANT_COLLECTION` (default: `frigate_faces`)
- `FACE_SERVICE_MIN_SIMILARITY` (default: `0.15`) used as recognition threshold
- `FACE_SERVICE_DEBUG` (`1` enables logs)
- `BUFFALO_MODEL_ROOT` (default: `/models`)
- `BUFFALO_MODEL_NAME` (default: `buffalo_l`)
- `BUFFALO_DET_SIZE` (default: `640`)
- `BUFFALO_MIN_DET_SCORE` (default: `0.5`) minimum detector score for accepting a face
- `BUFFALO_PROVIDERS` (example: `CUDAExecutionProvider,CPUExecutionProvider`)

- `FACE_SERVICE_API_KEY` (default: `your-secret-key`) security key for the `standard` data bucket
- `FACE_SERVICE_LEGACY_API_KEY` (optional) **master key**. Owns the `legacy` data bucket AND is the only key allowed to manage tenant keys via `/v1/keys` (create / list / delete). See **Key management**.
- `KEYS_COLLECTION` (default: `api_keys`) Qdrant collection that stores tenant keys.
- `KEY_RELOAD_SEC` (default: `15`) interval for auto-reloading the key registry from Qdrant (also force-reloaded on create/delete).

GPU / performance:

- `GPU_INFERENCE_MANAGER` (`1` enables serialized GPU execution, `0` disables)
- `GPU_QUEUE_MAX` (default: `2048`) max queue size for GPU inference manager
- `GPU_BATCH_WINDOW_MS` (default: `10`) micro-batching window
- `GPU_NUM_WORKERS` (default: `4`) number of concurrent GPU inference workers
- `BUFFALO_ENABLE_FALLBACK_VARIANTS` (`0` disables expensive rotation/scale fallback on no-face)

Storage:

- `EVENTS_DIR` (default: `/data/events`) where event images are stored (`accepted/`, `rejected/`, `no_match/`)
- `THUMBS_DIR` (default: `/data/thumbs`) thumbnails directory
- `IMAGES_DIR` (default: `/data/images`) optional image storage directory

Notes:

- Endpoints marked with **Auth: API key** require an `x-api-key` header.
- **Data Segregation (Isolation)** — every key maps to its own data bucket (`access_key`). The resolver order:
  - empty key → `standard` (on optional/read endpoints) or `403` (on protected ones).
  - matches `FACE_SERVICE_API_KEY` → `standard` bucket.
  - matches `FACE_SERVICE_LEGACY_API_KEY` (master) → `legacy` bucket.
  - matches a **tenant key** created via `/v1/keys` → that key's assigned bucket (`t_…`).
  - any other non-empty key → its own deterministic ad-hoc bucket (`k_…`).
  - Isolation applies to Subjects, Recognition Events, Search History, Groups, and Branches. Data enrolled under one key is invisible to all other keys. A single key scopes the **entire** dashboard.
- **Flexible Routing**: The API supports an optional `/api` prefix. Both `/v1/...` and `/api/v1/...` are valid.
- **Image Input Flexibility**: 
  - Most endpoints accept `image_b64`, `images_b64`, or `image` as field names.
  - You can provide either a **Base64 encoded string** or a **direct HTTP/HTTPS URL**. If a URL is provided, the server will automatically download and process the image.
- Qdrant point IDs must be **UUID** or **integer**. This service uses deterministic UUIDs derived from `subject_id` and `image_id`.
- Detection robustness:
  - If detection fails, the service retries with rotated and downscaled variants (when `BUFFALO_ENABLE_FALLBACK_VARIANTS=1`).
- Thumbnails are served at `/thumbs/{image_id}.jpg` (root configurable via `THUMBS_DIR`).
- Original images are served at `/images/{subject_id}/{image_id}.jpg` (root configurable via `IMAGES_DIR`).
- **CORS**: The service allows standard CORS headers including `Cache-Control` for frontend compatibility.

here is the reference for the api: fs_9f2b8a71c4d04e5e9b3d8a7c6b5a4f3e

## Data model

### Face embedding

- A single image produces one embedding vector (Buffalo-L, typically 512-D)
- The service stores one vector per uploaded/enrolled image in Qdrant
- Each vector has payload:
  - `subject_id` (string)
  - `image_id` (string, UUID per image)
  - `created_at` (ISO8601 string)
  - `thumb_path` (string, e.g. `/thumbs/{image_id}.jpg`)
  - `image_path` (string, e.g. `/images/{subject_id}/{image_id}.jpg`)
  - `source` (one of: `enroll`, `external`, `ingested`, `auto_recognized`)
  - `branch` (string, optional)
  - optional `filename` (string) for upload endpoints

## Endpoints

---

### Health

#### `GET /health` or `POST /health`

Returns service health and Qdrant status. Supports both methods for compatibility with various monitoring tools.

**Auth:** None

Response (example):
```json
{
  "ok": true,
  "subjects": 1,
  "groups": 2,
  "qdrant_enabled": true,
  "qdrant_collection": "frigate_faces",
  "gpu_inference": {
    "queue_size": 0,
    "max_queue": 2048,
    "workers": 4,
    "batch_window_s": 0.01
  },
  "system": {
    "memory_rss_mb": 450.5,
    "memory_vms_mb": 1200.2,
    "cpu_percent": 12.5,
    "threads": 15,
    "gc_objects": 45000
  }
}
```

Curl:
```bash
curl -s http://localhost:8001/health | python3 -m json.tool
```

---

## Stats

### `GET /v1/stats`

Returns global counters and Qdrant status.

**Auth:** Optional (x-api-key for segregation)

Response (example):
```json
{
  "subjects_total": 145,
  "embeddings_total": 5621,
  "last_24h_enrolls": 23,
  "last_24h_searches": 410,
  "qdrant_enabled": true,
  "qdrant_collection": "frigate_faces"
}
```

Curl:
```bash
curl -s http://localhost:8001/v1/stats | python3 -m json.tool
```

---

## Observability / Debug

### `GET /metrics`

Prometheus metrics in text format.

**Auth:** None

Curl:
```bash
curl -s http://localhost:8001/metrics
```

### `GET /robots.txt`

Static robots policy (disallows all).

**Auth:** None

### `GET /debug/providers`

Shows ONNXRuntime providers and what providers InsightFace sessions were created with.

**Auth:** None

Response (example):
```json
{
  "onnxruntime": {
    "version": "1.17.1",
    "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "embedder": {
    "class": "BuffaloLEmbedder",
    "configured_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "insightface": {
    "models": ["landmark_3d_68", "landmark_2d_106", "detection", "genderage", "recognition"],
    "session_providers": {
      "detection": ["CUDAExecutionProvider", "CPUExecutionProvider"],
      "recognition": ["CUDAExecutionProvider", "CPUExecutionProvider"]
    }
  }
}
```

Curl:
```bash
curl -s http://localhost:8001/debug/providers | python3 -m json.tool
```

### `GET /ui`

Serves the built-in HTML debug UI page.

**Auth:** None

---

## Group management

### `POST /v1/groups`

Create a new group.

**Auth:** API key

Request body:
```json
{
  "group_id": "employees",
  "name": "Employee Group",
  "meta": {}
}
```

Response:
```json
{
  "group_id": "employees",
  "name": "Employee Group",
  "meta": {}
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/groups" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"group_id":"employees","name":"Employee Group","meta":{}}'
```

### `GET /v1/groups`

List all groups.

**Auth:** API key

Response:
```json
{
  "groups": [
    {"group_id": "employees", "name": "Employee Group", "meta": {}}
  ]
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/groups" \
  -H "x-api-key: YOUR_API_KEY"
```

### `DELETE /v1/groups/{group_id}`

Delete a group.

**Auth:** API key

Response:
```json
{
  "deleted": true,
  "group_id": "employees"
}
```

Curl:
```bash
curl -s -X DELETE "http://localhost:8001/v1/groups/employees" \
  -H "x-api-key: YOUR_API_KEY"
```

---

## Branch management

### `POST /v1/branches`

Create a new branch.

**Auth:** API key

Request body:
```json
{
  "branch_id": "branch-001",
  "name": "Main Branch",
  "meta": {}
}
```

Response:
```json
{
  "branch_id": "branch-001",
  "name": "Main Branch",
  "meta": {}
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/branches" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"branch_id":"branch-001","name":"Main Branch","meta":{}}'
```

### `GET /v1/branches`

List all branches.

**Auth:** API key

Response:
```json
{
  "branches": [
    {"branch_id": "branch-001", "name": "Main Branch", "meta": {}}
  ]
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/branches" \
  -H "x-api-key: YOUR_API_KEY"
```

### `DELETE /v1/branches/{branch_id}`

Delete a branch.

**Auth:** API key

Response:
```json
{
  "deleted": true,
  "branch_id": "branch-001"
}
```

Curl:
```bash
curl -s -X DELETE "http://localhost:8001/v1/branches/branch-001" \
  -H "x-api-key: YOUR_API_KEY"
```

---

## Key management

Manage tenant API keys. Each tenant key isolates its own data bucket (`access_key`).

**Auth:** Master key only — `x-api-key` must equal `FACE_SERVICE_LEGACY_API_KEY`. Any other key (or none) → `403`. If the master key is not configured → `503`.

Keys are stored in the Qdrant `api_keys` collection and auto-reloaded into memory (every `KEY_RELOAD_SEC`, plus an immediate reload on create/delete) so new keys take effect **without a restart**.

### `POST /v1/keys`

Create a tenant key. The raw key is returned **once** here; later listings only show a masked value.

**Auth:** Master key

Request body:
```json
{
  "name": "TMJ branch dashboard",
  "api_key": "fs_optional_custom_key"
}
```

- `name` (string, optional) human label.
- `api_key` (string, optional; alias `key`) supply your own key value. If omitted, the server generates one (`fs_<random>`).

Response:
```json
{
  "key_id": "8f1c1e2a-....",
  "name": "TMJ branch dashboard",
  "access_key": "t_4b8c1d2e3f4a5b6c",
  "created_at": "2026-06-01T10:00:00+00:00",
  "active": true,
  "api_key": "fs_9a8b7c6d5e4f...   (RAW — shown only once)",
  "api_key_masked": "fs_9a8…f3e2"
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/keys" \
  -H "x-api-key: MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"TMJ branch dashboard"}'
```

### `GET /v1/keys`

List all tenant keys (masked).

**Auth:** Master key

Response:
```json
{
  "keys": [
    {
      "key_id": "8f1c1e2a-....",
      "name": "TMJ branch dashboard",
      "access_key": "t_4b8c1d2e3f4a5b6c",
      "created_at": "2026-06-01T10:00:00+00:00",
      "active": true,
      "api_key_masked": "fs_9a8…f3e2"
    }
  ]
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/keys" \
  -H "x-api-key: MASTER_KEY"
```

### `DELETE /v1/keys/{key_id}`

Revoke a tenant key. After deletion the key no longer resolves to its bucket, so that tenant's data becomes inaccessible (hidden, not erased).

**Auth:** Master key

Response:
```json
{
  "key_id": "8f1c1e2a-....",
  "deleted": true
}
```

Curl:
```bash
curl -s -X DELETE "http://localhost:8001/v1/keys/8f1c1e2a-...." \
  -H "x-api-key: MASTER_KEY"
```

Notes:
- After creating a key, use it as `x-api-key` on any normal endpoint (enroll, search, recognize, events, etc.) — all data is scoped to that key's bucket automatically.
- A freshly created key starts with an **empty** dataset until you enroll/ingest under it.
- Raw keys are stored in the Qdrant payload to allow header matching.

---

## Enrollment (Add)

You can enroll faces using JSON (base64/URL), multipart upload, or image URLs.

### `POST /v1/faces/add` (JSON)

**Auth:** API key

Request body:
```json
{
  "subject_id": "alice",
  "images_b64": ["<base64-encoded-image>"],
  "image_urls": ["https://example.com/photo.jpg"],
  "branch": "branch-001",
  "created_at": "2024-05-20T10:00:00Z",
  "ts": 1716199200
}
```

- `subject_id` (string, required)
- `images_b64` (array of strings, optional) base64-encoded images or HTTP/HTTPS URLs
- `image_urls` (array of strings, optional) HTTP/HTTPS image URLs
- `branch` (string, optional)
- `created_at` (string, optional) ISO8601 string
- `ts` (float, optional) Unix timestamp

At least one of `images_b64` or `image_urls` must be provided.

Response:
```json
{
  "subject_id": "alice",
  "num_images": 2,
  "num_embedded": 2,
  "embedding_dim": 512,
  "meta": {}
}
```

Curl (base64):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"alice","images_b64":["...base64..."]}'
```

Curl (URL):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"alice","image_urls":["https://example.com/photo.jpg"]}'
```

### `POST /v1/faces/add_upload` (multipart)

**Auth:** API key

Form fields:

- `subject_id` (text, required)
- `branch` (text, optional)
- `created_at` (text, optional) ISO8601 string
- `ts` (float, optional) Unix timestamp
- `files` (one or more images, optional)
- `image_urls` (one or more URLs, optional)

At least one of `files` or `image_urls` must be provided.

Response: Same as `/v1/faces/add`.

Curl (file upload):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "subject_id=alice" \
  -F "files=@/path/to/img1.jpg;type=image/jpeg" \
  -F "files=@/path/to/img2.png;type=image/png"
```

Curl (URL):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "subject_id=alice" \
  -F "image_urls=https://example.com/photo1.jpg" \
  -F "image_urls=https://example.com/photo2.jpg"
```

Curl (mixed - files + URLs):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "subject_id=alice" \
  -F "files=@/path/to/local.jpg;type=image/jpeg" \
  -F "image_urls=https://example.com/remote.jpg"
```

---

## Search (Top-K)

### `POST /v1/faces/search` (JSON)

**Auth:** API key

Request body:
```json
{
  "image_b64": "<base64-encoded-image or http-url>",
  "top_k": 5,
  "branch": "branch-001"
}
```

Note: You can use `images_b64` or `image` as aliases for `image_b64`.

Response:
```json
{
  "results": [
    {"subject_id": "alice", "similarity": 0.83, "point_id": "...", "image_id": "...", "thumb_path": "/thumbs/....jpg"},
    {"subject_id": "bob", "similarity": 0.71, "point_id": "...", "image_id": "...", "thumb_path": "/thumbs/....jpg"}
  ],
  "query_thumb_path": "/thumbs/tmp-....jpg"
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/search" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","top_k":5}'
```

### `POST /v1/faces/search_upload` (multipart)

**Auth:** API key

Form fields:

- `file` (image, required)
- `top_k` (optional, default `5`)
- `branch` (optional)

Response: Same as `/v1/faces/search`.

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/search_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "top_k=5" \
  -F "branch=branch-001" \
  -F "file=@/path/to/query.jpg;type=image/jpeg"
```

---


## Recognize (Best match + threshold)

### `POST /v1/faces/recognize` (JSON)

**Auth:** API key

Request body:
```json
{
  "image_b64": "<base64-encoded-image>",
  "top_k": 5,
  "min_similarity": 0.75,
  "branch": "branch-001",
  "day": "2024-05-20",
  "from_day": "2024-05-01",
  "to_day": "2024-05-31",
  "since_ts": 1716163200,
  "until_ts": 1717113600
}
```

Response:
```json
{
  "matched": true,
  "subject_id": "alice",
  "similarity": 0.83,
  "results": [
    {"subject_id": "alice", "similarity": 0.83, "point_id": "...", "image_id": "...", "thumb_path": "/thumbs/....jpg"}
  ],
  "meta": {
    "quality": {
      "blur": 127.0,
      "brightness": 149.1,
      "face_ratio": 0.115,
      "face_abs_px": 180.9,
      "landmark_score": 0.897,
      "yaw": 0.128,
      "pitch": -1.562,
      "face_crop_shape": [417, 290],
      "status": "ok",
      "reason": ""
    },
    "decision": {
      "status": "match",
      "min_similarity": 0.75,
      "top2_second": null,
      "top2_margin": null,
      "top2_required": 0.12
    }
  }
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/recognize" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","top_k":5,"min_similarity":0.75}'
```

### `POST /v1/faces/recognize_upload` (multipart)

**Auth:** API key
Form fields:

- `file` (image, required)
- `top_k` (optional, default `5`)
- `min_similarity` (optional)
- `branch` (optional)
- `day`, `from_day`, `to_day`, `since_ts`, `until_ts` (optional)

Response: Same as `/v1/faces/recognize`.

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/recognize_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "top_k=5" \
  -F "min_similarity=0.75" \
  -F "branch=branch-001" \
  -F "file=@/path/to/query.jpg;type=image/jpeg"
```

---

## Face comparison

### `POST /v1/face/compare_upload` (multipart)

Compare two face images directly and return similarity score.

**Auth:** API key

Form fields:

- `file1` (image, required)
- `file2` (image, required)

Response:
```json
{
  "similarity": 0.92,
  "match": true,
  "confidence": "High",
  "meta": {
    "timing_ms": 150,
    "image1_meta": {},
    "image2_meta": {}
  }
}
```

Confidence levels:
- `High`: similarity > 0.45
- `Medium`: similarity > 0.35
- `Low`: similarity <= 0.35

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/face/compare_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file1=@/path/to/face1.jpg;type=image/jpeg" \
  -F "file2=@/path/to/face2.jpg;type=image/jpeg"
```

---

## Cross-match

### `GET /v1/faces/cross_match/{subject_id}`

Find other subjects that look similar to the given subject. Useful for finding duplicate enrollments.

**Auth:** Optional (x-api-key for segregation)

Path params:

- `subject_id` (required)

Query params:

- `top_k` (default `20`)

Response:
```json
{
  "results": [
    {"subject_id": "bob", "similarity": 0.72, "point_id": "...", "image_id": "...", "thumb_path": "/thumbs/....jpg"}
  ]
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/faces/cross_match/alice?top_k=10"
```

---

## Frigate-compatible endpoints

Frigate typically calls `/v1/face/search`.

### `POST /v1/face/search` (JSON)

**Auth:** API key

Request body:
```json
{
  "image_b64": "<base64-encoded-image>",
  "camera": "optional",
  "reid_id": "optional",
  "frame_time": 0.0
}
```

Response:
```json
{
  "subject_id": "alice",
  "similarity": 0.83,
  "meta": {
    "quality": { "...": "..." },
    "decision": { "status": "embedded" }
  }
}
```

Behavior:

- If Qdrant is configured, this endpoint uses Qdrant search (top-1).
- Otherwise it falls back to the legacy in-memory index.
- If best similarity < `FACE_SERVICE_MIN_SIMILARITY`, returns `404`.

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/face/search" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64..."}'
```

### `POST /v1/face/search_upload` (multipart)

**Auth:** API key

Form fields:

- `file` (image, required)

Response: Same as `/v1/face/search`.

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/face/search_upload" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@/path/to/query.png;type=image/png"
```

---

## Subject management

### `GET /v1/faces/subjects`

Returns unique `subject_id` values currently present in Qdrant.

**Auth:** Optional (x-api-key for segregation)

Response:
```json
{
  "subjects": ["alice", "bob"]
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/faces/subjects"
```

### `DELETE /v1/faces/subjects/{subject_id}`

Deletes all vectors for a subject from Qdrant.

**Auth:** API key (Required)

Response:
```json
{
  "subject_id": "alice",
  "deleted": true
}
```

Curl:
```bash
curl -s -X DELETE "http://localhost:8001/v1/faces/subjects/alice"
```

---

## Subjects browser (Qdrant)

These endpoints are used by the web UI to browse subjects and their images.

### `GET /v1/subjects`

**Auth:** Optional (x-api-key for segregation)

Query params:

- `cursor` (optional) pagination cursor
- `limit` (default `50`, max `10000`)
- `with_counts` (default `true`) include embeddings count per subject
- `q` (optional) search/filter query string

Response (example):
```json
{
  "items": [
    {
      "subject_id": "alice",
      "embeddings_count": 12,
      "embeddings_cap": 10,
      "embeddings_capped": true
    },
    {
      "subject_id": "bob",
      "embeddings_count": 4,
      "embeddings_cap": 10,
      "embeddings_capped": false
    }
  ],
  "cursor": null
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/subjects?with_counts=true&limit=50"
```

### `GET /v1/subjects/{subject_id}`

Get details for a single subject.

**Auth:** Optional (x-api-key for segregation)

Response:
```json
{
  "subject_id": "alice",
  "embeddings_count": 12,
  "embeddings_cap": 10,
  "embeddings_capped": true
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/subjects/alice"
```

### `GET /v1/subjects/{subject_id}/images`

**Auth:** Optional (x-api-key for segregation)

Query params:

- `cursor` (optional)
- `limit` (default `50`, max `500`)

Response:
```json
{
  "items": [
    {
      "image_id": "6b87951debdac32b",
      "thumb_path": "/thumbs/6b87951debdac32b.jpg",
      "image_path": "/images/alice/6b87951debdac32b.jpg",
      "created_at": "2026-05-14T11:21:10.612673+00:00",
      "source": "enroll"
    }
  ],
  "cursor": null
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/subjects/alice/images?limit=50"
```

---

## Quality checks (eligibility filter)

Use this endpoint when you want to know whether an image is **eligible** to be embedded/enrolled under the current quality thresholds.

### `POST /v1/quality/check_upload` (multipart)

**Auth:** Optional (x-api-key for segregation)

Form fields:

- `file` (image, required)

Response (example):

```json
{
  "ok": true,
  "total_quality": "pass",
  "faces": [
    {
      "ok": true,
      "quality": {
        "blur": 127.0,
        "brightness": 149.1,
        "face_ratio": 0.115,
        "face_abs_px": 180.9,
        "landmark_score": 0.897,
        "yaw": 0.128,
        "pitch": -1.562,
        "face_crop_shape": [417, 290],
        "status": "ok",
        "reason": ""
      },
      "det_score": 0.897,
      "bbox": [271.8, 109.6, 452.7, 370.1]
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "timing": {
    "decode_ms": 3,
    "quality_ms": 12,
    "total_ms": 18
  }
}
```

Notes:

- `total_quality` is a single overall decision:
  - `pass` => eligible for embedding/enrollment
  - `fail` => not eligible under current thresholds
- `faces` is an array of per-face quality results (supports multi-face images).
- `annotated_image` contains a base64-encoded JPEG with face bounding boxes drawn.
- When a quality check fails, `quality.status` will be `rejected` and `quality.reason` indicates why (example: `pose_yaw`).

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/quality/check_upload" \
  -F "file=@/path/to/image.jpg"
```

---

## Privacy-Focused Extraction

Use this endpoint to extract individual face crops from an image while protecting the privacy of other people in the frame.

### `POST /v1/faces/privacy_extract` (JSON)

**Auth:** API key

Extracts individual face crops from an image with multiple people. For each detected face, it returns a focused crop where all other detected faces are blurred for privacy. Each face is also evaluated for quality on the original image before cropping. Optionally, it can perform face recognition for each extracted face against the enrolled database.

Request body:
```json
{
  "image_b64": "<base64-encoded-image or http-url>",
  "recognition": false,
  "top_k": 1,
  "branch": "optional-branch-filter",
  "day": "2026-05-22",
  "since_ts": 1777713396.0
}
```

Response:
```json
{
  "results": [
    {
      "bbox": [271.8, 109.6, 452.7, 370.1],
      "quality": {
        "blur": 127.0,
        "brightness": 149.1,
        "face_ratio": 0.115,
        "face_abs_px": 180.9,
        "landmark_score": 0.897,
        "yaw": 0.128,
        "pitch": -1.562,
        "face_crop_shape": [417, 290],
        "status": "ok",
        "reason": ""
      },
      "image_b64": "data:image/jpeg;base64,...",
      "recognition": {
        "matched": true,
        "subject_id": "employee-emerald-e00643",
        "similarity": 0.892,
        "results": [
           { "subject_id": "employee-emerald-e00643", "similarity": 0.892, "point_id": "..." }
        ]
      }
    }
  ]
}
```

Notes:
- `bbox`: Coordinates `[x1, y1, x2, y2]` of the face in the original image.
- `quality`: Quality evaluation metrics performed on the face before blurring/cropping.
- `image_b64`: A base64-encoded JPEG crop (with exactly 150px padding) where all *other* detected faces in the original frame have been obscured with a Gaussian blur.
- `recognition`: (Optional) Recognition results. Only included if `recognition: true` is passed in the request.
- `top_k`: (Optional) Max results for recognition. Default is 1. Supports `top_n` as alias.
- `branch`: (Optional) Filter for recognition.
- Date Filtering: Recognition search can be limited by time using `day`, `from_day`, `to_day`, `since_ts`, or `until_ts`.

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/privacy_extract" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...", "recognition": true, "top_k": 1}'
```

---

## Privacy Blur (v2 — bbox-targeted)

### `POST /v1/faces/privacy_blur` (JSON)

**Auth:** API key

Blur every detected face on the **full frame** except the one at the supplied `bbox`. Unlike `privacy_extract` (which auto-detects subjects and returns per-face crops), v2 lets the **caller specify which face to keep** and returns a single full-size image. Useful when you already know the target face box (e.g. from a prior detection/recognition call).

Request body:
```json
{
  "image_b64": "<base64-encoded-image or http-url>",
  "bbox": [271.8, 109.6, 452.7, 370.1],
  "blur_all": false
}
```

- `image_b64` (string, required) base64 **or** HTTP/HTTPS URL. Aliases: `image`, `images_b64`, `url`.
- `bbox` (array `[x1, y1, x2, y2]`, optional) the face to **keep** unblurred. The detected face with the highest IoU overlap is kept.
- `blur_all` (bool, default `false`) when `true`, blur **every** detected face including the `bbox` target.

Behavior:

- All other detected faces are obscured with a Gaussian blur scaled to face size.
- If `bbox` overlaps no detected face, that region simply stays clear (only detected faces are blurred).
- If `bbox` is omitted and `blur_all=false`, all detected faces are blurred (no target kept).
- If no faces are detected, the original image is returned with `blurred_count = 0`.

Response:
```json
{
  "image_b64": "data:image/jpeg;base64,...",
  "faces_total": 5,
  "blurred_count": 4,
  "kept_bbox": [271.8, 109.6, 452.7, 370.1]
}
```

- `image_b64`: full-frame JPEG (base64 data URL) with the non-target faces blurred.
- `faces_total`: number of faces detected in the frame.
- `blurred_count`: number of faces actually blurred.
- `kept_bbox`: the box left unblurred (detected match, or the raw input bbox if no overlap), `null` when `blur_all=true` or no bbox given.

Curl (base64):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/privacy_blur" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"...base64...","bbox":[271.8,109.6,452.7,370.1]}'
```

Curl (URL + blur all):
```bash
curl -s -X POST "http://localhost:8001/v1/faces/privacy_blur" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"https://example.com/photo.jpg","blur_all":true}'
```

---

## Async Jobs (202 + poll) — burst-friendly ingestion

Submit heavy image work as a **job**: the call returns **immediately** (202 + `job_id`) and background workers process it at server capacity. Use this when firing **bursts of events** — the client never waits on the processing queue.

**Measured:** 30-job burst → all submits answered in ~37 ms each, all 30 processed in ~2 s, zero drops.

### `POST /v1/jobs`

**Auth:** API key (forwarded to the underlying endpoint — data segregation applies as usual)

Request body:
```json
{
  "endpoint": "/v1/faces/privacy_extract",
  "payload": { "image_b64": "<base64 or http-url>", "recognition": true, "top_k": 1 }
}
```

- `endpoint` (string, required) — which heavy endpoint to run. Allowed:
  `/v1/faces/privacy_extract`, `/v1/faces/privacy_blur`, `/v1/faces/recognize`, `/v1/faces/search`, `/v1/face/search`
- `payload` (object, required) — the exact JSON body that endpoint normally takes.

Response — **202 Accepted, returns in milliseconds**:
```json
{ "job_id": "4d7d1c85-9b74-4273-afc4-0830f267e8da", "status": "queued", "poll": "/v1/jobs/4d7d1c85-..." }
```

Errors: `400` endpoint not allowed · `503 Retry-After` job queue full (> `JOB_QUEUE_MAX`).

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/jobs" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"endpoint":"/v1/faces/privacy_extract","payload":{"image_b64":"...base64...","recognition":true}}'
```

### `GET /v1/jobs/{job_id}`

Poll job status and fetch the result.

**Auth:** none required for polling (job_id is an unguessable UUID)

Response while running:
```json
{ "job_id": "...", "status": "running", "endpoint": "/v1/faces/privacy_extract", "started_at": 1783156000.1 }
```

Response when finished:
```json
{
  "job_id": "...",
  "status": "done",
  "http_status": 200,
  "duration_ms": 746,
  "result": { "results": [ { "bbox": [...], "quality": {...}, "image_b64": "data:image/jpeg;base64,...", "recognition": {...} } ] }
}
```

- `status`: `queued` → `running` → `done` | `failed`
- `result`: the **exact response body** the underlying endpoint would have returned synchronously.
- `http_status`: the underlying endpoint's status (a 4xx there → job `status:"failed"` with the error in `result`).
- `404`: unknown `job_id` or the job expired (`JOB_TTL_SEC`, default 1 h).

Curl:
```bash
curl -s "http://localhost:8001/v1/jobs/4d7d1c85-9b74-4273-afc4-0830f267e8da" | python3 -m json.tool
```

### Configuration

- `JOB_WORKERS` (default `4`) — background executors per server process.
- `JOB_QUEUE_MAX` (default `256`) — max queued jobs per process; beyond → `503 Retry-After`.
- `JOB_TTL_SEC` (default `3600`) — results kept this long, then deleted.
- `JOBS_DIR` (default `/data/jobs`) — shared result store (works across worker processes).

### Notes / semantics

- **Fire a burst safely:** submit N jobs (each 202 in ms), poll each `job_id` — the server drains at its own capacity; nothing is dropped up to `JOB_QUEUE_MAX`.
- Jobs are **best-effort ephemeral**: a server restart loses queued/running jobs (their status stops updating). Re-submit on timeout.
- The job executes with the submitter's `x-api-key`, so tenant data-segregation is identical to calling the endpoint directly.
- Sync endpoints are unchanged — use them when you want the answer in one round-trip; use jobs for bursts.

---

## Recognition events (ingestion + audit)

These endpoints store recognition attempts in the local events DB (SQLite) and save event images/thumbnails.

### `POST /v1/events/recognition` (multipart)

**Auth:** Optional (x-api-key for segregation)

Form fields:

- `file` (image, required)
- `camera` (text, required)
- `source_path` (text, optional, default `""`)
- `ts` (float, optional; default current time)
- `top_k` (int, optional, default `5`)
- `min_similarity` (float, optional)
- `process_all_faces` (bool, optional, default `false`) process multiple faces per image
- `branch` (text, optional) filter recognition by branch

Response:
```json
{
  "event_id": "42c72ad0-1f38-4015-9986-0d572ce9a05e",
  "ts": 1778759909.49,
  "camera": "front_door",
  "source_path": "snapshot.jpg",
  "decision": "match",
  "subject_id": "alice",
  "similarity": 0.95,
  "processing_ms": 70,
  "model_ms": 82,
  "rejected_reason": null,
  "bbox": [271.8, 109.6, 452.7, 370.1, 622.0, 656.0],
  "det_score": 0.897,
  "image_path": "/events/accepted/front_door/42c72ad0....jpg",
  "thumb_path": "/thumbs/evt-42c72ad0....jpg",
  "image_saved_at": 1778759909.59,
  "meta": {
    "quality": { "...": "..." },
    "decision": {
      "status": "match",
      "matched": true,
      "min_similarity": 0.6,
      "auto_add_embedding": { "enabled": true, "added": true, "reason": null, "min_similarity": 0.8 },
      "no_match_auto_enroll": { "enabled": true, "enrolled": false }
    },
    "timing": {
      "decode_ms": 11,
      "detect_embed_ms": 70,
      "gpu_queue_wait_ms": 0.16,
      "gpu_exec_ms": 69.8,
      "quality_ms": 1,
      "qdrant_ms": 2,
      "save_ms": 16,
      "face_total_ms": 113,
      "total_ms": 195
    },
    "top_k": 5,
    "face_index": 0,
    "faces_total": 1,
    "faces_processed": 1,
    "multi_face": false
  },
  "feedback_label": null,
  "feedback_note": null,
  "feedback_updated_at": null
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/events/recognition" \
  -F "camera=front_door" \
  -F "top_k=5" \
  -F "file=@/path/to/image.jpg;type=image/jpeg"
```

### `GET /v1/events/recognition`

List recognition events with filtering and pagination.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `camera` (optional) filter by camera name
- `subject_id` (optional) filter by matched subject
- `source_path` (optional) filter by source path
- `decision` (optional) filter by decision: `match`, `no_match`, `rejection`
- `min_similarity` (optional) minimum similarity filter
- `max_similarity` (optional) maximum similarity filter
- `day` (optional) filter by specific date (e.g. `2026-05-14`)
- `from_day` (optional) start date range
- `to_day` (optional) end date range
- `since_ts` (optional) start timestamp (float epoch)
- `until_ts` (optional) end timestamp (float epoch)
- `limit` (default `100`)
- `cursor` (optional) pagination cursor (float timestamp)

Response:
```json
{
  "items": [ { "...recognition event..." } ],
  "cursor": 1778759909.59
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/events/recognition?camera=front_door&limit=10"
```

### `GET /v1/events/recognition/{event_id}`

Fetch a single stored event by ID.

**Auth:** Optional (x-api-key for segregation)

Response: Single `RecognitionEventResponse` object (same shape as items in the list).

Curl:
```bash
curl -s "http://localhost:8001/v1/events/recognition/42c72ad0-1f38-4015-9986-0d572ce9a05e"
```

### `POST /v1/events/recognition/{event_id}/feedback`

Submit human feedback for a recognition event (used for accuracy tracking).

**Auth:** Optional (x-api-key for segregation)

Request body:
```json
{
  "label": "tp",
  "note": "Confirmed correct match"
}
```

Valid `label` values: `tp` (true positive), `fp` (false positive), `fn` (false negative), `ignore`.

Response:
```json
{
  "event_id": "42c72ad0-...",
  "updated": true,
  "feedback_label": "tp",
  "feedback_note": "Confirmed correct match",
  "feedback_updated_at": 1778760000.0
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/events/recognition/42c72ad0-.../feedback" \
  -H "Content-Type: application/json" \
  -d '{"label":"tp","note":"Confirmed correct match"}'
```

### `POST /v1/events/recognition/forward`

Forward a stored recognition event to an external webhook URL.

**Auth:** Optional (x-api-key for segregation)

Request body:
```json
{
  "event_id": "<uuid>",
  "target_url": "https://example.com/hook"
}
```

Response:
```json
{
  "forwarded": true,
  "status_code": 200
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/events/recognition/forward" \
  -H "Content-Type: application/json" \
  -d '{"event_id":"42c72ad0-...","target_url":"https://example.com/hook"}'
```

### `GET /v1/events/recognition/cameras`

List all unique camera names from stored events.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `limit` (default `5000`)

Response:
```json
["front_door", "lobby", "parking"]
```

Curl:
```bash
curl -s "http://localhost:8001/v1/events/recognition/cameras"
```

### `GET /v1/events/recognition/stats`

Get aggregated recognition statistics.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `day` (optional) specific date
- `from_day` (optional) start date
- `to_day` (optional) end date
- `since_ts` (optional) start timestamp
- `until_ts` (optional) end timestamp
- `camera` (optional) filter by camera

Response:
```json
{
  "total": 500,
  "match": 350,
  "no_match": 120,
  "rejection": 30,
  "unique_matches": 45,
  "by_camera": {
    "front_door": { "total": 200, "match": 150, "no_match": 40, "rejection": 10 }
  }
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/events/recognition/stats?day=2026-05-14"
```

### `GET /v1/events/recognition/feedback_stats`

Get feedback accuracy statistics.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `since_ts` (optional)
- `until_ts` (optional)
- `camera` (optional)

Response:
```json
{
  "total": 500,
  "labeled": 100,
  "unlabeled": 400,
  "tp": 80,
  "fp": 10,
  "fn": 5,
  "ignore": 5,
  "fp_rate_match": 0.05,
  "by_decision": {
    "match": { "total": 350, "tp": 80, "fp": 10, "fn": 0, "ignore": 3 }
  }
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/events/recognition/feedback_stats"
```

---

## Search history

These endpoints log and query face search operations.

### `GET /v1/search_history`

List search events with pagination.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `limit` (default `100`)
- `cursor` (optional) pagination cursor (float timestamp)
- `day` (optional) specific date
- `from_day` (optional) start date
- `to_day` (optional) end date
- `since_ts` (optional) start timestamp
- `until_ts` (optional) end timestamp

Response:
```json
{
  "items": [
    {
      "event_id": "...",
      "ts": 1778760000.0,
      "query_image_path": "/events/...",
      "query_thumb_path": "/thumbs/...",
      "top_subject_id": "alice",
      "top_similarity": 0.95,
      "results": [ { "...FaceSearchTopKItem..." } ],
      "meta": {}
    }
  ],
  "cursor": null
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/search_history?limit=20"
```

### `GET /v1/search_history/stats`

Get search history statistics.

**Auth:** Optional (x-api-key for segregation)

Query params:

- `match_threshold` (default `0.8`) similarity threshold to count as a match
- `day` (optional) specific date
- `from_day` (optional) start date
- `to_day` (optional) end date
- `since_ts` (optional) start timestamp
- `until_ts` (optional) end timestamp

Response:
```json
{
  "match": 150,
  "no_match": 30,
  "total": 180
}
```

Curl:
```bash
curl -s "http://localhost:8001/v1/search_history/stats?match_threshold=0.8"
```

### `GET /v1/search_history/asset/image/{event_id}`

Serve the full query image for a search event.

**Auth:** Optional (x-api-key for segregation)

Response: JPEG image (`image/jpeg`).

Curl:
```bash
curl -s "http://localhost:8001/v1/search_history/asset/image/EVENT_ID" -o query.jpg
```

### `GET /v1/search_history/asset/thumb/{event_id}`

Serve the thumbnail for a search event query.

**Auth:** Optional (x-api-key for segregation)

Response: JPEG image (`image/jpeg`).

Curl:
```bash
curl -s "http://localhost:8001/v1/search_history/asset/thumb/EVENT_ID" -o thumb.jpg
```

---

## Troubleshooting

### Face not detected

If you get:

- `no face detected`

Try:

- Use a clearer, front-facing image
- Lower `BUFFALO_MIN_DET_SCORE` (example: `0.2`)
- Ensure the image is not extremely dark/blurry

### Enrollment blocked by duplicate check

If you get `no faces embedded from provided images` but quality check passes, the duplicate check may be blocking. The face may already exist under a different `subject_id`. Check `ENROLL_DUPLICATE_CHECK_ENABLE` and `ENROLL_DUPLICATE_MIN_SIM`.

### Qdrant errors

- Ensure `qdrant` service is running:
  - `docker compose up -d qdrant`
- Ensure `QDRANT_URL` is reachable from `face_service`

### CUDA provider not used (slow inference)

Check:

- `GET /debug/providers` and confirm `CUDAExecutionProvider` appears in `insightface.session_providers`.
- If it only shows `CPUExecutionProvider`, inference will be slow (hundreds of ms).
