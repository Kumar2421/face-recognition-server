# Face Service API (Swagger-style)

This service provides face embedding, enrollment, and similarity search using **InsightFace (Buffalo-L)** + **Qdrant**. It stores one vector per image and generates 256px thumbnails for gallery views.

## Interactive docs

FastAPI automatically serves Swagger UI:

- `http://localhost:8001/docs` (Swagger UI)
- `http://localhost:8001/redoc` (ReDoc)

The built-in demo web UI is:

- `http://localhost:8001/ui`

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

GPU / performance:

- `GPU_INFERENCE_MANAGER` (`1` enables serialized GPU execution, `0` disables)
- `GPU_QUEUE_MAX` (default: `256`) max queue size for GPU inference manager
- `GPU_BATCH_WINDOW_MS` (default: `0`) micro-batching window (0 disables batching)
- `BUFFALO_ENABLE_FALLBACK_VARIANTS` (`0` disables expensive rotation/scale fallback on no-face)

Storage:

- `EVENTS_DIR` (default: `/data/events`) where event images are stored (`accepted/`, `rejected/`, `no_match/`)
- `THUMBS_DIR` (default: `/data/thumbs`) thumbnails directory
- `IMAGES_DIR` (default: `/data/images`) optional image storage directory

Notes:

- Qdrant point IDs must be **UUID** or **integer**. This service uses deterministic UUIDs derived from `subject_id` and `image_id`.
- Detection robustness:
  - If detection fails, the service retries with rotated and downscaled variants.
- Thumbnails are served at `/thumbs/{image_id}.jpg` (root configurable via `THUMBS_DIR`).

## Data model

### Face embedding

- A single image produces one embedding vector (Buffalo-L, typically 512-D)
- The service stores one vector per uploaded/enrolled image in Qdrant
- Each vector has payload:
  - `subject_id` (string)
  - `image_id` (string, UUID per image)
  - `created_at` (ISO8601 string)
  - `thumb_path` (string, e.g. `/thumbs/{image_id}.jpg`)
  - `source` (one of: `enroll`, `external`, `ingested`)
  - optional `filename` (string) for upload endpoints

## Endpoints

### Health

#### `GET /health`

Returns service health and Qdrant status.

Response (example):
```json
{
  "ok": true,
  "subjects": 1,
  "qdrant_enabled": true,
  "qdrant_collection": "frigate_faces"
}
```

---

## Stats

### `GET /v1/stats`

Returns global counters and Qdrant status.

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

---

## Observability / Debug

### `GET /metrics`

Prometheus metrics.

### `GET /debug/providers`

Shows ONNXRuntime providers and what providers InsightFace sessions were created with.

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
    "models": ["detection", "recognition"],
    "session_providers": {
      "detection": ["CUDAExecutionProvider", "CPUExecutionProvider"],
      "recognition": ["CUDAExecutionProvider", "CPUExecutionProvider"]
    }
  }
}
```

## Enrollment (Add)

You can enroll faces using either JSON (base64) or multipart upload.

### `POST /v1/faces/add` (JSON)

Request body:
```json
{
  "subject_id": "alice",
  "images_b64": ["<base64-encoded-image>", "<base64-encoded-image>"]
}
```

Response:
```json
{
  "subject_id": "alice",
  "num_images": 2,
  "num_embedded": 2,
  "embedding_dim": 512
}
```

Curl:
```bash
curl -s -X POST "http://localhost:8001/v1/faces/add" \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"alice","images_b64":["...base64..."]}'
```

### `POST /v1/faces/add_upload` (multipart)

Form fields:

- `subject_id` (text)
- `files` (one or more images)

Curl (Windows `curl.exe`):
```powershell
curl.exe -s -X POST "http://localhost:8001/v1/faces/add_upload" `
  -F "subject_id=alice" `
  -F "files=@D:\\path\\to\\img1.jpg;type=image/jpeg" `
  -F "files=@D:\\path\\to\\img2.png;type=image/png"
```

---

## Search (Top-K)

### `POST /v1/faces/search` (JSON)

Request body:
```json
{
  "image_b64": "<base64-encoded-image>",
  "top_k": 5
}
```

Response:
```json
{
  "results": [
    {"subject_id":"alice","similarity":0.83,"point_id":"...","image_id":"...","thumb_path":"/thumbs/....jpg"},
    {"subject_id":"bob","similarity":0.71,"point_id":"...","image_id":"...","thumb_path":"/thumbs/....jpg"}
  ]
}
```

### `POST /v1/faces/search_upload` (multipart)

Form fields:

- `file` (image)
- `top_k` (optional, default `5`)

Curl:
```powershell
curl.exe -s -X POST "http://localhost:8001/v1/faces/search_upload" `
  -F "top_k=5" `
  -F "file=@D:\\path\\to\\query.jpg;type=image/jpeg"
```

Response:
```json
{
  "results": [
    {"subject_id":"alice","similarity":0.83,"point_id":"...","image_id":"...","thumb_path":"/thumbs/....jpg"}
  ],
  "query_thumb_path": "/thumbs/tmp-....jpg"
}
```

---

## Recognize (Best match + threshold)

### `POST /v1/faces/recognize` (JSON)

Request body:
```json
{
  "image_b64": "<base64-encoded-image>",
  "top_k": 5,
  "min_similarity": 0.75
}
```

Response:
```json
{
  "matched": true,
  "subject_id": "alice",
  "similarity": 0.83,
  "results": [
    {"subject_id":"alice","similarity":0.83,"point_id":"...","image_id":"...","thumb_path":"/thumbs/....jpg"}
  ]
}
```

### `POST /v1/faces/recognize_upload` (multipart)

Form fields:

- `file` (image)
- `top_k` (optional, default `5`)
- `min_similarity` (optional)

Curl:
```powershell
curl.exe -s -X POST "http://localhost:8001/v1/faces/recognize_upload" `
  -F "top_k=5" `
  -F "min_similarity=0.75" `
  -F "file=@D:\\path\\to\\query.jpg;type=image/jpeg"
```

---

## Frigate-compatible endpoint

Frigate typically calls `/v1/face/search`.

### `POST /v1/face/search` (JSON)

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
  "similarity": 0.83
}
```

Behavior:

- If Qdrant is configured, this endpoint uses Qdrant search (top-1).
- Otherwise it falls back to the legacy in-memory index.
- If best similarity < `FACE_SERVICE_MIN_SIMILARITY`, returns `404`.

### `POST /v1/face/search_upload` (multipart)

Form fields:

- `file` (image)

Curl:
```powershell
curl.exe -s -X POST "http://localhost:8001/v1/face/search_upload" `
  -F "file=@D:\\path\\to\\query.png;type=image/png"
```

---

## Subject management

### `GET /v1/faces/subjects`

Returns unique `subject_id` values currently present in Qdrant.

Response:
```json
{
  "subjects": ["alice", "bob"]
}
```

### `DELETE /v1/faces/subjects/{subject_id}`

Deletes all vectors for a subject.

Response:
```json
{
  "subject_id": "alice",
  "deleted": true
}
```

---

## Subjects browser (Qdrant)

These endpoints are implemented and used by the web UI to browse subjects and their images.

### `GET /v1/subjects`

Query params:

- `cursor` (optional)
- `limit` (default `50`)
- `with_counts` (default `true`) include embeddings count per subject

Response (example):
```json
{
  "items": [
    {"subject_id": "alice", "embeddings_count": 12},
    {"subject_id": "bob", "embeddings_count": 4}
  ],
  "cursor": null
}
```

### `GET /v1/subjects/{subject_id}/images`

Query params:

- `cursor` (optional)
- `limit` (default `50`)

Response contains a paginated list of subject images/vectors (includes `image_id` and `thumb_path` where available).

---

## Recognition events (ingestion + audit)

These endpoints store recognition attempts in the local events DB (SQLite) and save event images/thumbnails.

### `POST /v1/events/recognition` (multipart)

Form fields:

- `file` (image)
- `camera` (required)
- `source_path` (optional)
- `ts` (optional float seconds; default now)
- `top_k` (optional, default `5`)
- `min_similarity` (optional)
- `process_all_faces` (optional, default `false`) process multiple faces per image

Response includes `meta.timing` when available:

- `decode_ms`
- `detect_embed_ms`
- `gpu_queue_wait_ms`
- `gpu_exec_ms`
- `quality_ms`
- `qdrant_ms`
- `save_ms`
- `face_total_ms`
- `total_ms`

### `GET /v1/events/recognition`

Query params:

- `camera`, `subject_id`, `decision`
- `since_ts`, `until_ts`
- `limit` (default `100`)
- `cursor` (optional)

### `GET /v1/events/recognition/{event_id}`

Fetch a single stored event.

### `POST /v1/events/recognition/forward`

Request body:
```json
{ "event_id": "<uuid>", "target_url": "https://example.com/hook" }
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

### Qdrant errors

- Ensure `qdrant` service is running:
  - `docker compose up -d qdrant`
- Ensure `QDRANT_URL` is reachable from `face_service`

### CUDA provider not used (slow inference)

Check:

- `GET /debug/providers` and confirm `CUDAExecutionProvider` appears in `insightface.session_providers`.
- If it only shows `CPUExecutionProvider`, inference will be slow (hundreds of ms).

