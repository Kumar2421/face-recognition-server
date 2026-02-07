# Face Service API (Swagger-style)

This service provides face embedding, enrollment, and similarity search using **InsightFace (Buffalo-L)** + **Qdrant**. It stores one vector per image and generates 256px thumbnails for gallery views.

## Interactive docs

FastAPI automatically serves Swagger UI:

- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

The built-in demo web UI is:

- `http://localhost:8000/ui`

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

## Roadmap endpoints (to be added)

- `GET /v1/subjects?cursor=&limit=&with_counts=true`
- `GET /v1/subjects/{subject_id}/images?cursor=&limit=`
- `POST /v1/ingest/url`
- `POST /v1/ingest/confirm`
- `GET /v1/activity?cursor=&limit=`

Auth and Ops (Phase 4):

- API keys on mutating routes via `X-API-Key`.
- Basic rate limiting per IP/key for enroll/search.
- Audit log JSONL at `/data/logs/audit.jsonl`.
- Prometheus metrics at `/metrics` in addition to `/v1/stats`.

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
curl -s -X POST "http://localhost:8000/v1/faces/add" \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"alice","images_b64":["...base64..."]}'
```

### `POST /v1/faces/add_upload` (multipart)

Form fields:

- `subject_id` (text)
- `files` (one or more images)

Curl (Windows `curl.exe`):
```powershell
curl.exe -s -X POST "http://localhost:8000/v1/faces/add_upload" `
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
curl.exe -s -X POST "http://localhost:8000/v1/faces/search_upload" `
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
curl.exe -s -X POST "http://localhost:8000/v1/faces/recognize_upload" `
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
curl.exe -s -X POST "http://localhost:8000/v1/face/search_upload" `
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

