# API Latency Report

**Generated:** 2026-07-08
**Service:** Face Recognition (InsightFace buffalo_l + Qdrant)
**Tested against:** `https://api.face.service.tools.thefusionapps.com` (public) and `http://localhost:8001` (in-box)
**Method:** single warm request per endpoint, `curl -w` timing breakdown. Test image ~325 KB JPEG.

---

## 1. What a request's latency is made of

Every number below is **wall-clock at the client**. It is the sum of these layers, not just model time:

| Layer | What it is | Typical cost |
|-------|-----------|--------------|
| **DNS** | resolve domain → IP | 4–24 ms (cached after 1st) |
| **TCP connect** | open socket | 3–10 ms |
| **TLS handshake** | HTTPS setup (cert exchange) | **~80 ms per NEW connection** |
| **Upload** | send request body to server | ∝ payload size ÷ client uplink |
| **Server compute** | decode + GPU inference + DB + disk | the actual work (see §4) |
| **Download** | response body back to client | usually small |

**Key point:** TLS (~80 ms) and upload are paid **per request** if the client (e.g. Postman) does not reuse the connection. They are **not** server slowness. Enable HTTP keep-alive to pay TLS once.

**Payload note:** JSON base64 inflates image size by **+33%** (325 KB image → 433 KB body). On a slow client uplink this alone can add seconds. Multipart upload sends the raw 325 KB.

---

## 2. GET endpoints (public domain, warm)

| Endpoint | Total | Server work | Latency includes |
|----------|-------|-------------|------------------|
| `GET /health` | 0.12 s | ~40 ms | TLS + counters + psutil system stats |
| `GET /v1/stats` | 0.16 s | ~70 ms | TLS + Qdrant count |
| `GET /metrics` | 0.11 s | ~20 ms | TLS + Prometheus dump |
| `GET /v1/subjects` | 0.10 s | ~15 ms | TLS + Qdrant scroll |
| `GET /v1/faces/subjects` | 0.11 s | ~30 ms | TLS + Qdrant distinct subjects |
| `GET /v1/groups` | 0.10 s | ~10 ms | TLS + Qdrant fetch |
| `GET /v1/branches` | 0.11 s | ~30 ms | TLS + Qdrant fetch |
| `GET /v1/events/recognition` (list) | 0.11 s | ~15 ms | TLS + SQLite indexed query |
| `GET /v1/events/recognition/stats` | 0.16 s | ~70 ms | TLS + SQLite aggregate (22k rows) |
| **`GET /v1/events/recognition/cameras`** | **0.61 s** | **~535 ms** ⚠ | TLS + SQLite (21 ms) **+ filesystem walk of `/data/events` (~450 ms)** |
| `GET /v1/events/recognition/feedback_stats` | 0.12 s | ~40 ms | TLS + SQLite aggregate |
| `GET /v1/search_history` | 0.10 s | ~15 ms | TLS + SQLite indexed query |
| `GET /v1/search_history/stats` | 0.11 s | ~35 ms | TLS + SQLite aggregate |
| `GET /debug/providers` | 0.09 s | ~10 ms | TLS + ORT provider list |

---

## 3. POST / GPU endpoints (public domain, warm)

| Endpoint | Total | Body size | Latency includes |
|----------|-------|-----------|------------------|
| `POST /v1/faces/recognize_upload` | 0.32 s | 325 KB | TLS + upload + decode + **GPU detect+embed (~53 ms)** + Qdrant search |
| `POST /v1/faces/recognize` (json) | 0.31 s | 433 KB | TLS + upload(b64+33%) + decode + GPU + Qdrant |
| `POST /v1/faces/search_upload` | 0.43 s | 325 KB | above + inline thumbnail save to disk |
| `POST /v1/faces/search` (json) | 0.45 s | 433 KB | above + inline thumbnail save |
| `POST /v1/quality/check_upload` | 0.32 s local / **1.42 s Postman** | 325 KB up, **172 KB down** | TLS + upload + decode + GPU detect + quality eval + **`annotated_image` = 172 KB base64 JPEG in response** (see note) |
| `POST /v1/face/compare_upload` | 0.35 s | 650 KB | TLS + upload(2 images) + **2× GPU embed** |
| `POST /v1/faces/privacy_blur` | 0.42 s | 433 KB | TLS + upload + decode + GPU detect-all + Gaussian blur + JPEG encode |
| `POST /v1/faces/privacy_extract` | 0.28 s | 433 KB | TLS + upload + decode + GPU detect-all + per-face crop/blur |
| `POST /v1/events/recognition` (ingest) | 0.50 s | 325 KB | TLS + upload + decode + GPU + Qdrant + **SQLite write + image/thumb save to disk** |

Model inference itself (GPU) is only **~26–53 ms**. Most of each POST's time is TLS + upload + decode + disk I/O, not the model.

### `quality/check_upload` deep-dive (the "1.42 s in Postman" case)

Server compute = **66 ms** (from its own `timing`: `detect_ms 49, quality_ms 50, total_ms 66`). The 1.42 s was **all client/network**, dominated by the response payload:

| piece | cost |
|-------|------|
| server compute | **66 ms** |
| **`annotated_image` download (boxes-drawn JPEG, base64)** | **172 KB** ← the killer over a remote link |
| upload (325 KB image) | ∝ client uplink |
| TLS handshake | ~80 ms |
| Postman rendering 172 KB base64 | UI overhead |

**Fix applied:** added `annotate` form field (default `true` = backward compatible). Pass **`annotate=false`** to skip the annotated image:

| mode | total (local) | response |
|------|---------------|----------|
| `annotate=true` (old) | 92 ms | **172 KB** |
| **`annotate=false`** | 75 ms | **535 B** |

Over a slow/remote link the 172 KB download is what turned 66 ms of work into 1.42 s. `annotate=false` removes it → seconds gone.

---

## 4. Server-side breakdown (from `meta.timing`, recognize)

```
decode_ms: 20      # JPEG → numpy (CPU)
model_ms:  53      # GPU detect + embed
search_ms:  3      # Qdrant vector search
total_ms:  77      # server compute only (no network/TLS/upload)
```

---

## 5. Findings

**Server is healthy.** No endpoint is slow warm — all 0.1–0.6 s. Nothing takes "seconds" server-side.

The perceived "seconds in Postman" comes from **3 sources, none is the model:**

1. **`/events/recognition/cameras` = 0.6 s always** — redundant filesystem walk of `/data/events` (stats thousands of image files). SQL alone is 21 ms. **Fix: drop the fs-walk, use SQL result.**
2. **Client upload size** — JSON base64 = 433 KB/request. On a slow/remote uplink this takes seconds (client network, not server). **Fix: use multipart, or resize images before upload.**
3. **Lock contention under ingest** — SQLite runs in `journal_mode=delete` (rollback journal, not WAL) + a global write lock. While the poller ingests events, dashboard reads (`stats`/`cameras`/`events`) block behind the exclusive lock → seconds. Observed `stats` at 4.1 s under concurrent load vs 50 ms idle. **Fix: enable WAL mode + `busy_timeout`.**

Plus: **TLS ~80 ms per request** if the client doesn't reuse the connection (Postman keep-alive off).

---

## 6. Recommended fixes (ranked)

| # | Fix | Where | Impact |
|---|-----|-------|--------|
| 1 | `PRAGMA journal_mode=WAL; synchronous=NORMAL; busy_timeout=5000` | `src/services/events_store.py` `_connect` | Kills seconds-under-ingest for all dashboard reads |
| 2 | Drop filesystem walk in cameras handler, use SQL only | `app.py` `list_recognition_cameras` | `cameras` 0.6 s → 0.03 s |
| 3 | Background the `_save_search_query_assets` disk write | `app.py` search/recognize handlers | search endpoints −100–200 ms |
| ✅ | **`annotate=false`** option on quality/check_upload (DONE) | `app.py` `quality_check_upload` | response 172 KB → 535 B; kills the 1.42 s |
| 4 | Client: multipart instead of base64, resize before upload | frontend / caller | −33% upload; avoids slow-uplink seconds |
| 5 | Client: HTTP keep-alive | Postman / caller | −80 ms TLS per request |

**Bottom line:** Model + GPU are fast (~53 ms). Latency lives in network (TLS + upload), one bad endpoint (cameras fs-walk), and DB lock mode (not WAL). Fixes 1–2 are code, low-risk, no pipeline change.
