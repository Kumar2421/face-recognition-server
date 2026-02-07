import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from quality import FaceQualityEvaluator
from embedders.buffalo_l import (
    BuffaloLEmbedder,
    _l2_normalize,
    _quality_check_and_embed as _embed_quality_check_and_embed,
)
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from ui_page import ui_html
from events_store import EventsStore, RecognitionEvent
from config_loader import apply_env_defaults_from_config, load_config


logger = logging.getLogger("uvicorn.error")


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


def _passes_top2_margin(results: list[dict[str, Any]], best_sim: float) -> tuple[bool, float | None, float | None, float | None]:
    """Adaptive top-2 margin gating.

    Returns: (passes, second_sim, margin, required_margin)
    """
    try:
        required = _as_float(os.environ.get("FACE_SERVICE_TOP2_MARGIN", "0"), 0.0)
    except Exception:
        required = 0.0
    if required <= 0.0:
        return True, None, None, None

    try:
        high_conf = _as_float(os.environ.get("FACE_SERVICE_TOP2_HIGH_CONF", "0"), 0.0)
    except Exception:
        high_conf = 0.0
    if high_conf > 0.0 and float(best_sim) >= float(high_conf):
        return True, None, None, required

    second_sim: float | None = None
    try:
        if results and len(results) >= 2:
            second_sim = float(results[1].get("similarity") or 0.0)
    except Exception:
        second_sim = None
    if second_sim is None:
        return True, None, None, required

    margin = float(best_sim) - float(second_sim)
    return margin >= float(required), second_sim, margin, required


# ---------------------- Metrics (Prometheus) ----------------------
_REQ_TOTAL = Counter(
    "face_requests_total",
    "Total requests by endpoint",
    labelnames=("endpoint",),
)
_REQ_LAT = Histogram(
    "face_request_latency_seconds",
    "Request latency by endpoint",
    labelnames=("endpoint",),
)

_EMB_LAT = Histogram("face_embedding_latency_seconds", "Embedding latency seconds")
_EMB_FAIL = Counter("face_embedding_failures_total", "Embedding failures total")

_QCHECK_TOTAL = Counter("face_quality_checked_total", "Quality checks total")
_QREJ_TOTAL = Counter(
    "face_quality_rejected_total",
    "Quality rejections by reason",
    labelnames=("reason",),
)

_SEARCH_TOTAL = Counter("face_search_total", "Face search total")
_SEARCH_MATCH = Counter("face_search_match_total", "Face search matches total")
_SEARCH_NOMATCH = Counter("face_search_nomatch_total", "Face search no-matches total")

_QDRANT_SEARCH_LAT = Histogram("qdrant_search_latency_seconds", "Qdrant search latency seconds")
_QDRANT_UPSERT_LAT = Histogram("qdrant_upsert_latency_seconds", "Qdrant upsert latency seconds")
_QDRANT_ERR = Counter("qdrant_errors_total", "Qdrant errors total")


def _t() -> float:
    try:
        return time.time()
    except Exception:
        return float(datetime.now(tz=timezone.utc).timestamp())


def _debug_enabled() -> bool:
    return os.environ.get("FACE_SERVICE_DEBUG", "0") in ("1", "true", "True")


def _debug(msg: str) -> None:
    if _debug_enabled():
        try:
            logger.info("%s", msg)
        except Exception:
            pass


@dataclass
class FaceIndex:
    subject_ids: list[str]
    mean_embeddings: np.ndarray  # shape: (N, D)


class FaceSearchRequest(BaseModel):
    image_b64: str
    camera: str | None = None
    reid_id: str | None = None
    frame_time: float | None = None


class FaceSearchResponse(BaseModel):
    subject_id: str
    similarity: float
    meta: dict[str, Any] | None = None


class FaceAddRequest(BaseModel):
    subject_id: str
    images_b64: list[str]


class FaceAddResponse(BaseModel):
    subject_id: str
    num_images: int
    num_embedded: int
    embedding_dim: int | None = None
    meta: dict[str, Any] | None = None


class FaceSearchTopKRequest(BaseModel):
    image_b64: str
    top_k: int = 5


class FaceSearchTopKItem(BaseModel):
    subject_id: str
    similarity: float
    point_id: str
    image_id: str | None = None
    thumb_path: str | None = None


class FaceSearchTopKResponse(BaseModel):
    results: list[FaceSearchTopKItem]
    query_thumb_path: str | None = None


class FaceRecognizeRequest(BaseModel):
    image_b64: str
    top_k: int = 5
    min_similarity: float | None = None


class FaceRecognizeResponse(BaseModel):
    matched: bool
    subject_id: str | None = None
    similarity: float | None = None
    results: list[FaceSearchTopKItem] = []
    meta: dict[str, Any] | None = None


class RecognitionEventResponse(BaseModel):
    event_id: str
    ts: float
    camera: str
    source_path: str
    decision: str
    subject_id: str | None = None
    similarity: float | None = None
    rejected_reason: str | None = None
    bbox: list[float] | None = None
    det_score: float | None = None
    image_path: str
    thumb_path: str
    image_saved_at: float | None = None
    meta: dict[str, Any] | None = None


class RecognitionEventsListResponse(BaseModel):
    items: list[RecognitionEventResponse]
    cursor: float | None = None


class FaceSubjectsResponse(BaseModel):
    subjects: list[str]


class FaceDeleteSubjectResponse(BaseModel):
    subject_id: str
    deleted: bool


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _sha1_bytes_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _uuid5_from_name(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


def _decode_b64_image(image_b64: str) -> np.ndarray:
    try:
        img = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_b64")
    arr = np.frombuffer(img, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    return bgr


def _quality_check_and_embed(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        evaluator = getattr(app.state, "quality", None)
    except Exception:
        evaluator = None

    t0 = _t()
    try:
        emb, meta = _embed_quality_check_and_embed(
            bgr,
            embedder=app.state.embedder,
            evaluator=evaluator,
        )
    except ValueError as e:
        msg = str(e)
        if msg.startswith("quality_reject:"):
            reason = msg.split(":", 1)[1] or "unknown"
            _QCHECK_TOTAL.inc()
            _QREJ_TOTAL.labels(reason=str(reason)).inc()
            try:
                logger.info(json.dumps({
                    "event": "quality_check",
                    "status": "rejected",
                    "reason": str(reason),
                }))
            except Exception:
                pass
            raise HTTPException(status_code=422, detail=msg)

        # no face / no embedding cases surface as 404
        if "no face" in msg.lower():
            try:
                logger.info(json.dumps({
                    "event": "quality_check",
                    "status": "rejected",
                    "reason": "no_face_detected",
                }))
            except Exception:
                pass
            _QCHECK_TOTAL.inc()
            _QREJ_TOTAL.labels(reason="no_face_detected").inc()
        _EMB_FAIL.inc()
        raise HTTPException(status_code=404, detail=msg)
    except RuntimeError:
        _EMB_FAIL.inc()
        raise HTTPException(status_code=500, detail="quality evaluator failure")
    except Exception:
        _EMB_FAIL.inc()
        raise HTTPException(status_code=500, detail="embedder failure")
    finally:
        try:
            _EMB_LAT.observe(max(0.0, _t() - t0))
            logger.info(json.dumps({
                "event": "embedding",
                "model": "buffalo_l",
                "latency_ms": int((max(0.0, _t() - t0)) * 1000.0),
            }))
        except Exception:
            pass

    # keep existing logging for quality meta (if present)
    try:
        q = (meta or {}).get("quality") if isinstance(meta, dict) else None
        if isinstance(q, dict):
            _QCHECK_TOTAL.inc()
            logger.info(json.dumps({
                "event": "quality_check",
                "status": q.get("status"),
                "reason": q.get("reason"),
                "blur": q.get("blur"),
                "brightness": q.get("brightness"),
            }))
    except Exception:
        pass

    return emb, meta


def _decode_b64_bytes(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_b64")


def _now_ts() -> float:
    try:
        return time.time()
    except Exception:
        return float(datetime.now(tz=timezone.utc).timestamp())


def _iso_now() -> str:
    try:
        return datetime.now(tz=timezone.utc).isoformat()
    except Exception:
        return str(int(_now_ts()))


def _record_event(buf: list[float]) -> None:
    try:
        buf.append(_now_ts())
        cutoff = _now_ts() - 24 * 3600.0
        k = 0
        for t in buf:
            if t >= cutoff:
                break
            k += 1
        if k > 0:
            del buf[:k]
    except Exception:
        pass


def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def _save_thumb(bgr: np.ndarray, thumbs_dir: str, image_id: str) -> str:
    try:
        h, w = bgr.shape[:2]
        scale = 256.0 / float(max(h, w)) if max(h, w) > 0 else 1.0
        if scale < 1.0:
            nh = max(2, int(round(h * scale)))
            nw = max(2, int(round(w * scale)))
            thumb = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            thumb = bgr
        rel = f"{image_id}.jpg"
        _ensure_dir(thumbs_dir)
        abs_path = os.path.join(thumbs_dir, rel)
        cv2.imwrite(abs_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "/thumbs/" + rel
    except Exception:
        return ""


def _save_event_image(bgr: np.ndarray, events_dir: str, rel_path: str) -> str:
    try:
        abs_path = os.path.join(str(events_dir), str(rel_path).lstrip("/"))
        _ensure_dir(os.path.dirname(abs_path))
        cv2.imwrite(str(abs_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return "/events/" + str(rel_path).lstrip("/")
    except Exception:
        return ""


def _save_image(bgr: np.ndarray, images_dir: str, subject_id: str, image_id: str) -> str:
    try:
        from pathlib import Path as _Path
        _ensure_dir(images_dir)
        subdir = _Path(images_dir) / str(subject_id)
        subdir.mkdir(parents=True, exist_ok=True)
        abs_path = subdir / f"{image_id}.jpg"
        cv2.imwrite(str(abs_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"/images/{subject_id}/{image_id}.jpg"
    except Exception:
        return ""



def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    return bgr


def _qdrant_enabled() -> bool:
    return bool(os.environ.get("QDRANT_URL"))


def _get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except Exception as e:
        raise RuntimeError("qdrant-client is required") from e
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL not configured")
    return QdrantClient(url=url)


def _ensure_qdrant_collection(client, collection: str, vector_size: int) -> None:
    try:
        from qdrant_client.http.models import Distance, VectorParams
    except Exception as e:
        raise RuntimeError("qdrant-client models unavailable") from e

    try:
        exists = client.collection_exists(collection_name=collection)
    except Exception:
        exists = False
    if exists:
        return

    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
        )
    except Exception as e:
        msg = str(e)
        content = ""
        try:
            raw = getattr(e, "content", None)
            if isinstance(raw, (bytes, bytearray)):
                content = raw.decode("utf-8", errors="ignore")
            elif isinstance(raw, str):
                content = raw
        except Exception:
            content = ""

        hay = (msg + "\n" + content).lower()
        if "already exists" in hay and collection.lower() in hay:
            return
        raise


def _qdrant_search(client, collection: str, emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    t0 = _t()
    try:
        hits = client.search(
            collection_name=collection,
            query_vector=emb.astype(np.float32).reshape(-1).tolist(),
            limit=int(top_k),
            with_payload=True,
        )
    except Exception as e:
        _QDRANT_ERR.inc()
        raise HTTPException(status_code=500, detail=f"qdrant search failed: {str(e)}")
    finally:
        try:
            _QDRANT_SEARCH_LAT.observe(max(0.0, _t() - t0))
        except Exception:
            pass

    out: list[dict[str, Any]] = []
    for h in hits or []:
        try:
            payload = getattr(h, "payload", None) or {}
            out.append(
                {
                    "subject_id": str(payload.get("subject_id") or ""),
                    "similarity": float(getattr(h, "score", 0.0) or 0.0),
                    "point_id": str(getattr(h, "id", "")),
                    "image_id": str(payload.get("image_id") or ""),
                    "thumb_path": str(payload.get("thumb_path") or ""),
                }
            )
        except Exception:
            continue
    return out


def _qdrant_list_subjects(client, collection: str, limit: int = 5000) -> list[str]:
    try:
        points, _ = client.scroll(
            collection_name=collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant scroll failed: {str(e)}")

    s: set[str] = set()
    for p in points or []:
        try:
            payload = getattr(p, "payload", None) or {}
            subject_id = str(payload.get("subject_id") or "").strip()
            if subject_id:
                s.add(subject_id)
        except Exception:
            continue
    return sorted(s)


def _load_face_dataset(face_dir: str, embed_fn) -> FaceIndex:
    if not os.path.isdir(face_dir):
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    subject_ids: list[str] = []
    mean_embeddings: list[np.ndarray] = []

    for name in sorted(os.listdir(face_dir)):
        if name == "train":
            continue
        folder = os.path.join(face_dir, name)
        if not os.path.isdir(folder):
            continue

        embs: list[np.ndarray] = []
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            img = cv2.imread(p)
            if img is None:
                continue
            try:
                emb = embed_fn(img)
            except Exception:
                continue
            embs.append(_l2_normalize(emb))

        if not embs:
            continue

        m = np.mean(np.stack(embs, axis=0), axis=0)
        subject_ids.append(str(name))
        mean_embeddings.append(_l2_normalize(m))

    if not mean_embeddings:
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    mat = np.stack(mean_embeddings, axis=0).astype(np.float32)
    return FaceIndex(subject_ids=subject_ids, mean_embeddings=mat)


def _load_index_json_embeddings(index_path: str) -> FaceIndex:
    if not index_path or not os.path.isfile(index_path):
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    output_dir = data.get("output_dir") if isinstance(data, dict) else None
    output_dir = str(output_dir) if output_dir else None
    index_dir = os.path.dirname(index_path)

    per_label: dict[str, list[np.ndarray]] = {}
    for it in items:
        try:
            if not isinstance(it, dict):
                continue
            if not bool(it.get("success")):
                continue
            label = str(it.get("label") or "").strip()
            emb_rel = str(it.get("embedding") or "").strip()
            if not label or not emb_rel:
                continue
            emb_path = emb_rel
            if not os.path.isabs(emb_path):
                # Prefer index-local resolution so index.json generated on Windows host still works in Docker.
                emb_path = os.path.join(index_dir, emb_path)

            # Back-compat: if index.json contains an output_dir, try it as a fallback.
            if (not os.path.isfile(emb_path)) and output_dir and not os.path.isabs(emb_rel):
                try:
                    candidate = os.path.join(output_dir, emb_rel)
                    if os.path.isfile(candidate):
                        emb_path = candidate
                except Exception:
                    pass
            if not os.path.isfile(emb_path):
                continue
            emb = np.load(emb_path).astype(np.float32).reshape(-1)
            emb = _l2_normalize(emb)
            per_label.setdefault(label, []).append(emb)
        except Exception:
            continue

    if not per_label:
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    subject_ids: list[str] = []
    mean_embeddings: list[np.ndarray] = []
    for label in sorted(per_label.keys()):
        embs = per_label[label]
        if not embs:
            continue
        m = np.mean(np.stack(embs, axis=0), axis=0)
        subject_ids.append(label)
        mean_embeddings.append(_l2_normalize(m))

    if not mean_embeddings:
        return FaceIndex(subject_ids=[], mean_embeddings=np.zeros((0, 1), dtype=np.float32))

    mat = np.stack(mean_embeddings, axis=0).astype(np.float32)
    return FaceIndex(subject_ids=subject_ids, mean_embeddings=mat)


def _infer_tflite(model_path: str, input_data: np.ndarray) -> np.ndarray:
    raise RuntimeError("tflite inference not supported in Buffalo-L only mode")


## BuffaloLEmbedder moved to embedders.buffalo_l


app = FastAPI()

# CORS for UI and external clients
cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if cors_origins_env:
    for o in cors_origins_env.split(","):
        o = o.strip()
        if o:
            cors_origins.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static thumbnails: ensure directory exists before mounting
_thumbs_dir_mount = os.environ.get("THUMBS_DIR", "/data/thumbs")
try:
    os.makedirs(_thumbs_dir_mount, exist_ok=True)
except Exception:
    pass
app.mount(
    "/thumbs",
    StaticFiles(directory=_thumbs_dir_mount),
    name="thumbs",
)

# Static originals: ensure directory exists before mounting
_images_dir_mount = os.environ.get("IMAGES_DIR", "/data/images")
try:
    os.makedirs(_images_dir_mount, exist_ok=True)
except Exception:
    pass
app.mount(
    "/images",
    StaticFiles(directory=_images_dir_mount),
    name="images",
)

# Static recognition event images
_events_dir_mount = os.environ.get("EVENTS_DIR", "/data/events")
try:
    os.makedirs(_events_dir_mount, exist_ok=True)
except Exception:
    pass
app.mount(
    "/events",
    StaticFiles(directory=_events_dir_mount),
    name="events",
)


@app.middleware("http")
async def _metrics_middleware(request, call_next):
    path = request.url.path
    t0 = _t()
    try:
        _REQ_TOTAL.labels(endpoint=path).inc()
    except Exception:
        pass
    response = await call_next(request)
    try:
        _REQ_LAT.labels(endpoint=path).observe(max(0.0, _t() - t0))
    except Exception:
        pass
    return response


@app.get("/metrics")
def metrics() -> Response:
    try:
        body = generate_latest()
    except Exception:
        body = b""
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
def _startup() -> None:
    # Optional config.yaml for defaults (environment variables still take precedence)
    try:
        cfg_path = os.environ.get("CONFIG_PATH", "/app/config.yaml")
        cfg = load_config(cfg_path)
        apply_env_defaults_from_config(cfg)
    except Exception as e:
        logger.error("failed to load config: %s", str(e))

    face_dir = os.environ.get("FACE_DIR", "/media/frigate/face")
    try:
        face_dir = os.path.normpath(str(face_dir))
    except Exception:
        pass

    # Ensure events folders exist for operational visibility
    try:
        events_dir = os.environ.get("EVENTS_DIR", "/data/events")
        _ensure_dir(os.path.join(events_dir, "accepted"))
        _ensure_dir(os.path.join(events_dir, "rejected"))
        _ensure_dir(os.path.join(events_dir, "no_match"))
    except Exception:
        pass

    if _debug_enabled():
        try:
            logging.basicConfig(level=logging.INFO)
        except Exception:
            pass
        _debug("debug_enabled=1")

    model_root = os.environ.get("BUFFALO_MODEL_ROOT", "/models")
    model_name = os.environ.get("BUFFALO_MODEL_NAME", "buffalo_l")
    det_size = int(os.environ.get("BUFFALO_DET_SIZE", "640"))
    min_det_score = _as_float(os.environ.get("BUFFALO_MIN_DET_SCORE", "0.65"), 0.65)
    providers = os.environ.get("BUFFALO_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")

    app.state.embedder = BuffaloLEmbedder(
        model_root=model_root,
        model_name=model_name,
        det_size=det_size,
        min_det_score=min_det_score,
        providers=providers,
    )

    app.state.qdrant_url = os.environ.get("QDRANT_URL")
    app.state.qdrant_collection = os.environ.get("QDRANT_COLLECTION", "frigate_faces")
    app.state.qdrant = None
    if _qdrant_enabled():
        try:
            app.state.qdrant = _get_qdrant_client()
        except Exception as e:
            logger.error("failed to init qdrant client: %s", str(e))
            app.state.qdrant = None

    embeddings_index = os.environ.get("FACE_EMBEDDINGS_INDEX")
    try:
        if embeddings_index:
            embeddings_index = os.path.normpath(str(embeddings_index))
    except Exception:
        pass

    derived_index: str | None = None
    try:
        base = os.path.basename(face_dir.rstrip(os.sep))
        parent = os.path.dirname(face_dir.rstrip(os.sep))
        if base and parent:
            derived_index = os.path.join(parent, f"{base}_embeddings", "index.json")
    except Exception:
        derived_index = None

    index_path: str | None = None
    if embeddings_index and os.path.isfile(embeddings_index):
        index_path = embeddings_index
    elif derived_index and os.path.isfile(derived_index):
        index_path = derived_index

    if index_path:
        app.state.index = _load_index_json_embeddings(index_path)
    else:
        app.state.index = _load_face_dataset(face_dir, app.state.embedder.embed_bgr)

    try:
        idx: FaceIndex = app.state.index
        print(f"[face_service] enrolled_subjects={len(idx.subject_ids)}")
        if len(idx.subject_ids) > 0:
            print(f"[face_service] first_subjects={idx.subject_ids[:10]}")
    except Exception:
        pass

    app.state.min_similarity = _as_float(
        os.environ.get("FACE_SERVICE_MIN_SIMILARITY", "0.25"), 0.25
    )

    if app.state.qdrant is not None:
        try:
            idx: FaceIndex = app.state.index
            if idx.mean_embeddings.size > 0:
                dim = int(idx.mean_embeddings.shape[1])
                _ensure_qdrant_collection(
                    app.state.qdrant, app.state.qdrant_collection, vector_size=dim
                )
        except Exception as e:
            logger.error("failed to ensure qdrant collection: %s", str(e))

    # Initialize thumbs dir and activity counters
    app.state.thumbs_dir = os.environ.get("THUMBS_DIR", "/data/thumbs")
    try:
        _ensure_dir(app.state.thumbs_dir)
    except Exception:
        pass
    app.state.search_events: list[float] = []
    app.state.enroll_events: list[float] = []

    # Quality evaluator
    try:
        app.state.quality = FaceQualityEvaluator()
    except Exception as e:
        logger.error("failed to init quality evaluator: %s", str(e))
        app.state.quality = None

    # Recognition events store
    try:
        db_path = os.environ.get("EVENTS_DB", "/data/events/events.db")
        app.state.events = EventsStore(db_path)
    except Exception as e:
        logger.error("failed to init events store: %s", str(e))
        app.state.events = None


@app.post("/v1/face/search", response_model=FaceSearchResponse)
def face_search(req: FaceSearchRequest) -> FaceSearchResponse:
    if _debug_enabled():
        try:
            _debug(
                f"request camera={req.camera} reid_id={req.reid_id} frame_time={req.frame_time} image_b64_len={len(req.image_b64) if req.image_b64 else 0}"
            )
        except Exception:
            pass
    bgr = _decode_b64_image(req.image_b64)

    try:
        h, w = bgr.shape[:2]
        _debug(f"decoded_image_shape={w}x{h}")
    except Exception:
        pass

    q = getattr(app.state, "qdrant", None)
    if q is not None:
        emb, meta = _quality_check_and_embed(bgr)
        results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1)
        if not results:
            _debug("no_match")
            raise HTTPException(status_code=404, detail="no match")

        best = results[0]
        best_subject = str(best.get("subject_id") or "")
        best_sim = float(best.get("similarity") or 0.0)

        try:
            _debug(
                f"best_candidate subject_id={best_subject} similarity={float(best_sim):.4f} min_similarity={float(app.state.min_similarity):.4f}"
            )
        except Exception:
            pass

        if float(best_sim) < float(app.state.min_similarity):
            _debug(
                f"no_match_above_threshold best_subject={best_subject} best_sim={float(best_sim):.4f} min_similarity={float(app.state.min_similarity):.4f}"
            )
            raise HTTPException(status_code=404, detail="no match above threshold")

        if float(best_sim) < float(app.state.min_similarity):
            logger.info("recognition_below_threshold subject_id=%s score=%.4f min=%.4f", best_subject, float(best_sim), float(app.state.min_similarity))
        _debug(f"match subject_id={best_subject} similarity={float(best_sim):.4f}")
        return FaceSearchResponse(subject_id=best_subject, similarity=float(best_sim), meta=meta)

    idx: FaceIndex = app.state.index
    if idx.mean_embeddings.size == 0 or len(idx.subject_ids) == 0:
        _debug("no_enrolled_faces")
        raise HTTPException(status_code=404, detail="no enrolled faces")

    _debug(f"search_candidates={len(idx.subject_ids)}")

    emb, meta = _quality_check_and_embed(bgr)

    best_i = -1
    best_sim = -1.0
    for i in range(len(idx.subject_ids)):
        sim = _cosine_similarity(emb, idx.mean_embeddings[i])
        if sim > best_sim:
            best_sim = sim
            best_i = i

    if best_i < 0:
        _debug("no_match")
        raise HTTPException(status_code=404, detail="no match")

    try:
        _debug(
            f"best_candidate subject_id={idx.subject_ids[best_i]} similarity={float(best_sim):.4f} min_similarity={float(app.state.min_similarity):.4f}"
        )
    except Exception:
        pass

    if float(best_sim) < float(app.state.min_similarity):
        _debug(
            f"no_match_above_threshold best_subject={idx.subject_ids[best_i]} best_sim={float(best_sim):.4f} min_similarity={float(app.state.min_similarity):.4f}"
        )
        raise HTTPException(status_code=404, detail="no match above threshold")

    _debug(
        f"match subject_id={idx.subject_ids[best_i]} similarity={float(best_sim):.4f}"
    )

    return FaceSearchResponse(subject_id=str(idx.subject_ids[best_i]), similarity=float(best_sim), meta=meta)


@app.post("/v1/faces/add", response_model=FaceAddResponse)
def faces_add(req: FaceAddRequest) -> FaceAddResponse:
    if not req.subject_id or not str(req.subject_id).strip():
        raise HTTPException(status_code=400, detail="subject_id is required")
    if not req.images_b64:
        raise HTTPException(status_code=400, detail="images_b64 must be non-empty")

    subject_id = str(req.subject_id).strip()
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    num_embedded = 0
    emb_dim: int | None = None
    last_meta: dict[str, Any] | None = None

    for i, img_b64 in enumerate(req.images_b64):
        image_bytes = _decode_b64_bytes(img_b64)
        bgr = _decode_image_bytes(image_bytes)
        try:
            emb, meta = _quality_check_and_embed(bgr)
        except HTTPException as e:
            _debug(f"add_skip subject_id={subject_id} idx={i} reason={e.detail}")
            continue
        last_meta = meta
        emb_dim = int(emb.reshape(-1).shape[0])
        try:
            _ensure_qdrant_collection(q, app.state.qdrant_collection, vector_size=emb_dim)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant init failed: {str(e)}")

        # Deterministic IDs: image_hash and point_id
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_id = image_hash[:16]
        point_id = hashlib.sha256(f"{subject_id}:{image_hash}".encode("utf-8")).hexdigest()
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, image_id)
        image_path = _save_image(bgr, os.environ.get("IMAGES_DIR", "/data/images"), subject_id, image_id)

        try:
            from qdrant_client.http.models import PointStruct
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

        t0 = _t()
        try:
            q.upsert(
                collection_name=app.state.qdrant_collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=emb.astype(np.float32).reshape(-1).tolist(),
                        payload={
                            "subject_id": subject_id,
                            "image_id": image_id,
                            "created_at": _iso_now(),
                            "thumb_path": thumb_path,
                            "image_path": image_path,
                            "source": "enroll",
                        },
                    )
                ],
            )
        except Exception as e:
            _QDRANT_ERR.inc()
            raise HTTPException(status_code=500, detail=f"qdrant upsert failed: {str(e)}")
        finally:
            try:
                _QDRANT_UPSERT_LAT.observe(max(0.0, _t() - t0))
            except Exception:
                pass

        num_embedded += 1

    if num_embedded == 0:
        raise HTTPException(status_code=404, detail="no faces embedded from provided images")

    _record_event(app.state.enroll_events)
    return FaceAddResponse(
        subject_id=subject_id,
        num_images=len(req.images_b64),
        num_embedded=num_embedded,
        embedding_dim=emb_dim,
        meta=last_meta,
    )


@app.post("/v1/faces/add_upload", response_model=FaceAddResponse)
async def faces_add_upload(
    subject_id: str = Form(...),
    files: list[UploadFile] = File(...),
) -> FaceAddResponse:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    if not files:
        raise HTTPException(status_code=400, detail="files must be non-empty")

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    num_embedded = 0
    emb_dim: int | None = None

    for i, f in enumerate(files):
        image_bytes = await f.read()
        bgr = _decode_image_bytes(image_bytes)
        try:
            emb, meta = _quality_check_and_embed(bgr)
        except HTTPException as e:
            _debug(f"add_skip subject_id={subject_id} idx={i} reason={e.detail}")
            continue
        last_meta = meta
        emb_dim = int(emb.reshape(-1).shape[0])
        try:
            _ensure_qdrant_collection(q, app.state.qdrant_collection, vector_size=emb_dim)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant init failed: {str(e)}")

        # Deterministic IDs: image_hash and point_id
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_id = image_hash[:16]
        point_id = hashlib.sha256(f"{subject_id}:{image_hash}".encode("utf-8")).hexdigest()
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, image_id)
        image_path = _save_image(bgr, os.environ.get("IMAGES_DIR", "/data/images"), subject_id, image_id)
        try:
            from qdrant_client.http.models import PointStruct
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

        t0 = _t()
        try:
            q.upsert(
                collection_name=app.state.qdrant_collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=emb.astype(np.float32).reshape(-1).tolist(),
                        payload={
                            "subject_id": subject_id,
                            "image_id": image_id,
                            "created_at": _iso_now(),
                            "thumb_path": thumb_path,
                            "source": "enroll",
                            "filename": str(getattr(f, "filename", "") or ""),
                            "image_path": image_path,
                        },
                    )
                ],
            )
        except Exception as e:
            _QDRANT_ERR.inc()
            raise HTTPException(status_code=500, detail=f"qdrant upsert failed: {str(e)}")
        finally:
            try:
                _QDRANT_UPSERT_LAT.observe(max(0.0, _t() - t0))
            except Exception:
                pass

        num_embedded += 1

    if num_embedded == 0:
        raise HTTPException(status_code=404, detail="no faces embedded from provided images")

    _record_event(app.state.enroll_events)
    return FaceAddResponse(
        subject_id=subject_id,
        num_images=len(files),
        num_embedded=num_embedded,
        embedding_dim=emb_dim,
        meta=last_meta,
    )


@app.post("/v1/faces/search", response_model=FaceSearchTopKResponse)
def faces_search(req: FaceSearchTopKRequest) -> FaceSearchTopKResponse:
    top_k = int(req.top_k or 5)
    top_k = max(1, min(top_k, 50))

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    bgr = _decode_b64_image(req.image_b64)
    emb, _meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
    _record_event(app.state.search_events)
    query_image_id = f"tmp-{uuid.uuid4()}"
    query_thumb_path = _save_thumb(bgr, app.state.thumbs_dir, query_image_id)
    _record_event(app.state.search_events)
    return FaceSearchTopKResponse(results=[FaceSearchTopKItem(**r) for r in results], query_thumb_path=query_thumb_path or None)


@app.post("/v1/faces/search_upload", response_model=FaceSearchTopKResponse)
async def faces_search_upload(
    file: UploadFile = File(...),
    top_k: int = Form(5),
) -> FaceSearchTopKResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    top_k = int(top_k or 5)
    top_k = max(1, min(top_k, 50))
    image_bytes = await file.read()
    bgr = _decode_image_bytes(image_bytes)
    emb, _meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
    _record_event(app.state.search_events)
    query_image_id = f"tmp-{uuid.uuid4()}"
    query_thumb_path = _save_thumb(bgr, app.state.thumbs_dir, query_image_id)
    _record_event(app.state.search_events)
    return FaceSearchTopKResponse(results=[FaceSearchTopKItem(**r) for r in results], query_thumb_path=query_thumb_path or None)


@app.post("/v1/faces/recognize", response_model=FaceRecognizeResponse)
def faces_recognize(req: FaceRecognizeRequest) -> FaceRecognizeResponse:
    top_k = int(req.top_k or 5)
    top_k = max(1, min(top_k, 50))
    min_sim = float(req.min_similarity) if req.min_similarity is not None else float(app.state.min_similarity)

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    bgr = _decode_b64_image(req.image_b64)
    emb, meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
    items = [FaceSearchTopKItem(**r) for r in results]

    if not items:
        return FaceRecognizeResponse(matched=False, results=[], meta=meta)

    best = items[0]
    if float(best.similarity) >= float(min_sim) and str(best.subject_id).strip():
        ok, second, margin, req = _passes_top2_margin(results, float(best.similarity))
        try:
            meta = dict(meta or {})
            meta["decision"] = {
                "status": "match" if ok else "no_match",
                "min_similarity": float(min_sim),
                "top2_second": second,
                "top2_margin": margin,
                "top2_required": req,
            }
        except Exception:
            pass
        if not ok:
            return FaceRecognizeResponse(matched=False, results=items, meta=meta)
        return FaceRecognizeResponse(
            matched=True,
            subject_id=best.subject_id,
            similarity=float(best.similarity),
            results=items,
            meta=meta,
        )
    return FaceRecognizeResponse(matched=False, results=items, meta=meta)


@app.post("/v1/events/recognition", response_model=RecognitionEventResponse)
async def ingest_recognition_event(
    file: UploadFile = File(...),
    camera: str = Form(...),
    source_path: str = Form(""),
    ts: float | None = Form(None),
    top_k: int = Form(5),
    min_similarity: float | None = Form(None),
    process_all_faces: bool = Form(False),
) -> RecognitionEventResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    camera = str(camera or "").strip()
    if not camera:
        raise HTTPException(status_code=400, detail="camera is required")

    image_bytes = await file.read()
    bgr = _decode_image_bytes(image_bytes)
    h, w = bgr.shape[:2]

    event_id = str(uuid.uuid4())
    ts_val = float(ts) if ts is not None else _now_ts()
    source_path = str(source_path or str(getattr(file, "filename", "") or "")).strip()

    events_dir = os.environ.get("EVENTS_DIR", "/data/events")

    def _face_bbox_for_meta(face: Any) -> tuple[list[float] | None, float | None]:
        bbox: list[float] | None = None
        det_score: float | None = None
        try:
            b = np.asarray(getattr(face, "bbox", None), dtype=np.float32).reshape(-1)
            if b.size >= 4:
                bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(w), float(h)]
        except Exception:
            bbox = None
        try:
            det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        except Exception:
            det_score = None
        return bbox, det_score

    def _embed_from_face(face: Any) -> np.ndarray | None:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        return _l2_normalize(np.asarray(emb, dtype=np.float32))

    # detect faces
    faces: list[Any] = []
    try:
        if process_all_faces:
            faces = list(app.state.embedder.detect_all(bgr))
        else:
            faces = [app.state.embedder.detect_best(bgr)]
    except ValueError:
        # no faces at all
        img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{event_id}.jpg")
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{event_id}")
        image_saved_at = _now_ts() if img_path else None
        reason = "no_face_detected"
        meta = {
            "quality": {"status": "rejected", "reason": reason},
            "decision": {"status": "rejected"},
            "faces_total": 0,
            "faces_processed": 0,
            "multi_face": bool(process_all_faces),
        }
        store.insert_event(
            RecognitionEvent(
                event_id=event_id,
                ts=ts_val,
                camera=camera,
                source_path=source_path,
                decision="rejected",
                subject_id=None,
                similarity=None,
                rejected_reason=reason,
                bbox=None,
                det_score=None,
                image_path=img_path,
                thumb_path=thumb_path,
                image_saved_at=image_saved_at,
                meta=meta,
            )
        )
        return RecognitionEventResponse(
            event_id=event_id,
            ts=ts_val,
            camera=camera,
            source_path=source_path,
            decision="rejected",
            subject_id=None,
            similarity=None,
            rejected_reason=reason,
            bbox=None,
            det_score=None,
            image_path=img_path,
            thumb_path=thumb_path,
            image_saved_at=image_saved_at,
            meta=meta,
        )

    evaluator = getattr(app.state, "quality", None)
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    top_k = max(1, min(int(top_k or 5), 50))
    min_sim = float(min_similarity) if min_similarity is not None else float(app.state.min_similarity)
    max_faces = int(os.environ.get("FACE_SERVICE_MAX_FACES_PER_IMAGE", "5") or "5")
    max_faces = max(1, min(max_faces, 20))

    # process highest-confidence faces first
    def _score(face: Any) -> float:
        try:
            return float(getattr(face, "det_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    faces_sorted = sorted([f for f in faces if f is not None], key=_score, reverse=True)[:max_faces]
    faces_total = len(faces)

    primary_resp: RecognitionEventResponse | None = None
    faces_processed = 0
    for idx, face in enumerate(faces_sorted):
        ev_id = str(uuid.uuid4())
        bbox, det_score = _face_bbox_for_meta(face)

        quality_meta: dict[str, Any] | None = None
        if evaluator is not None:
            try:
                quality_meta = evaluator.evaluate(bgr, face)
            except Exception:
                quality_meta = {"status": "rejected", "reason": "quality_eval_failed"}
            if isinstance(quality_meta, dict) and quality_meta.get("status") == "rejected":
                reason = str(quality_meta.get("reason") or "unknown")
                img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{ev_id}.jpg")
                thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
                image_saved_at = _now_ts() if img_path else None
                meta = {
                    "quality": quality_meta,
                    "decision": {"status": "rejected"},
                    "face_index": int(idx),
                    "faces_total": int(faces_total),
                    "faces_processed": None,
                    "multi_face": bool(process_all_faces),
                }
                store.insert_event(
                    RecognitionEvent(
                        event_id=ev_id,
                        ts=ts_val,
                        camera=camera,
                        source_path=source_path,
                        decision="rejected",
                        subject_id=None,
                        similarity=None,
                        rejected_reason=reason,
                        bbox=bbox,
                        det_score=det_score,
                        image_path=img_path,
                        thumb_path=thumb_path,
                        image_saved_at=image_saved_at,
                        meta=meta,
                    )
                )
                faces_processed += 1
                if primary_resp is None:
                    primary_resp = RecognitionEventResponse(
                        event_id=ev_id,
                        ts=ts_val,
                        camera=camera,
                        source_path=source_path,
                        decision="rejected",
                        subject_id=None,
                        similarity=None,
                        rejected_reason=reason,
                        bbox=bbox,
                        det_score=det_score,
                        image_path=img_path,
                        thumb_path=thumb_path,
                        image_saved_at=image_saved_at,
                        meta=meta,
                    )
                continue

        emb = _embed_from_face(face)
        if emb is None:
            reason = "no_embedding"
            img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{ev_id}.jpg")
            thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
            image_saved_at = _now_ts() if img_path else None
            meta = {
                "quality": quality_meta,
                "decision": {"status": "rejected"},
                "face_index": int(idx),
                "faces_total": int(faces_total),
                "faces_processed": None,
                "multi_face": bool(process_all_faces),
            }
            store.insert_event(
                RecognitionEvent(
                    event_id=ev_id,
                    ts=ts_val,
                    camera=camera,
                    source_path=source_path,
                    decision="rejected",
                    subject_id=None,
                    similarity=None,
                    rejected_reason=reason,
                    bbox=bbox,
                    det_score=det_score,
                    image_path=img_path,
                    thumb_path=thumb_path,
                    image_saved_at=image_saved_at,
                    meta=meta,
                )
            )
            faces_processed += 1
            if primary_resp is None:
                primary_resp = RecognitionEventResponse(
                    event_id=ev_id,
                    ts=ts_val,
                    camera=camera,
                    source_path=source_path,
                    decision="rejected",
                    subject_id=None,
                    similarity=None,
                    rejected_reason=reason,
                    bbox=bbox,
                    det_score=det_score,
                    image_path=img_path,
                    thumb_path=thumb_path,
                    image_saved_at=image_saved_at,
                    meta=meta,
                )
            continue

        results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
        matched = False
        subject_id: str | None = None
        similarity: float | None = None
        top2_second: float | None = None
        top2_margin: float | None = None
        top2_required: float | None = None
        if results:
            try:
                best = results[0]
                similarity = float(best.get("similarity") or 0.0)
                sid = str(best.get("subject_id") or "").strip()
                if sid and similarity >= float(min_sim):
                    ok, second, margin, req = _passes_top2_margin(results, similarity)
                    top2_second, top2_margin, top2_required = second, margin, req
                    if ok:
                        matched = True
                        subject_id = sid
            except Exception:
                matched = False

        decision = "match" if matched else "no_match"
        img_path = _save_event_image(
            bgr,
            events_dir,
            f"{'accepted' if matched else 'no_match'}/{camera}/{ev_id}.jpg",
        )
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
        image_saved_at = _now_ts() if img_path else None

        meta = {
            "quality": quality_meta,
            "decision": {
                "status": decision,
                "min_similarity": float(min_sim),
                "top2_second": top2_second,
                "top2_margin": top2_margin,
                "top2_required": top2_required,
            },
            "top_k": int(top_k),
            "face_index": int(idx),
            "faces_total": int(faces_total),
            "faces_processed": None,
            "multi_face": bool(process_all_faces),
        }

        store.insert_event(
            RecognitionEvent(
                event_id=ev_id,
                ts=ts_val,
                camera=camera,
                source_path=source_path,
                decision=decision,
                subject_id=subject_id,
                similarity=similarity,
                rejected_reason=None,
                bbox=bbox,
                det_score=det_score,
                image_path=img_path,
                thumb_path=thumb_path,
                image_saved_at=image_saved_at,
                meta=meta,
            )
        )

        faces_processed += 1
        if primary_resp is None:
            primary_resp = RecognitionEventResponse(
                event_id=ev_id,
                ts=ts_val,
                camera=camera,
                source_path=source_path,
                decision=decision,
                subject_id=subject_id,
                similarity=similarity,
                rejected_reason=None,
                bbox=bbox,
                det_score=det_score,
                image_path=img_path,
                thumb_path=thumb_path,
                image_saved_at=image_saved_at,
                meta=meta,
            )

    # Patch faces_processed into primary event meta for audit
    if primary_resp is None:
        raise HTTPException(status_code=500, detail="failed to process faces")
    try:
        meta = dict(primary_resp.meta or {})
        meta["faces_total"] = int(faces_total)
        meta["faces_processed"] = int(faces_processed)
        meta["multi_face"] = bool(process_all_faces)
        primary_resp.meta = meta
    except Exception:
        pass
    return primary_resp


@app.get("/v1/events/recognition", response_model=RecognitionEventsListResponse)
def list_recognition_events(
    camera: str | None = None,
    subject_id: str | None = None,
    decision: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    limit: int = 100,
    cursor: float | None = None,
) -> RecognitionEventsListResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")
    items, next_cur = store.list_events(
        camera=camera,
        subject_id=subject_id,
        decision=decision,
        since_ts=since_ts,
        until_ts=until_ts,
        limit=limit,
        cursor_ts=cursor,
    )
    return RecognitionEventsListResponse(
        items=[RecognitionEventResponse(**it) for it in items],
        cursor=next_cur,
    )


@app.get("/v1/events/recognition/{event_id}", response_model=RecognitionEventResponse)
def get_recognition_event(event_id: str) -> RecognitionEventResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")
    it = store.get_event(event_id)
    if not it:
        raise HTTPException(status_code=404, detail="event not found")
    return RecognitionEventResponse(**it)


@app.post("/v1/faces/recognize_upload", response_model=FaceRecognizeResponse)
async def faces_recognize_upload(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    min_similarity: float | None = Form(None),
) -> FaceRecognizeResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    top_k = int(top_k or 5)
    top_k = max(1, min(top_k, 50))
    min_sim = float(min_similarity) if min_similarity is not None else float(app.state.min_similarity)

    image_bytes = await file.read()
    bgr = _decode_image_bytes(image_bytes)
    emb, meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
    items = [FaceSearchTopKItem(**r) for r in results]
    if not items:
        return FaceRecognizeResponse(matched=False, results=[], meta=meta)

    best = items[0]
    if float(best.similarity) >= float(min_sim) and str(best.subject_id).strip():
        return FaceRecognizeResponse(
            matched=True,
            subject_id=best.subject_id,
            similarity=float(best.similarity),
            results=items,
            meta=meta,
        )
    return FaceRecognizeResponse(matched=False, results=items, meta=meta)


@app.post("/v1/face/search_upload", response_model=FaceSearchResponse)
async def face_search_upload(file: UploadFile = File(...)) -> FaceSearchResponse:
    image_bytes = await file.read()
    bgr = _decode_image_bytes(image_bytes)
    req = FaceSearchRequest(image_b64="")
    return face_search(req.model_copy(update={"image_b64": base64.b64encode(image_bytes).decode("ascii")}))


@app.get("/ui")
def ui() -> Response:
    return Response(content=ui_html(), media_type="text/html")


@app.get("/v1/faces/subjects", response_model=FaceSubjectsResponse)
def faces_subjects() -> FaceSubjectsResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    subjects = _qdrant_list_subjects(q, app.state.qdrant_collection)
    return FaceSubjectsResponse(subjects=subjects)


@app.delete("/v1/faces/subjects/{subject_id}", response_model=FaceDeleteSubjectResponse)
def faces_delete_subject(subject_id: str) -> FaceDeleteSubjectResponse:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

    try:
        q.delete(
            collection_name=app.state.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="subject_id",
                        match=MatchValue(value=subject_id),
                    )
                ]
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant delete failed: {str(e)}")

    return FaceDeleteSubjectResponse(subject_id=subject_id, deleted=True)


@app.get("/v1/stats")
def stats() -> dict[str, Any]:
    q = getattr(app.state, "qdrant", None)
    enabled = q is not None
    collection = getattr(app.state, "qdrant_collection", None)
    subjects_total = 0
    embeddings_total = 0
    if enabled:
        try:
            cnt = q.count(collection_name=collection, exact=True)
            embeddings_total = int(getattr(cnt, "count", 0) or 0)
        except Exception:
            embeddings_total = 0
        try:
            subjects_total = len(_qdrant_list_subjects(q, collection))
        except Exception:
            subjects_total = 0

    cutoff = _now_ts() - 24 * 3600.0
    try:
        app.state.search_events = [t for t in app.state.search_events if t >= cutoff]
    except Exception:
        app.state.search_events = []
    try:
        app.state.enroll_events = [t for t in app.state.enroll_events if t >= cutoff]
    except Exception:
        app.state.enroll_events = []

    return {
        "subjects_total": subjects_total,
        "embeddings_total": embeddings_total,
        "last_24h_enrolls": len(app.state.enroll_events or []),
        "last_24h_searches": len(app.state.search_events or []),
        "qdrant_enabled": enabled,
        "qdrant_collection": collection,
    }



class SubjectItem(BaseModel):
    subject_id: str
    embeddings_count: int


class SubjectsListResponse(BaseModel):
    items: list[SubjectItem]
    cursor: str | None = None


class SubjectImageItem(BaseModel):
    image_id: str
    thumb_path: str | None = None
    image_path: str | None = None
    created_at: str | None = None
    source: str | None = None


class SubjectImagesResponse(BaseModel):
    items: list[SubjectImageItem]
    cursor: str | None = None


@app.get("/v1/subjects", response_model=SubjectsListResponse)
def list_subjects(cursor: str | None = None, limit: int = 50, with_counts: bool = True) -> SubjectsListResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    limit = max(1, min(int(limit or 50), 500))
    try:
        scroll_kwargs: dict[str, Any] = {
            "collection_name": app.state.qdrant_collection,
            "limit": int(limit),
            "with_payload": True,
            "with_vectors": False,
        }
        if cursor:
            scroll_kwargs["offset"] = cursor
        points, next_cur = q.scroll(**scroll_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant scroll failed: {str(e)}")

    uniq: dict[str, None] = {}
    for pnt in points or []:
        try:
            payload = getattr(pnt, "payload", None) or {}
            sid = str(payload.get("subject_id") or '').strip()
            if sid:
                uniq.setdefault(sid, None)
        except Exception:
            continue

    items: list[SubjectItem] = []
    if with_counts:
        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")
        for sid in uniq.keys():
            try:
                cnt = q.count(
                    collection_name=app.state.qdrant_collection,
                    exact=True,
                    count_filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=sid))]),
                )
                n = int(getattr(cnt, "count", 0) or 0)
            except Exception:
                n = 0
            items.append(SubjectItem(subject_id=sid, embeddings_count=n))
    else:
        items = [SubjectItem(subject_id=sid, embeddings_count=0) for sid in uniq.keys()]

    next_cursor = str(next_cur) if next_cur is not None else None
    return SubjectsListResponse(items=items, cursor=next_cursor)


@app.get("/v1/subjects/{subject_id}/images", response_model=SubjectImagesResponse)
def list_subject_images(subject_id: str, cursor: str | None = None, limit: int = 50) -> SubjectImagesResponse:
    subject_id = str(subject_id or '').strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    limit = max(1, min(int(limit or 50), 500))
    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

    try:
        points, next_cur = q.scroll(
            collection_name=app.state.qdrant_collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            offset=cursor,
            scroll_filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]),
        )
    except TypeError:
        # older qdrant_client versions use 'filter' parameter name
        points, next_cur = q.scroll(
            collection_name=app.state.qdrant_collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            offset=cursor,
            filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant scroll failed: {str(e)}")

    items: list[SubjectImageItem] = []
    for pnt in points or []:
        try:
            payload = getattr(pnt, "payload", None) or {}
            items.append(SubjectImageItem(
                image_id=str(payload.get("image_id") or ''),
                thumb_path=str(payload.get("thumb_path") or '') or None,
                image_path=str(payload.get("image_path") or '') or None,
                created_at=str(payload.get("created_at") or '') or None,
                source=str(payload.get("source") or '') or None,
            ))
        except Exception:
            continue

    next_cursor = str(next_cur) if next_cur is not None else None
    return SubjectImagesResponse(items=items, cursor=next_cursor)
@app.get("/health")
def health() -> dict[str, Any]:
    q = getattr(app.state, "qdrant", None)
    n = 0
    if q is not None:
        try:
            n = len(_qdrant_list_subjects(q, getattr(app.state, "qdrant_collection", None)))
        except Exception:
            n = 0
    return {
        "ok": True,
        "subjects": n,
        "qdrant_enabled": q is not None,
        "qdrant_collection": getattr(app.state, "qdrant_collection", None),
    }
