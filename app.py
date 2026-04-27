import base64
import hashlib
import json
import logging
import os
import ipaddress
import socket
import time
import uuid
import asyncio
import collections
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone, date, timedelta
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from quality import FaceQualityEvaluator
from embedders.buffalo_l import (
    BuffaloLEmbedder,
    _l2_normalize,
    _quality_check_and_embed as _embed_quality_check_and_embed,
)
from inference_manager import GPUInferenceManager
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from ui_page import ui_html
from events_store import EventsStore, RecognitionEvent
from config_loader import apply_env_defaults_from_config, load_config

from cross_check import cross_check_router

import httpx

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = logging.getLogger("uvicorn.error")


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


# In-memory cache for recently enrolled embeddings to prevent race conditions
# Stores (embedding_hash, subject_id, timestamp)
_RECENT_ENROLLMENTS = collections.deque(maxlen=100)
_RECENT_ENROLL_LOCK = asyncio.Lock()

def _get_recent_enrollment_match(emb: np.ndarray, threshold: float) -> str | None:
    now = time.time()
    # Clean up old entries (older than 5 seconds)
    while _RECENT_ENROLLMENTS and (now - _RECENT_ENROLLMENTS[0][2] > 5.0):
        _RECENT_ENROLLMENTS.popleft()
    
    for cached_emb, sid, ts in _RECENT_ENROLLMENTS:
        # Simple cosine similarity (embeddings are already normalized)
        sim = float(np.dot(emb, cached_emb))
        if sim >= threshold:
            return sid
    return None

def _add_recent_enrollment(emb: np.ndarray, sid: str):
    _RECENT_ENROLLMENTS.append((emb.copy(), sid, time.time()))

def _tz() -> Any:
    name = str(os.environ.get("FACE_SERVICE_TIMEZONE", "Asia/Kolkata") or "Asia/Kolkata").strip() or "Asia/Kolkata"
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _day_range_ts(day_str: str) -> tuple[float | None, float | None]:
    s = str(day_str or "").strip()
    if not s:
        return None, None
    try:
        d = date.fromisoformat(s)
    except Exception:
        return None, None
    tz = _tz()
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz).timestamp()
    end = (datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz) + timedelta(days=1)).timestamp()
    return float(start), float(end)


def _date_window_from_params(
    *,
    day: str | None,
    from_day: str | None,
    to_day: str | None,
    since_ts: float | None,
    until_ts: float | None,
) -> tuple[float | None, float | None]:
    if since_ts is not None or until_ts is not None:
        return since_ts, until_ts

    if day:
        return _day_range_ts(day)

    s0 = None
    e0 = None
    if from_day:
        s0, _ = _day_range_ts(from_day)
    if to_day:
        _, e0 = _day_range_ts(to_day)
    return s0, e0


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


def _pc() -> float:
    try:
        return time.perf_counter()
    except Exception:
        return float(_t())


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
    image_b64: str = Field(..., validation_alias=AliasChoices("image_b64", "image", "images_b64"))
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


class FaceQualityResult(BaseModel):
    ok: bool
    quality: dict[str, Any] | None = None
    det_score: float | None = None
    bbox: list[float] | None = None

class QualityCheckResponse(BaseModel):
    ok: bool
    total_quality: str | None = None
    faces: list[FaceQualityResult] = []
    annotated_image: str | None = None
    timing: dict[str, Any] | None = None


class RecognitionEventResponse(BaseModel):
    event_id: str
    ts: float
    camera: str
    source_path: str
    decision: str
    subject_id: str | None = None
    similarity: float | None = None
    processing_ms: int | None = None
    model_ms: int | None = None
    rejected_reason: str | None = None
    bbox: list[float] | None = None
    det_score: float | None = None
    image_path: str
    thumb_path: str
    image_saved_at: float | None = None
    meta: dict[str, Any] | None = None
    feedback_label: str | None = None
    feedback_note: str | None = None
    feedback_updated_at: float | None = None


class RecognitionEventsListResponse(BaseModel):
    items: list[RecognitionEventResponse]
    cursor: float | None = None


class SearchEventResponse(BaseModel):
    event_id: str
    ts: float
    query_image_path: str
    query_thumb_path: str
    top_subject_id: str | None = None
    top_similarity: float | None = None
    results: list[FaceSearchTopKItem] = []
    meta: dict[str, Any] | None = None


class SearchEventsListResponse(BaseModel):
    items: list[SearchEventResponse]
    cursor: float | None = None


class SearchEventsStatsResponse(BaseModel):
    match: int
    no_match: int
    total: int


class RecognitionStatsResponse(BaseModel):
    total: int
    match: int
    no_match: int
    rejection: int
    unique_matches: int = 0
    by_camera: dict[str, dict[str, int]]


class EventFeedbackRequest(BaseModel):
    label: str | None = None
    note: str | None = None


class EventFeedbackResponse(BaseModel):
    event_id: str
    updated: bool
    feedback_label: str | None = None
    feedback_note: str | None = None
    feedback_updated_at: float | None = None


class FeedbackStatsResponse(BaseModel):
    total: int
    labeled: int
    unlabeled: int
    tp: int
    fp: int
    fn: int
    ignore: int
    fp_rate_match: float | None = None
    by_decision: dict[str, dict[str, int]]


class RecognitionFetchRequest(BaseModel):
    url: str
    camera: str
    source_path: str | None = None
    ts: float | None = None
    top_k: int = 5
    min_similarity: float | None = None
    process_all_faces: bool = False


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
        raise HTTPException(status_code=400, detail="invalid base64")
    arr = np.frombuffer(img, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    return bgr


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("unable to decode image")
        return bgr
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="unable to decode image")


def _quality_check_and_embed(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        evaluator = getattr(app.state, "quality", None)
    except Exception:
        evaluator = None

    try:
        embedder = getattr(app.state, "gpu", None) or app.state.embedder
    except Exception:
        embedder = app.state.embedder

    t0 = _t()
    try:
        emb, meta = _embed_quality_check_and_embed(
            bgr,
            embedder=embedder,
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

    return emb, meta


def _quality_check_all(bgr: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    try:
        evaluator = getattr(app.state, "quality", None)
    except Exception:
        evaluator = None

    try:
        embedder = getattr(app.state, "gpu", None) or app.state.embedder
    except Exception:
        embedder = app.state.embedder

    t0 = _t()
    try:
        # Use detect_all to get all faces
        faces = embedder.detect_all(bgr)
        results = []
        
        annotated = bgr.copy()
        
        for i, face in enumerate(faces):
            det_score = float(getattr(face, "det_score", 0.0) or 0.0)
            bbox_arr = np.asarray(getattr(face, "bbox", None), dtype=np.float32).reshape(-1)
            bbox = [float(x) for x in bbox_arr.tolist()] if bbox_arr.size == 4 else None

            q = None
            face_ok = True
            if evaluator is not None and face is not None:
                try:
                    q = evaluator.evaluate(bgr, face)
                    if isinstance(q, dict) and q.get("status") == "rejected":
                        face_ok = False
                except Exception:
                    pass

            results.append({
                "ok": face_ok,
                "quality": q,
                "det_score": det_score,
                "bbox": bbox
            })
            
            # Annotate image
            if bbox:
                x1, y1, x2, y2 = [int(x) for x in bbox]
                color = (0, 255, 0) if face_ok else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{i} {'PASS' if face_ok else 'FAIL'}"
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        timing = {"detect_ms": int(max(0.0, (_t() - t0)) * 1000.0)}
        return results, {"timing": timing}, annotated
    except Exception as e:
        logger.error(f"detect failure: {str(e)}")
        raise HTTPException(status_code=500, detail="detect failure")



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


def _quarantine_enroll_possible_match(
    bgr: np.ndarray,
    subject_id: str,
    image_id: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        events_dir = os.environ.get("EVENTS_DIR", "/data/events")
        rel = f"no_match/enroll/{str(subject_id).strip()}/{str(image_id).strip()}.jpg"
        img_path = _save_event_image(bgr, events_dir, rel)
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"enroll-{image_id}")
    except Exception:
        img_path = ""
        thumb_path = ""

    out: dict[str, Any] = {
        "status": "no_match",
        "reason": str(reason),
        "subject_id": str(subject_id),
        "image_id": str(image_id),
        "image_path": img_path,
        "thumb_path": thumb_path,
    }
    if extra:
        try:
            out.update(extra)
        except Exception:
            pass
    return out


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



def _save_search_query_assets(bgr: np.ndarray, image_id: str) -> tuple[str, str]:
    try:
        from pathlib import Path as _Path
        _ensure_dir(app.state.search_events_dir)
        _ensure_dir(app.state.search_thumbs_dir)
        
        # Save main query image
        img_path = _Path(app.state.search_events_dir) / f"{image_id}.jpg"
        cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Save thumbnail
        thumb_path = _save_thumb(bgr, app.state.search_thumbs_dir, image_id)
        
        return f"/v1/search_history/asset/image/{image_id}", thumb_path
    except Exception as e:
        logger.error("failed to save search query assets: %s", str(e))
        return "", ""
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
        try:
            info = client.get_collection(collection_name=collection)
            cfg = getattr(info, "config", None)
            params = getattr(cfg, "params", None) if cfg is not None else None
            vectors = getattr(params, "vectors", None) if params is not None else None

            existing_size: int | None = None
            if vectors is not None:
                if hasattr(vectors, "size"):
                    existing_size = int(getattr(vectors, "size"))
                elif isinstance(vectors, dict) and vectors:
                    v0 = next(iter(vectors.values()))
                    if hasattr(v0, "size"):
                        existing_size = int(getattr(v0, "size"))

            if existing_size is not None and int(existing_size) != int(vector_size):
                raise RuntimeError(
                    f"qdrant collection '{collection}' vector_size mismatch: existing={existing_size} expected={int(vector_size)}"
                )
        except RuntimeError:
            raise

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


def _subject_embedding_cap() -> int:
    try:
        return max(1, int(os.environ.get("SUBJECT_MAX_EMBEDDINGS", "10") or "10"))
    except Exception:
        return 10


def _qdrant_count_subject_embeddings(client, collection: str, subject_id: str) -> int:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        return 0
    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    except Exception:
        return 0
    try:
        cnt = client.count(
            collection_name=collection,
            exact=True,
            count_filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]),
        )
        return int(getattr(cnt, "count", 0) or 0)
    except Exception:
        return 0


def _auto_add_enabled() -> bool:
    return str(os.environ.get("AUTO_ADD_EMBEDDING_ENABLE", "0") or "0").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def _auto_add_min_similarity() -> float:
    try:
        return float(os.environ.get("AUTO_ADD_EMBEDDING_MIN_SIM", "0.95") or "0.95")
    except Exception:
        return 0.95


def _enroll_dup_check_enabled() -> bool:
    return str(os.environ.get("ENROLL_DUPLICATE_CHECK_ENABLE", "1") or "1").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def _enroll_dup_min_similarity() -> float:
    try:
        return float(os.environ.get("ENROLL_DUPLICATE_MIN_SIM", "0.80") or "0.55")
    except Exception:
        return 0.55


def _no_match_auto_enroll_enabled() -> bool:
    return str(os.environ.get("NO_MATCH_AUTO_ENROLL_ENABLE", "0") or "0").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )


def _no_match_auto_enroll_prefix() -> str:
    try:
        p = str(os.environ.get("NO_MATCH_AUTO_ENROLL_PREFIX", "unknown") or "unknown").strip()
        return p if p else "unknown"
    except Exception:
        return "unknown"


def _no_match_auto_enroll_block_min_similarity() -> float:
    try:
        return float(os.environ.get("NO_MATCH_AUTO_ENROLL_BLOCK_MIN_SIM", "0.80") or "0.80")
    except Exception:
        return 0.80


def _no_match_auto_attach_min_similarity() -> float:
    try:
        return float(os.environ.get("NO_MATCH_AUTO_ATTACH_MIN_SIM", "0.70") or "0.70")
    except Exception:
        return 0.70


def _no_match_auto_attach_min_margin() -> float:
    try:
        return float(os.environ.get("NO_MATCH_AUTO_ATTACH_MIN_MARGIN", "0.10") or "0.10")
    except Exception:
        return 0.10


def _qdrant_search(client, collection: str, emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    t0 = _t()
    try:
        search_params = None
        try:
            from qdrant_client.http.models import SearchParams
        except Exception:
            SearchParams = None  # type: ignore

        if SearchParams is not None:
            try:
                ef_raw = str(os.environ.get("QDRANT_HNSW_EF", "") or "").strip()
                ef = int(ef_raw) if ef_raw else None
            except Exception:
                ef = None

            exact_raw = str(os.environ.get("QDRANT_EXACT", "") or "").strip().lower()
            exact = exact_raw in ("1", "true", "yes") if exact_raw else None

            indexed_only_raw = str(os.environ.get("QDRANT_INDEXED_ONLY", "") or "").strip().lower()
            indexed_only = indexed_only_raw in ("1", "true", "yes") if indexed_only_raw else None

            rescore_raw = str(os.environ.get("QDRANT_QUANTIZATION_RESCORE", "1")).strip().lower()
            rescore = rescore_raw in ("1", "true", "yes")

            try:
                from qdrant_client.http.models import QuantizationSearchParams
                quant_params = QuantizationSearchParams(rescore=rescore)
            except Exception:
                quant_params = None

            try:
                search_params = SearchParams(
                    hnsw_ef=ef, 
                    exact=exact, 
                    indexed_only=indexed_only,
                    quantization=quant_params
                )
            except Exception:
                search_params = None

        kwargs = dict(
            collection_name=collection,
            query_vector=emb.astype(np.float32).reshape(-1).tolist(),
            limit=int(top_k),
            with_payload=["subject_id", "image_id", "thumb_path"],
        )
        if search_params is not None:
            kwargs["search_params"] = search_params

        try:
            hits = client.search(**kwargs)
        except TypeError:
            # older qdrant_client versions may not support search_params
            kwargs.pop("search_params", None)
            hits = client.search(**kwargs)
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

app.include_router(cross_check_router)

# CORS for UI and external clients
cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*")
if cors_origins_raw == "*" or not cors_origins_raw.strip():
    cors_origins = ["*"]
else:
    # Explicitly include common variants and ensure the provided origins are present
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    if "https://face.service.tools.thefusionapps.com" not in cors_origins:
        cors_origins.append("https://face.service.tools.thefusionapps.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    origin = request.headers.get("origin")
    response = await call_next(request)
    if origin:
        # If origin matches our list or we allow all
        if "*" in cors_origins or origin in cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
    return response

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


def _collect_provider_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import onnxruntime as ort  # type: ignore

        try:
            out["onnxruntime"] = {
                "version": str(getattr(ort, "__version__", "")),
                "available_providers": list(ort.get_available_providers() or []),
            }
        except Exception:
            out["onnxruntime"] = {"version": str(getattr(ort, "__version__", ""))}
    except Exception as e:
        out["onnxruntime"] = {"error": str(e)}

    try:
        embedder = getattr(app.state, "embedder", None)
        out["embedder"] = {
            "class": str(embedder.__class__.__name__) if embedder is not None else None,
            "configured_providers": list(getattr(embedder, "providers", []) or []),
        }
    except Exception as e:
        out["embedder"] = {"error": str(e)}

    try:
        embedder = getattr(app.state, "embedder", None)
        fa = getattr(embedder, "app", None) if embedder is not None else None
        models = getattr(fa, "models", None)
        sess_providers: dict[str, Any] = {}
        if isinstance(models, dict):
            for k, m in models.items():
                sess = getattr(m, "session", None)
                if sess is None:
                    sess = getattr(m, "sess", None)
                if sess is not None and hasattr(sess, "get_providers"):
                    try:
                        sess_providers[str(k)] = list(sess.get_providers() or [])
                    except Exception as e:
                        sess_providers[str(k)] = {"error": str(e)}
                else:
                    sess_providers[str(k)] = None
        out["insightface"] = {
            "models": list(models.keys()) if isinstance(models, dict) else None,
            "session_providers": sess_providers,
        }
    except Exception as e:
        out["insightface"] = {"error": str(e)}

    return out


@app.get("/debug/providers")
def debug_providers() -> dict[str, Any]:
    return _collect_provider_info()


from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error("422 Validation Error at %s: %s", request.url, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc)},
    )


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

    # Expose internal helpers for cross_check module
    app.state.decode_image_bytes = _decode_image_bytes
    app.state.quality_check_and_embed = _quality_check_and_embed

    try:
        logger.info("provider_info=%s", _collect_provider_info())
    except Exception:
        pass

    # Single-GPU inference manager (caps concurrent GPU work; prevents thrash)
    try:
        gpu_enabled = str(os.environ.get("GPU_INFERENCE_MANAGER", "1") or "1").strip() not in (
            "0",
            "false",
            "False",
        )
    except Exception:
        gpu_enabled = True
    if gpu_enabled:
        try:
            max_q = int(os.environ.get("GPU_QUEUE_MAX", "256") or "256")
        except Exception:
            max_q = 256
        try:
            batch_ms = int(os.environ.get("GPU_BATCH_WINDOW_MS", "0") or "0")
        except Exception:
            batch_ms = 0
        app.state.gpu = GPUInferenceManager(
            embedder=app.state.embedder,
            max_queue=max_q,
            batch_window_ms=batch_ms,
        )
    else:
        app.state.gpu = None

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
        try:
            infer = getattr(app.state, "gpu", None) or app.state.embedder
        except Exception:
            infer = app.state.embedder
        app.state.index = _load_face_dataset(face_dir, infer.embed_bgr)

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
    app.state.enroll_lock = asyncio.Lock()
    app.state.thumbs_dir = os.environ.get("THUMBS_DIR", "/data/thumbs")
    app.state.search_events_dir = os.environ.get("SEARCH_EVENTS_DIR", "/data/events/search_query")
    app.state.search_thumbs_dir = os.environ.get("SEARCH_THUMBS_DIR", "/data/events/search_thumbs")
    try:
        _ensure_dir(app.state.thumbs_dir)
        _ensure_dir(app.state.search_events_dir)
        _ensure_dir(app.state.search_thumbs_dir)
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


@app.on_event("shutdown")
def _shutdown() -> None:
    try:
        mgr = getattr(app.state, "gpu", None)
    except Exception:
        mgr = None
    if mgr is not None:
        try:
            mgr.close()
        except Exception:
            pass


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

    cap = _subject_embedding_cap()
    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id)
    if existing >= cap:
        raise HTTPException(status_code=409, detail=f"subject embedding cap reached ({existing}/{cap})")

    num_embedded = 0
    emb_dim: int | None = None
    last_meta: dict[str, Any] | None = None

    for i, img_b64 in enumerate(req.images_b64):
        if existing >= cap:
            break
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
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{subject_id}:{image_hash}"))
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, image_id)
        image_path = _save_image(bgr, os.environ.get("IMAGES_DIR", "/data/images"), subject_id, image_id)

        if _enroll_dup_check_enabled():
            try:
                thr = float(_enroll_dup_min_similarity())
                hits = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1)
                if hits:
                    best = hits[0]
                    best_sid = str(best.get("subject_id") or "").strip()
                    best_sim = float(best.get("similarity") or 0.0)
                    if best_sid and best_sid != subject_id and best_sim >= thr:
                        extra = {
                            "matched_subject_id": best_sid,
                            "similarity": float(best_sim),
                            "threshold": float(thr),
                        }
                        last_meta = last_meta or {}
                        last_meta["enroll_duplicate_check"] = _quarantine_enroll_possible_match(
                            bgr,
                            subject_id=subject_id,
                            image_id=image_id,
                            reason="possible_match",
                            extra=extra,
                        )
                        continue
            except Exception:
                pass

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
        existing += 1
 

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

    cap = _subject_embedding_cap()
    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id)
    if existing >= cap:
        raise HTTPException(status_code=409, detail=f"subject embedding cap reached ({existing}/{cap})")

    num_embedded = 0
    emb_dim: int | None = None
    last_meta: dict[str, Any] | None = None

    for i, f in enumerate(files):
        if existing >= cap:
            break
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
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{subject_id}:{image_hash}"))
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, image_id)
        image_path = _save_image(bgr, os.environ.get("IMAGES_DIR", "/data/images"), subject_id, image_id)

        if _enroll_dup_check_enabled():
            try:
                thr = float(_enroll_dup_min_similarity())
                hits = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1)
                if hits:
                    best = hits[0]
                    best_sid = str(best.get("subject_id") or "").strip()
                    best_sim = float(best.get("similarity") or 0.0)
                    if best_sid and best_sid != subject_id and best_sim >= thr:
                        extra = {
                            "matched_subject_id": best_sid,
                            "similarity": float(best_sim),
                            "threshold": float(thr),
                            "filename": str(getattr(f, "filename", "") or ""),
                        }
                        last_meta = last_meta or {}
                        last_meta["enroll_duplicate_check"] = _quarantine_enroll_possible_match(
                            bgr,
                            subject_id=subject_id,
                            image_id=image_id,
                            reason="possible_match",
                            extra=extra,
                        )
                        continue
            except Exception:
                pass
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
        existing += 1

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

    thumb_path = ""
    try:
        from events_store import SearchEvent

        store: EventsStore | None = getattr(app.state, "events", None)
        if store is not None:
            event_id = str(uuid.uuid4())
            img_url, thumb_path = _save_search_query_assets(bgr, event_id)

            top_sid = results[0]["subject_id"] if results else None
            top_sim = results[0]["similarity"] if results else None
            ev = SearchEvent(
                event_id=event_id,
                ts=_t(),
                query_image_path=img_url,
                query_thumb_path=thumb_path,
                top_subject_id=top_sid,
                top_similarity=top_sim,
                results=results,
                meta=_meta,
            )
            store.insert_search_event(ev)
    except Exception as e:
        try:
            logger.error("failed to log search event: %s", str(e))
        except Exception:
            pass

    _record_event(app.state.search_events)
    return FaceSearchTopKResponse(
        results=[FaceSearchTopKItem(**r) for r in results],
        query_thumb_path=(thumb_path or None),
    )


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
    
    # Persistent Logging
    try:
        from events_store import SearchEvent
        event_id = str(uuid.uuid4())
        img_url, thumb_path = _save_search_query_assets(bgr, event_id)
        
        top_sid = results[0]["subject_id"] if results else None
        top_sim = results[0]["similarity"] if results else None
        
        ev = SearchEvent(
            event_id=event_id,
            ts=_t(),
            query_image_path=img_url,
            query_thumb_path=thumb_path,
            top_subject_id=top_sid,
            top_similarity=top_sim,
            results=results,
            meta=_meta
        )
        if app.state.events:
            app.state.events.insert_search_event(ev)
            logger.info("Search Event Logged: %s", event_id)
    except Exception as e:
        logger.error("failed to log search event: %s", str(e))

    _record_event(app.state.search_events)
    return FaceSearchTopKResponse(results=[FaceSearchTopKItem(**r) for r in results], query_thumb_path=thumb_path or None)


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

    # Persistent Logging
    try:
        from events_store import SearchEvent
        event_id = str(uuid.uuid4())
        img_url, thumb_path = _save_search_query_assets(bgr, event_id)
        
        top_sid = results[0]["subject_id"] if results else None
        top_sim = results[0]["similarity"] if results else None
        
        ev = SearchEvent(
            event_id=event_id,
            ts=_t(),
            query_image_path=img_url,
            query_thumb_path=thumb_path,
            top_subject_id=top_sid,
            top_similarity=top_sim,
            results=results,
            meta=meta
        )
        if app.state.events:
            app.state.events.insert_search_event(ev)
            logger.info("Recognize Event Logged: %s", event_id)
    except Exception as e:
        logger.error("failed to log recognize event: %s", str(e))

    if not items:
        return FaceRecognizeResponse(matched=False, results=[], meta=meta)

    best = items[0]
    if float(best.similarity) >= float(min_sim) and str(best.subject_id).strip():
        ok, second, margin, req_m = _passes_top2_margin(results, float(best.similarity))
        try:
            meta = dict(meta or {})
            meta["decision"] = {
                "status": "match" if ok else "no_match",
                "min_similarity": float(min_sim),
                "top2_second": second,
                "top2_margin": margin,
                "top2_required": req_m,
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
    


@app.get("/v1/faces/cross_match/{subject_id}", response_model=FaceSearchTopKResponse)
async def faces_cross_match(subject_id: str, top_k: int = 20) -> FaceSearchTopKResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    # 1. Get embedding for the subject
    vector = None
    try:
        idx_obj = getattr(app.state, "index", None)
        if idx_obj and subject_id in idx_obj.subject_ids:
            i = idx_obj.subject_ids.index(subject_id)
            vector = idx_obj.mean_embeddings[i]
    except Exception:
        pass

    if vector is None:
        # Fallback: Search Qdrant for the subject's own points to get a vector
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            hits = q.scroll(
                collection_name=app.state.qdrant_collection,
                scroll_filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=subject_id))]),
                limit=1,
                with_vectors=True
            )
            if hits and hits[0]:
                vector = hits[0][0].vector
        except Exception as e:
            logger.error("cross_match fallback failed: %s", str(e))

    if vector is None:
        # One last try: if it's the subject name directly
        raise HTTPException(status_code=404, detail=f"subject {subject_id} not found or has no embeddings")

    # 2. Search for similar subjects
    hits = _qdrant_search(q, app.state.qdrant_collection, vector, top_k=100)
    
    # 3. Filter for different subject_ids (usually visitor-*) and high similarity
    threshold = 0.85
    filtered = []
    seen_sids = {subject_id}
    
    for h in hits:
        sid = str(h.get("subject_id") or "").strip()
        sim = float(h.get("similarity") or 0.0)
        if sid and sid not in seen_sids and sim >= threshold:
            filtered.append(FaceSearchTopKItem(**h))
            seen_sids.add(sid)
            if len(filtered) >= top_k:
                break

    return FaceSearchTopKResponse(results=filtered)


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

    t_req0 = _t()
    t_total0 = _pc()
    t_decode0 = _pc()
    image_bytes = await file.read()
    bgr = _decode_image_bytes(image_bytes)
    decode_ms = int(max(0.0, (_pc() - t_decode0)) * 1000.0)
    h, w = bgr.shape[:2]

    event_id = str(uuid.uuid4())
    ts_val = float(ts) if ts is not None else _now_ts()
    source_path = str(source_path or str(getattr(file, "filename", "") or "")).strip()

    events_dir = os.environ.get("EVENTS_DIR", "/data/events")

    def _model_ms_from_infer_timing(detect_embed_ms_val: int, infer_timing_val: dict[str, Any]) -> int:
        try:
            gpu_exec = float(infer_timing_val.get("exec_ms", 0.0) or 0.0)
            if gpu_exec > 0.0:
                return int(round(gpu_exec))
        except Exception:
            pass
        return int(detect_embed_ms_val)

    primary_resp: RecognitionEventResponse | None = None
    faces_total = 0
    faces_processed = 0

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

    def _crop_face(bgr0: np.ndarray, face: Any) -> np.ndarray:
        try:
            bb = np.asarray(getattr(face, "bbox", None), dtype=np.float32).reshape(-1)
            if bb.size < 4:
                return bgr0
            x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            ih, iw = bgr0.shape[:2]
            x1i = max(0, min(int(x1), int(iw - 1)))
            y1i = max(0, min(int(y1), int(ih - 1)))
            x2i = max(0, min(int(x2), int(iw)))
            y2i = max(0, min(int(y2), int(ih)))
            if x2i <= x1i or y2i <= y1i:
                return bgr0
            return bgr0[y1i:y2i, x1i:x2i].copy()
        except Exception:
            return bgr0

    def _pick_best_face_by_area(faces0: list[Any]) -> Any:
        best = None
        best_score = -1.0
        for f in faces0 or []:
            if f is None:
                continue
            try:
                bb = np.asarray(getattr(f, "bbox", None), dtype=np.float32).reshape(-1)
                if bb.size >= 4:
                    area = max(0.0, float(bb[2] - bb[0])) * max(0.0, float(bb[3] - bb[1]))
                else:
                    area = 0.0
            except Exception:
                area = 0.0
            try:
                ds = float(getattr(f, "det_score", 0.0) or 0.0)
            except Exception:
                ds = 0.0
            score = float(area) * (0.5 + float(ds))
            if score > best_score:
                best_score = score
                best = f
        return best

    # detect faces
    t_model0 = _t()
    t_detect0 = _pc()
    faces: list[Any] = []
    infer_timing: dict[str, Any] = {}
    try:
        infer = getattr(app.state, "gpu", None) or app.state.embedder
        if process_all_faces:
            if hasattr(infer, "detect_all_timed"):
                faces, tinfo = infer.detect_all_timed(bgr)
                infer_timing = dict(tinfo or {})
            else:
                faces = list(infer.detect_all(bgr))
        else:
            use_largest = str(os.environ.get("FACE_SERVICE_USE_LARGEST_FACE", "0") or "0").strip() in (
                "1",
                "true",
                "True",
                "yes",
                "YES",
            )
            if use_largest:
                if hasattr(infer, "detect_all_timed"):
                    faces_all, tinfo = infer.detect_all_timed(bgr)
                    infer_timing = dict(tinfo or {})
                    face0 = _pick_best_face_by_area(list(faces_all or []))
                    faces = [face0]
                else:
                    faces = [
                        _pick_best_face_by_area(list(infer.detect_all(bgr)))
                        if hasattr(infer, "detect_all")
                        else infer.detect_best(bgr)
                    ]
            else:
                if hasattr(infer, "detect_best_timed"):
                    face0, tinfo = infer.detect_best_timed(bgr)
                    faces = [face0]
                    infer_timing = dict(tinfo or {})
                else:
                    faces = [infer.detect_best(bgr)]
        faces_total = len(faces)
    except ValueError:
        if primary_resp is None:
            # should be rare (e.g. all faces were None)
            reason = "no_face_detected"
            detect_embed_ms = int(max(0.0, (_pc() - t_detect0)) * 1000.0)
            processing_ms = _model_ms_from_infer_timing(detect_embed_ms, infer_timing)
            img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{event_id}.jpg")
            thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{event_id}")
            image_saved_at = _now_ts() if img_path else None
            total_ms = int(max(0.0, (_pc() - t_total0)) * 1000.0)
            meta = {
                "quality": {"status": "rejected", "reason": reason},
                "decision": {"status": "rejected"},
                "timing": {
                    "decode_ms": int(decode_ms),
                    "detect_embed_ms": int(detect_embed_ms),
                    "gpu_queue_wait_ms": float(infer_timing.get("queue_wait_ms", 0.0) or 0.0),
                    "gpu_exec_ms": float(infer_timing.get("exec_ms", 0.0) or 0.0),
                    "total_ms": int(total_ms),
                },
                "faces_total": int(faces_total),
                "faces_processed": int(faces_processed),
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
                    processing_ms=processing_ms,
                    model_ms=processing_ms,
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
                processing_ms=processing_ms,
                model_ms=processing_ms,
                rejected_reason=reason,
                bbox=None,
                det_score=None,
                image_path=img_path,
                thumb_path=thumb_path,
                image_saved_at=image_saved_at,
                meta=meta,
            )
        else:
            return primary_resp


    detect_ms = int(max(0.0, (_t() - t_model0)) * 1000.0)
    detect_embed_ms = int(max(0.0, (_pc() - t_detect0)) * 1000.0)

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
    faces_processed = 0
    face_model_ms_total = 0
    for idx, face in enumerate(faces_sorted):
        t_face0 = _t()
        t_face_model0 = _t()
        t_face_pc0 = _pc()
        ev_id = str(uuid.uuid4())
        bbox, det_score = _face_bbox_for_meta(face)

        quality_ms = 0
        qdrant_ms = 0
        save_ms = 0

        quality_meta: dict[str, Any] | None = None
        if evaluator is not None:
            try:
                t_e0 = _pc()
                emb = _embed_from_face(face)
                if emb is not None:
                    try:
                        quality_meta = evaluator.evaluate(bgr, face)
                    except Exception:
                        quality_meta = None
                    if isinstance(quality_meta, dict) and quality_meta.get("status") == "rejected":
                        raise ValueError(f"quality_reject:{str(quality_meta.get('reason') or 'unknown')}")
                else:
                    bgr_face = _crop_face(bgr, face)
                    emb, quality_meta = _quality_check_and_embed(bgr_face)
                quality_ms = int(max(0.0, (_pc() - t_e0)) * 1000.0)

                try:
                    emb_dim = int(np.asarray(emb).reshape(-1).shape[0])
                    _ensure_qdrant_collection(q, app.state.qdrant_collection, vector_size=emb_dim)
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"qdrant init failed: {str(e)}")
            except Exception:
                quality_meta = {"status": "rejected", "reason": "quality_eval_failed"}
            if isinstance(quality_meta, dict) and quality_meta.get("status") == "rejected":
                reason = str(quality_meta.get("reason") or "unknown")
                face_model_ms = int(max(0.0, (_t() - t_face_model0)) * 1000.0)
                face_model_ms_total += face_model_ms
                t_s0 = _pc()
                img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{ev_id}.jpg")
                thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
                save_ms = int(max(0.0, (_pc() - t_s0)) * 1000.0)
                image_saved_at = _now_ts() if img_path else None
                processing_ms = _model_ms_from_infer_timing(detect_embed_ms, infer_timing)
                model_ms = int(detect_ms + face_model_ms_total)
                total_ms = int(max(0.0, (_pc() - t_total0)) * 1000.0)
                meta = {
                    "quality": quality_meta,
                    "decision": {"status": "rejected"},
                    "timing": {
                        "decode_ms": int(decode_ms),
                        "detect_embed_ms": int(detect_embed_ms),
                        "gpu_queue_wait_ms": float(infer_timing.get("queue_wait_ms", 0.0) or 0.0),
                        "gpu_exec_ms": float(infer_timing.get("exec_ms", 0.0) or 0.0),
                        "quality_ms": int(quality_ms),
                        "qdrant_ms": int(qdrant_ms),
                        "save_ms": int(save_ms),
                        "face_total_ms": int(max(0.0, (_pc() - t_face_pc0)) * 1000.0),
                        "total_ms": int(total_ms),
                    },
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
                        processing_ms=processing_ms,
                        model_ms=model_ms,
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
                        processing_ms=processing_ms,
                        model_ms=model_ms,
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
            face_model_ms = int(max(0.0, (_t() - t_face_model0)) * 1000.0)
            face_model_ms_total += face_model_ms
            t_s0 = _pc()
            img_path = _save_event_image(bgr, events_dir, f"rejected/{camera}/{ev_id}.jpg")
            thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
            save_ms = int(max(0.0, (_pc() - t_s0)) * 1000.0)
            image_saved_at = _now_ts() if img_path else None
            processing_ms = _model_ms_from_infer_timing(detect_embed_ms, infer_timing)
            model_ms = int(detect_ms + face_model_ms_total)
            total_ms = int(max(0.0, (_pc() - t_total0)) * 1000.0)
            meta = {
                "quality": quality_meta,
                "decision": {"status": "rejected"},
                "timing": {
                    "decode_ms": int(decode_ms),
                    "detect_embed_ms": int(detect_embed_ms),
                    "gpu_queue_wait_ms": float(infer_timing.get("queue_wait_ms", 0.0) or 0.0),
                    "gpu_exec_ms": float(infer_timing.get("exec_ms", 0.0) or 0.0),
                    "quality_ms": int(quality_ms),
                    "qdrant_ms": int(qdrant_ms),
                    "save_ms": int(save_ms),
                    "face_total_ms": int(max(0.0, (_pc() - t_face_pc0)) * 1000.0),
                    "total_ms": int(total_ms),
                },
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
                    processing_ms=processing_ms,
                    model_ms=model_ms,
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
                    processing_ms=processing_ms,
                    model_ms=model_ms,
                    rejected_reason=reason,
                    bbox=bbox,
                    det_score=det_score,
                    image_path=img_path,
                    thumb_path=thumb_path,
                    image_saved_at=image_saved_at,
                    meta=meta,
                )
            continue

        try:
            emb_dim = int(np.asarray(emb).reshape(-1).shape[0])
            _ensure_qdrant_collection(q, app.state.qdrant_collection, vector_size=emb_dim)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant init failed: {str(e)}")

        t_qs0 = _pc()
        results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k)
        
        # Check in-memory cache if no database match
        if not results or results[0]["similarity"] < float(min_sim):
            cached_sid = _get_recent_enrollment_match(emb, float(min_sim))
            if cached_sid:
                # Synthesize a result from cache
                results = [{"subject_id": cached_sid, "similarity": 0.99, "id": "cached"}] + (results or [])

        qdrant_ms = int(max(0.0, (_pc() - t_qs0)) * 1000.0)
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

        auto_added = False
        auto_add_reason: str | None = None

        face_model_ms = int(max(0.0, (_t() - t_face_model0)) * 1000.0)
        face_model_ms_total += face_model_ms
        model_ms = int(detect_ms + face_model_ms_total)
        t_s0 = _pc()
        img_path = _save_event_image(
            bgr,
            events_dir,
            f"{'accepted' if matched else 'no_match'}/{camera}/{ev_id}.jpg",
        )
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, f"evt-{ev_id}")
        save_ms = int(max(0.0, (_pc() - t_s0)) * 1000.0)
        image_saved_at = _now_ts() if img_path else None

        no_match_auto_enroll: dict[str, Any] | None = None
        if (not matched) and _no_match_auto_enroll_enabled():
            try:
                async with app.state.enroll_lock:
                    # Re-check match after acquiring lock to handle race conditions
                    try:
                        # 1. Check in-memory cache first (fastest)
                        # Use ENROLL_DUPLICATE_MIN_SIM for re-check if available
                        recheck_thr = float(os.environ.get("ENROLL_DUPLICATE_MIN_SIM", app.state.min_similarity))
                        cached_sid = _get_recent_enrollment_match(emb, recheck_thr)
                        if cached_sid:
                            logger.info("race_condition_prevented: matched %s via cache (thr=%.2f)", cached_sid, recheck_thr)
                            matched = True
                            subject_id = cached_sid
                            similarity = 0.99
                            decision = "match"
                        
                        # 2. Re-check Qdrant if still no match
                        if not matched:
                            q = getattr(app.state, "qdrant", None)
                            if q:
                                results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=5)
                                if results and results[0]["similarity"] >= recheck_thr:
                                    logger.info("race_condition_prevented: matched %s via DB re-check (thr=%.2f)", results[0]["subject_id"], recheck_thr)
                                    matched = True
                                    subject_id = results[0]["subject_id"]
                                    similarity = results[0]["similarity"]
                                    decision = "match"
                                # Fall through to the rest of the logic which will now see 'matched=True'
                    except Exception:
                        pass

                    if not matched:
                        try:
                            try:
                                block_thr = float(_no_match_auto_enroll_block_min_similarity())
                            except Exception:
                                block_thr = 0.80

                            try:
                                attach_thr = float(_no_match_auto_attach_min_similarity())
                            except Exception:
                                attach_thr = 0.70
                            try:
                                attach_margin_thr = float(_no_match_auto_attach_min_margin())
                            except Exception:
                                attach_margin_thr = 0.10

                            try:
                                best_sid = str((results[0].get("subject_id") if results else "") or "").strip()
                                best_sim = float((results[0].get("similarity") if results else 0.0) or 0.0)
                            except Exception:
                                best_sid = ""
                                best_sim = 0.0

                            second_sim: float | None = None
                            try:
                                if results and len(results) >= 2:
                                    second_sim = float(results[1].get("similarity") or 0.0)
                            except Exception:
                                second_sim = None

                            if best_sid:
                                try:
                                    prefix = _no_match_auto_enroll_prefix()
                                except Exception:
                                    prefix = "unknown"

                                if prefix and best_sid.startswith(f"{prefix}-") and best_sim >= float(attach_thr):
                                    margin_ok = True
                                    if second_sim is not None:
                                        try:
                                            margin_ok = (float(best_sim) - float(second_sim)) >= float(attach_margin_thr)
                                        except Exception:
                                            margin_ok = False
                                    if margin_ok:
                                        matched = True
                                        subject_id = best_sid
                                        similarity = float(best_sim)
                                        decision = "match"
                                        no_match_auto_enroll = {
                                            "enabled": True,
                                            "enrolled": False,
                                            "reason": "attached_existing_visitor",
                                            "subject_id": str(best_sid),
                                            "similarity": float(best_sim),
                                            "threshold": float(attach_thr),
                                            "second_similarity": float(second_sim) if second_sim is not None else None,
                                            "margin_threshold": float(attach_margin_thr),
                                        }
                                        raise RuntimeError("skip_auto_enroll_attached_existing")

                            if best_sid and best_sim >= block_thr:
                                no_match_auto_enroll = {
                                    "enabled": True,
                                    "enrolled": False,
                                    "reason": "possible_match",
                                    "matched_subject_id": best_sid,
                                    "similarity": float(best_sim),
                                    "threshold": float(block_thr),
                                }
                                raise RuntimeError("skip_auto_enroll_possible_match")

                            prefix = _no_match_auto_enroll_prefix()
                            try:
                                seq = store.next_counter(f"no_match_auto_enroll:{prefix}", start=1)
                            except Exception:
                                seq = int(_now_ts())
                            new_subject_id = f"{prefix}-{int(seq)}"
                            try:
                                from qdrant_client.http.models import PointStruct
                            except Exception as e:
                                raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

                            nm_image_id = f"nm-{ev_id}"
                            nm_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{new_subject_id}:{ev_id}"))
                            try:
                                q.upsert(
                                    collection_name=app.state.qdrant_collection,
                                    points=[
                                        PointStruct(
                                            id=nm_point_id,
                                            vector=np.asarray(emb, dtype=np.float32).reshape(-1).tolist(),
                                            payload={
                                                "subject_id": str(new_subject_id),
                                                "image_id": str(nm_image_id),
                                                "created_at": _iso_now(),
                                                "thumb_path": thumb_path,
                                                "image_path": img_path,
                                                "source": "no_match_auto_enroll",
                                                "event_id": str(ev_id),
                                            },
                                        )
                                    ],
                                    wait=True,
                                )
                                _add_recent_enrollment(emb, new_subject_id)
                            except Exception as e:
                                logger.error("auto-enroll upsert failed: %s", str(e))
                                raise

                            no_match_auto_enroll = {
                                "enabled": True,
                                "enrolled": True,
                                "subject_id": new_subject_id,
                            }
                        except RuntimeError as e:
                            if str(e) not in ("skip_auto_enroll_attached_existing", "skip_auto_enroll_possible_match"):
                                logger.error("auto-enroll error: %s", str(e))
                        except Exception as e:
                            logger.error("auto-enroll error: %s", str(e))
                            no_match_auto_enroll = {
                                "enabled": True,
                                "enrolled": False,
                                "error": str(e),
                            }
            except Exception as e:
                logger.error("Lock-protected auto-enroll failed: %s", str(e))
                if str(e) not in ("skip_auto_enroll_possible_match", "skip_auto_enroll_attached_existing"):
                    no_match_auto_enroll = {
                        "enabled": True,
                        "enrolled": False,
                        "error": str(e),
                    }

        if matched and subject_id and similarity is not None and _auto_add_enabled():
            if float(similarity) >= float(_auto_add_min_similarity()):
                try:
                    cap = _subject_embedding_cap()
                    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id)
                    if existing >= cap:
                        auto_add_reason = "cap_reached"
                    else:
                        try:
                            from qdrant_client.http.models import PointStruct
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

                        auto_image_id = f"auto-{ev_id}"
                        auto_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{subject_id}:{ev_id}"))
                        t0_up = _t()
                        try:
                            q.upsert(
                                collection_name=app.state.qdrant_collection,
                                points=[
                                    PointStruct(
                                        id=auto_point_id,
                                        vector=np.asarray(emb, dtype=np.float32).reshape(-1).tolist(),
                                        payload={
                                            "subject_id": str(subject_id),
                                            "image_id": str(auto_image_id),
                                            "created_at": _iso_now(),
                                            "thumb_path": thumb_path,
                                            "image_path": img_path,
                                            "source": "auto_recognized",
                                            "event_id": str(ev_id),
                                            "camera": str(camera),
                                            "similarity": float(similarity),
                                        },
                                    )
                                ],
                            )
                            auto_added = True
                        except Exception as e:
                            _QDRANT_ERR.inc()
                            auto_add_reason = f"qdrant_upsert_failed:{str(e)}"
                        finally:
                            try:
                                _QDRANT_UPSERT_LAT.observe(max(0.0, _t() - t0_up))
                            except Exception:
                                pass
                except Exception as e:
                    auto_add_reason = f"auto_add_failed:{str(e)}"
            else:
                auto_add_reason = "below_auto_add_min_sim"
        processing_ms = _model_ms_from_infer_timing(detect_embed_ms, infer_timing)
        total_ms = int(max(0.0, (_pc() - t_total0)) * 1000.0)

        meta = {
            "quality": quality_meta,
            "decision": {
                "status": decision,
                "matched": bool(matched),
                "min_similarity": float(min_sim),
                "auto_add_embedding": {
                    "enabled": bool(_auto_add_enabled()),
                    "added": bool(auto_added),
                    "reason": auto_add_reason,
                    "min_similarity": float(_auto_add_min_similarity()),
                },
                "no_match_auto_enroll": (
                    no_match_auto_enroll
                    if no_match_auto_enroll is not None
                    else {"enabled": bool(_no_match_auto_enroll_enabled()), "enrolled": False}
                ),
            },
            "top2_second": top2_second,
            "top2_margin": top2_margin,
            "top2_required": top2_required,
            "timing": {
                "decode_ms": int(decode_ms),
                "detect_embed_ms": int(detect_embed_ms),
                "gpu_queue_wait_ms": float(infer_timing.get("queue_wait_ms", 0.0) or 0.0),
                "gpu_exec_ms": float(infer_timing.get("exec_ms", 0.0) or 0.0),
                "quality_ms": int(quality_ms),
                "qdrant_ms": int(qdrant_ms),
                "save_ms": int(save_ms),
                "face_total_ms": int(max(0.0, (_pc() - t_face_pc0)) * 1000.0),
                "total_ms": int(total_ms),
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
                processing_ms=processing_ms,
                model_ms=model_ms,
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
                processing_ms=processing_ms,
                model_ms=model_ms,
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
    source_path: str | None = None,
    decision: str | None = None,
    min_similarity: float | None = None,
    max_similarity: float | None = None,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    limit: int = 100,
    cursor: float | None = None,
) -> RecognitionEventsListResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    items, next_cur = store.list_events(
        camera=camera,
        subject_id=subject_id,
        source_path=source_path,
        decision=decision,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        since_ts=since_ts,
        until_ts=until_ts,
        limit=limit,
        cursor_ts=cursor,
    )
    return RecognitionEventsListResponse(
        items=[RecognitionEventResponse(**it) for it in items],
        cursor=next_cur,
    )


@app.get("/v1/search_history", response_model=SearchEventsListResponse)
def list_search_events(
    limit: int = 100,
    cursor: float | None = None,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
) -> SearchEventsListResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    items, next_cursor = store.list_search_events(
        limit=limit,
        cursor_ts=cursor,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    res = []
    for it in items:
        # Convert results list of dicts to FaceSearchTopKItem
        results_objs = [FaceSearchTopKItem(**r) for r in it.get("results") or []]
        it["results"] = results_objs
        res.append(SearchEventResponse(**it))
    
    return SearchEventsListResponse(items=res, cursor=next_cursor)


@app.get("/v1/search_history/stats", response_model=SearchEventsStatsResponse)
def search_events_stats(
    match_threshold: float = 0.8,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
) -> SearchEventsStatsResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    out = store.search_events_stats(
        match_threshold=float(match_threshold),
        since_ts=since_ts,
        until_ts=until_ts,
    )
    return SearchEventsStatsResponse(**out)


@app.get("/v1/search_history/asset/image/{event_id}")
def get_search_event_image(event_id: str) -> Response:
    path = os.path.join(app.state.search_events_dir, f"{event_id}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="query image not found")
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@app.get("/v1/search_history/asset/thumb/{event_id}")
def get_search_event_thumb(event_id: str) -> Response:
    path = os.path.join(app.state.search_thumbs_dir, f"{event_id}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="query thumbnail not found")
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@app.get("/v1/events/recognition/cameras", response_model=list[str])
def list_recognition_cameras(limit: int = 5000) -> list[str]:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    out: set[str] = set()
    try:
        for c in store.list_cameras(limit=int(limit or 5000)):
            c = str(c or "").strip()
            if c:
                out.add(c)
    except Exception:
        pass

    events_dir = str(os.environ.get("EVENTS_DIR", "/data/events") or "/data/events")
    try:
        if os.path.isdir(events_dir):
            for name in os.listdir(events_dir):
                p = os.path.join(events_dir, name)
                if not os.path.isdir(p):
                    continue

                # Common layout: /data/events/<bucket>/<camera>/...
                try:
                    for cam in os.listdir(p):
                        cam_p = os.path.join(p, cam)
                        if not os.path.isdir(cam_p):
                            continue
                        cam = str(cam or "").strip()
                        if cam:
                            out.add(cam)
                except Exception:
                    # Also allow layout: /data/events/<camera>/...
                    name = str(name or "").strip()
                    if name:
                        out.add(name)
    except Exception:
        pass

    cams = sorted(out)
    lim = max(1, min(int(limit or 5000), 50000))
    return cams[:lim]


class ForwardEventRequest(BaseModel):
    event_id: str
    target_url: str | None = None


@app.post("/v1/events/recognition/forward")
async def forward_recognition_event(req: ForwardEventRequest) -> dict[str, Any]:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    event_id = str(req.event_id or "").strip()
    if not event_id:
        raise HTTPException(status_code=400, detail="event_id is required")

    it = store.get_event(event_id)
    if not it:
        raise HTTPException(status_code=404, detail="event not found")

    target_url = str(req.target_url or os.environ.get("FACE_SERVICE_FORWARD_URL", "") or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="target_url is required")

    img_path = str(it.get("image_path") or "")
    if not img_path.startswith("/events/"):
        raise HTTPException(status_code=400, detail="event has no persisted image")

    events_dir = os.environ.get("EVENTS_DIR", "/data/events")
    abs_path = os.path.join(str(events_dir), img_path.replace("/events/", "", 1).lstrip("/"))
    try:
        with open(abs_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read image: {e}")

    payload = {
        "event_id": it.get("event_id"),
        "ts": it.get("ts"),
        "camera": it.get("camera"),
        "source_path": it.get("source_path"),
        "decision": it.get("decision"),
        "subject_id": it.get("subject_id"),
        "similarity": it.get("similarity"),
        "processing_ms": it.get("processing_ms"),
        "rejected_reason": it.get("rejected_reason"),
        "bbox": it.get("bbox"),
        "det_score": it.get("det_score"),
        "meta": it.get("meta"),
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            target_url,
            data={"metadata_json": json.dumps(payload)},
            files={"file": (f"{event_id}.jpg", img_bytes, "image/jpeg")},
        )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"forward failed: {r.status_code} {r.text[:300]}")
    return {"forwarded": True, "status_code": int(r.status_code)}


@app.get("/v1/events/recognition/feedback_stats", response_model=FeedbackStatsResponse)
def recognition_feedback_stats(
    since_ts: float | None = None,
    until_ts: float | None = None,
    camera: str | None = None,
) -> FeedbackStatsResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    st = store.feedback_stats(since_ts=since_ts, until_ts=until_ts, camera=camera)
    return FeedbackStatsResponse(**st)


@app.get("/v1/events/recognition/stats", response_model=RecognitionStatsResponse)
def recognition_stats(
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    camera: str | None = None,
) -> RecognitionStatsResponse:
    s0, e0 = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )
    res = app.state.events.recognition_stats(
        since_ts=s0,
        until_ts=e0,
        camera=camera,
    )
    return RecognitionStatsResponse(**res)


@app.get("/v1/events/recognition/{event_id}", response_model=RecognitionEventResponse)
def get_recognition_event(event_id: str) -> RecognitionEventResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")
    it = store.get_event(event_id)
    if not it:
        raise HTTPException(status_code=404, detail="event not found")
    return RecognitionEventResponse(**it)


@app.post("/v1/events/recognition/{event_id}/feedback", response_model=EventFeedbackResponse)
def set_recognition_event_feedback(event_id: str, req: EventFeedbackRequest) -> EventFeedbackResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    event_id = str(event_id or "").strip()
    if not event_id:
        raise HTTPException(status_code=400, detail="event_id is required")

    label = str(req.label or "").strip().lower() if req.label is not None else None
    if label is not None:
        if label == "":
            label = None
        elif label not in ("tp", "fp", "fn", "ignore"):
            raise HTTPException(status_code=400, detail="invalid label; use one of: tp, fp, fn, ignore")

    note = str(req.note or "").strip() if req.note is not None else None
    updated_at = float(_now_ts())
    updated = bool(store.set_feedback(event_id, label=label, note=note, updated_at=updated_at))
    if not updated:
        raise HTTPException(status_code=404, detail="event not found")

    return EventFeedbackResponse(
        event_id=event_id,
        updated=True,
        feedback_label=label,
        feedback_note=note,
        feedback_updated_at=updated_at,
    )


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

    # Persistent Logging
    try:
        from events_store import SearchEvent
        event_id = str(uuid.uuid4())
        img_url, thumb_path = _save_search_query_assets(bgr, event_id)
        
        top_sid = results[0]["subject_id"] if results else None
        top_sim = results[0]["similarity"] if results else None
        
        ev = SearchEvent(
            event_id=event_id,
            ts=_t(),
            query_image_path=img_url,
            query_thumb_path=thumb_path,
            top_subject_id=top_sid,
            top_similarity=top_sim,
            results=results,
            meta=meta
        )
        if app.state.events:
            app.state.events.insert_search_event(ev)
            logger.info("Recognize Upload Event Logged: %s", event_id)
    except Exception as e:
        logger.error("failed to log recognize upload event: %s", str(e))

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


@app.post("/v1/quality/check_upload", response_model=QualityCheckResponse)
async def quality_check_upload(file: UploadFile = File(...)) -> QualityCheckResponse:
    t0 = _pc()
    image_bytes = await file.read()
    decode_t0 = _pc()
    bgr = _decode_image_bytes(image_bytes)
    decode_ms = int(max(0.0, (_pc() - decode_t0)) * 1000.0)

    q_t0 = _pc()
    results, meta, annotated = _quality_check_all(bgr)
    quality_ms = int(max(0.0, (_pc() - q_t0)) * 1000.0)

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    total_ok = all(r["ok"] for r in results) if results else False
    total_quality = "pass" if total_ok else "fail"

    total_ms = int(max(0.0, (_pc() - t0)) * 1000.0)
    timing = dict(meta.get("timing") or {})
    timing.update({"decode_ms": int(decode_ms), "quality_ms": int(quality_ms), "total_ms": int(total_ms)})

    return QualityCheckResponse(
        ok=bool(total_ok),
        total_quality=str(total_quality),
        faces=[FaceQualityResult(**r) for r in results],
        annotated_image=f"data:image/jpeg;base64,{annotated_b64}",
        timing=timing,
    )


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
    embeddings_cap: int | None = None
    embeddings_capped: bool | None = None


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
def list_subjects(
    cursor: str | None = None,
    limit: int = 50,
    with_counts: bool = True,
    q: str | None = None,
) -> SubjectsListResponse:
    client = getattr(app.state, "qdrant", None)
    if client is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    limit = max(1, min(int(limit or 50), 10000))

    qstr = str(q or "").strip().lower()
    want_filter = bool(qstr)

    try:
        scan_limit = 10000

        points: list[Any] = []
        next_cur: Any = cursor
        scanned = 0
        uniq: dict[str, None] = {}
        while True:
            scroll_kwargs: dict[str, Any] = {
                "collection_name": app.state.qdrant_collection,
                "limit": int(scan_limit),
                "with_payload": True,
                "with_vectors": False,
            }
            if next_cur:
                scroll_kwargs["offset"] = next_cur
            batch, new_next = client.scroll(**scroll_kwargs)
            next_cur = new_next

            for pnt in batch or []:
                scanned += 1
                try:
                    payload = getattr(pnt, "payload", None) or {}
                    sid = str(payload.get("subject_id") or "").strip()
                    if not sid:
                        continue
                    if want_filter and (qstr not in sid.lower()):
                        continue
                    if sid in uniq:
                        continue
                    uniq[sid] = None
                except Exception:
                    continue

                if len(uniq) >= int(limit):
                    break

            if len(uniq) >= int(limit):
                break
            if not next_cur:
                break

        points = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant scroll failed: {str(e)}")

    cap = _subject_embedding_cap()
    items: list[SubjectItem] = []
    if with_counts:
        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")
        for sid in uniq.keys():
            try:
                cnt = client.count(
                    collection_name=app.state.qdrant_collection,
                    exact=True,
                    count_filter=Filter(must=[FieldCondition(key="subject_id", match=MatchValue(value=sid))]),
                )
                n = int(getattr(cnt, "count", 0) or 0)
            except Exception:
                n = 0
            items.append(
                SubjectItem(
                    subject_id=sid,
                    embeddings_count=n,
                    embeddings_cap=cap,
                    embeddings_capped=bool(n >= cap),
                )
            )
    else:
        items = [
            SubjectItem(
                subject_id=sid,
                embeddings_count=0,
                embeddings_cap=cap,
                embeddings_capped=False,
            )
            for sid in uniq.keys()
        ]

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


@app.get("/v1/subjects/{subject_id}", response_model=SubjectItem)
def get_subject(subject_id: str) -> SubjectItem:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    cap = _subject_embedding_cap()
    n = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id)
    return SubjectItem(subject_id=subject_id, embeddings_count=n, embeddings_cap=cap, embeddings_capped=bool(n >= cap))
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
