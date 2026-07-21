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
import gc
import psutil
import concurrent.futures
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import datetime, timezone, date, timedelta
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, Depends, Security, BackgroundTasks, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices

from src.models.schemas import (
    FaceIndex, FaceSearchRequest, FaceSearchResponse, FaceAddRequest,
    GroupCreateRequest, GroupResponse, GroupListResponse, FaceAddResponse,
    BranchCreateRequest, BranchResponse, BranchListResponse,
    FaceSearchTopKRequest, FaceSearchTopKItem, FaceSearchTopKResponse,
    FaceRecognizeRequest, FaceRecognizeResponse, FaceQualityResult,
    QualityCheckResponse, RecognitionEventResponse, RecognitionEventsListResponse,
    SearchEventResponse, SearchEventsListResponse, SearchEventsStatsResponse,
    RecognitionStatsResponse, EventFeedbackRequest, EventFeedbackResponse,
    FeedbackStatsResponse, RecognitionFetchRequest, FaceSubjectsResponse,
    FaceDeleteSubjectResponse, FaceCompareRequest, FaceCompareResponse,
    PrivacyExtractRequest, PrivacyExtractResponse, PrivacyCropItem,
    PrivacyBlurRequest, PrivacyBlurResponse,
    KeyCreateRequest, KeyInfo, KeyListResponse, KeyDeleteResponse
)
from src.utils.helpers import (
    _sha1_hex, _sha1_bytes_hex, _uuid5_from_name, _decode_b64_image,
    _decode_image_bytes, _t, _now_ts, _iso_now, _ensure_dir, _save_thumb
)
from src.services.inference_manager import GPUInferenceManager
from src.services.events_store import EventsStore, RecognitionEvent, SearchEvent
from src.core.config_loader import apply_env_defaults_from_config, load_config

from quality import FaceQualityEvaluator
from embedders.buffalo_l import (
    BuffaloLEmbedder,
    _l2_normalize,
    _quality_check_and_embed as _embed_quality_check_and_embed,
)
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from ui_page import ui_html
from cross_check import cross_check_router

import httpx

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = logging.getLogger("uvicorn.error")

# NOTE: Do NOT capture API_KEY at module load time.
# config.yaml is applied via apply_env_defaults_from_config() inside @app.on_event("startup"),
# which runs AFTER this module is imported. Reading the key here would always get the
# fallback 'your-secret-key' even when config.yaml has a proper key set.
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _get_api_key() -> str:
    """Read the expected API key at request time so config.yaml-sourced values are honoured."""
    return os.environ.get("FACE_SERVICE_API_KEY", "your-secret-key")

def _get_legacy_api_key() -> str:
    """Read the legacy API key from environment."""
    return os.environ.get("FACE_SERVICE_LEGACY_API_KEY", "")

def _resolve_access_key(header_value: str, *, required: bool) -> str:
    """Map an API key -> data partition (access_key).

    Each distinct key owns an isolated bucket, so the dashboard shows only the
    data belonging to whatever single key the user supplies. The two configured
    keys keep their historical buckets for back-compat.
    """
    val = str(header_value or "").strip()
    if not val:
        if required:
            raise HTTPException(status_code=403, detail="API Key missing")
        return "standard"
    if val == str(_get_api_key()).strip():
        return "standard"
    legacy = _get_legacy_api_key()
    if legacy and val == str(legacy).strip():
        return "legacy"
    # Master-issued key stored in the registry -> its assigned tenant bucket.
    reg = _registry_lookup(val)
    if reg:
        return reg
    # Any other key = its own deterministic tenant bucket.
    return "k_" + hashlib.sha1(val.encode("utf-8")).hexdigest()[:16]

async def get_api_key(header_value: str = Security(api_key_header)):
    return _resolve_access_key(header_value, required=True)

async def get_optional_access_key(header_value: str = Security(api_key_header)) -> str:
    return _resolve_access_key(header_value, required=False)


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _normalize_created_at(created_at: str | None, ts: float | None) -> str:
    if created_at and str(created_at).strip():
        return str(created_at).strip()
    if ts is not None:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            pass
    return _iso_now()


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

# Load-shedding: requests rejected with 503 because the in-flight cap was hit.
_REQ_SHED_TOTAL = Counter("face_requests_shed_total", "Heavy requests rejected by concurrency gate")


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
        ok = cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
             logger.error(f"cv2.imwrite failed for {img_path}")
             return "", ""

        # Save thumbnail
        thumb_path = _save_thumb(bgr, app.state.search_thumbs_dir, image_id)
        if not thumb_path:
             logger.error(f"failed to save thumbnail for {image_id}")

        return f"/v1/search_history/asset/image/{image_id}", thumb_path
    except Exception as e:
        logger.error(f"failed to save search query assets for {image_id}: {str(e)}")
        return "", ""

def _decode_b64_bytes(image_b64: str) -> bytes:
    if image_b64.startswith(("http://", "https://")):
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(image_b64)
                r.raise_for_status()
                return r.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to download image from URL: {e}")
    try:
        return base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_b64")


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


# --- Event-loop safety: blocking CPU/GPU work (image decode, quality+embed,
#     face detection) must never run inline in an `async def` handler, or it
#     freezes the single event loop and starves /health -> orchestrator kills
#     the container under load. These helpers push the blocking call onto a
#     worker thread (ort/cv2/httpx all release the GIL) so the loop stays live.
async def _decode_b64_image_async(image_b64: str) -> np.ndarray:
    return await asyncio.to_thread(_decode_b64_image, image_b64)


async def _quality_check_and_embed_async(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    return await asyncio.to_thread(_quality_check_and_embed, bgr)


async def _detect_all_async(infer: Any, bgr: np.ndarray) -> list[Any]:
    return list(await asyncio.to_thread(infer.detect_all, bgr))


def _build_decode_pool() -> "concurrent.futures.ProcessPoolExecutor":
    """CPU decode pool. `max_tasks_per_child` recycles workers so a slow memory
    leak or a single huge image can't grow a child unbounded (py>=3.11); falls
    back cleanly on older runtimes."""
    workers = os.cpu_count() or 4
    try:
        recycle = int(os.environ.get("DECODE_POOL_RECYCLE", "200") or "200")
    except Exception:
        recycle = 200
    try:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, max_tasks_per_child=max(1, recycle)
        )
    except TypeError:
        # Python < 3.11: max_tasks_per_child unsupported.
        return concurrent.futures.ProcessPoolExecutor(max_workers=workers)


async def _decode_image_bytes_offloaded(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes on the CPU pool. If a child died (OOM ->
    BrokenProcessPool) rebuild the pool once and retry, instead of returning
    500 forever until the container is restarted."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(app.state.executor, _decode_image_bytes, image_bytes)
    except concurrent.futures.process.BrokenProcessPool:
        try:
            app.state.executor = _build_decode_pool()
        except Exception:
            pass
        return await loop.run_in_executor(app.state.executor, _decode_image_bytes, image_bytes)


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
        # Use detect_all to get all faces. A faceless image is a VALID input
        # (client sent a photo with no detectable face), not a server error:
        # return an empty result so the handler responds 200 with
        # total_quality="fail" instead of crashing with 500.
        try:
            faces = embedder.detect_all(bgr)
        except ValueError as e:
            if "no face" in str(e).lower():
                timing = {"detect_ms": int(max(0.0, (_t() - t0)) * 1000.0)}
                return [], {"timing": timing, "reason": "no_face_detected"}, bgr.copy()
            raise
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


def _qdrant_count_subject_embeddings(client, collection: str, subject_id: str, access_key: str = "standard") -> int:
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
            count_filter=Filter(must=[
                FieldCondition(key="subject_id", match=MatchValue(value=subject_id)),
                FieldCondition(key="access_key", match=MatchValue(value=access_key))
            ]),
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


def _recognition_logging_enabled() -> bool:
    val = str(os.environ.get("RECOGNITION_LOGGING_ENABLED", "1")).strip().lower()
    enabled = val not in ("0", "false", "no", "off")
    return enabled

def _qdrant_search(
    client, 
    collection: str, 
    emb: np.ndarray, 
    top_k: int, 
    branch: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    access_key: str = "standard",
) -> list[dict[str, Any]]:
    t0 = _pc()
    try:
        search_params = None
        try:
            from qdrant_client.http.models import SearchParams, Filter, FieldCondition, MatchValue, Range, DatetimeRange
        except Exception:
            SearchParams = None  # type: ignore

        must_filters = [FieldCondition(key="access_key", match=MatchValue(value=access_key))]
        if branch:
            must_filters.append(FieldCondition(key="branch", match=MatchValue(value=branch)))
        
        if since_ts or until_ts:
            r_kwargs = {}
            if since_ts:
                r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
            if until_ts:
                r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
            must_filters.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))
        
        search_filter = Filter(must=must_filters)

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

        must_filters = []
        if branch:
            must_filters.append(FieldCondition(key="branch", match=MatchValue(value=branch)))
        
        if since_ts or until_ts:
            r_kwargs = {}
            if since_ts:
                r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
            if until_ts:
                r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
            must_filters.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))

        search_filter = None
        if must_filters:
            search_filter = Filter(must=must_filters)

        kwargs = dict(
            collection_name=collection,
            query_vector=emb.astype(np.float32).reshape(-1).tolist(),
            limit=int(top_k),
            with_payload=["subject_id", "image_id", "thumb_path", "branch", "created_at"],
            query_filter=search_filter
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


def _qdrant_list_subjects(client, collection: str, limit: int = 5000, access_key: str = "standard") -> list[str]:
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        points, _ = client.scroll(
            collection_name=collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(must=[FieldCondition(key="access_key", match=MatchValue(value=access_key))])
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key", "Authorization", "Cache-Control", "Accept", "Origin", "X-Requested-With"],
)

@app.middleware("http")
async def rewrite_api_prefix(request, call_next):
    # Support /api prefix by stripping it from the path
    if request.url.path.startswith("/api/"):
        request.scope["path"] = request.url.path[4:]
    return await call_next(request)

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
    try:
        response = await call_next(request)
    except RuntimeError as e:
        # Starlette BaseHTTPMiddleware raises "No response returned." when the
        # client disconnects mid-request (slow upload, Postman cancel/retry,
        # proxy timeout). Benign — the peer is already gone. Swallow quietly
        # instead of dumping a multi-frame traceback per aborted request.
        if "No response returned" in str(e):
            return Response(status_code=499)
        raise
    try:
        _REQ_LAT.labels(endpoint=path).observe(max(0.0, _t() - t0))
    except Exception:
        pass
    return response


# --- Load-shedding concurrency gate -----------------------------------------
# Bound the number of in-flight *heavy* (image-decoding / GPU) requests. When
# the cap is hit we reject immediately with 503 instead of accepting unbounded
# work that decodes big images into RAM and floods the GPU queue -> OOM SIGKILL.
# Cheap/monitoring paths (/health, /metrics, static, GET list endpoints) are
# never gated. Registered last -> runs outermost -> sheds before any real work.
_INFLIGHT_MAX = max(1, int(os.environ.get("FACE_SERVICE_MAX_INFLIGHT", "32") or "32"))
# Max requests allowed to WAIT for a processing slot before we shed (protects
# memory/event-loop from an unbounded backlog). Beyond this depth -> 503.
_MAX_QUEUE = max(1, int(os.environ.get("FACE_SERVICE_MAX_QUEUE", "512") or "512"))
# How long a queued request may wait for a slot before we give up with 503.
_QUEUE_WAIT_SEC = float(os.environ.get("FACE_SERVICE_QUEUE_WAIT_SEC", "60") or "60")
_inflight_count = 0
# Path markers for handlers that decode an image / touch the GPU. Matched even
# with the optional /api prefix still attached (gate runs before prefix strip).
_HEAVY_MARKERS = ("/faces", "/face/", "/quality/", "/events/recognition")


def _is_heavy_request(request) -> bool:
    if request.method not in ("POST", "PUT"):
        return False
    p = request.url.path
    if not any(m in p for m in _HEAVY_MARKERS):
        return False
    # GET-style sub-resources are cheap; only the base decoding POSTs are heavy.
    # Feedback/forward sub-paths carry no image -> exempt.
    if p.endswith("/feedback") or p.endswith("/forward"):
        return False
    return True


class _ConcurrencyGateASGI:
    """Leak-proof heavy-request concurrency gate with a bounded wait queue.

    Pure ASGI middleware (NOT BaseHTTPMiddleware) so the slot release ALWAYS
    runs on client disconnect/cancel -- BaseHTTPMiddleware orphaned its finally
    and leaked the counter until every heavy request 503'd until restart.

    Behaviour (queue-and-process, don't drop):
      - At most `_INFLIGHT_MAX` heavy requests run concurrently (caps GPU/GIL
        contention so each stays fast).
      - Excess requests WAIT in FIFO for a free slot instead of being dropped,
        so a burst of events is fully processed (each returns its real result),
        just staggered.
      - A request is shed with 503 ONLY under true overload: the wait queue is
        already `_MAX_QUEUE` deep, or a request waits longer than
        `_QUEUE_WAIT_SEC`. That protects memory + the event loop.

    The GET path and non-heavy POSTs bypass the gate entirely and stay instant.
    """

    def __init__(self, app):
        self.app = app
        self._sem: "asyncio.Semaphore | None" = None
        self._waiters = 0

    @staticmethod
    def _is_heavy_scope(scope) -> bool:
        if scope.get("method") not in ("POST", "PUT"):
            return False
        p = scope.get("path", "") or ""
        if not any(m in p for m in _HEAVY_MARKERS):
            return False
        # Feedback/forward sub-paths carry no image -> exempt.
        if p.endswith("/feedback") or p.endswith("/forward"):
            return False
        return True

    def _get_sem(self) -> "asyncio.Semaphore":
        # Lazily created on the running loop (single event loop).
        if self._sem is None:
            self._sem = asyncio.Semaphore(_INFLIGHT_MAX)
        return self._sem

    async def _shed(self, scope, receive, send, detail: str):
        try:
            _REQ_SHED_TOTAL.inc()
        except Exception:
            pass
        resp = JSONResponse(
            status_code=503,
            content={"detail": detail},
            headers={"Retry-After": "1"},
        )
        await resp(scope, receive, send)

    async def __call__(self, scope, receive, send):
        global _inflight_count
        if scope.get("type") != "http" or not self._is_heavy_scope(scope):
            return await self.app(scope, receive, send)

        # Stamp arrival so the handler can report queue-wait (time spent waiting
        # for a slot, BEFORE the model runs) in a response header.
        scope["_gate_enter"] = time.perf_counter()

        # Overflow guard: refuse to grow the backlog without bound.
        if self._waiters >= _MAX_QUEUE:
            return await self._shed(scope, receive, send, "server overloaded, retry later")

        sem = self._get_sem()
        acquired = False
        self._waiters += 1
        try:
            await asyncio.wait_for(sem.acquire(), timeout=_QUEUE_WAIT_SEC)
            acquired = True
        except asyncio.TimeoutError:
            return await self._shed(scope, receive, send, "server busy (queue wait exceeded), retry later")
        finally:
            self._waiters -= 1

        # acquired == True here (timeout path returned above).
        _inflight_count += 1
        try:
            await self.app(scope, receive, send)
        finally:
            _inflight_count -= 1
            sem.release()


app.add_middleware(_ConcurrencyGateASGI)


@app.get("/metrics")
def metrics() -> Response:
    try:
        body = generate_latest()
    except Exception:
        body = b""
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)


@app.get("/robots.txt", include_in_schema=False)
def robots() -> Response:
    return Response("User-agent: *\nDisallow: /", media_type="text/plain")


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
from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error("422 Validation Error at %s: %s", request.url, exc.errors())
    # exc.errors() can embed the raw request input as bytes (e.g. a multipart
    # file sent to a JSON endpoint, or a malformed body). Plain json.dumps on
    # bytes raises "Object of type bytes is not JSON serializable", which
    # crashed THIS handler -> 500. jsonable_encoder + a bytes encoder keeps the
    # 422 response JSON-safe regardless of the offending input.
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {"detail": exc.errors(), "body": str(exc)},
            custom_encoder={bytes: lambda b: f"<{len(b)} bytes>"},
        ),
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
            batch_ms = int(os.environ.get("GPU_BATCH_WINDOW_MS", "10") or "10")
        except Exception:
            batch_ms = 10

        try:
            num_workers = int(os.environ.get("GPU_NUM_WORKERS", "4") or "4")
        except Exception:
            num_workers = 4

        app.state.gpu = GPUInferenceManager(
            embedder=app.state.embedder,
            max_queue=max_q,
            batch_window_ms=batch_ms,
            num_workers=num_workers,
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

    # CPU-bound task offloading (e.g., image decoding)
    app.state.executor = _build_decode_pool()

    # Dedicated thread pool for model inference (embedding/quality). The ONNX
    # session can't be shipped to a process pool, but ort releases the GIL so a
    # small thread pool keeps the event loop free without thrashing a single GPU.
    app.state.embed_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ.get("EMBED_THREADS", "2") or 2),
        thread_name_prefix="embed",
    )

    # Starvation fix: GPU-manager submits (asyncio.to_thread -> detect_*_timed)
    # park a worker thread on threading.Event (done.wait) for the ENTIRE GPU
    # queue-wait + exec. The default asyncio.to_thread pool is only
    # min(32, cpu+4) threads, so under a burst those slots fill with parked
    # waiters and newly-admitted requests block waiting for a *thread* before
    # they can even enqueue GPU work -- adding seconds of latency that is not
    # GPU compute. Parked-on-Event threads are cheap (no GIL/CPU while waiting),
    # so size the default executor comfortably above the inflight cap.
    try:
        _tt_workers = max(64, _INFLIGHT_MAX * 2 + 16)
        asyncio.get_event_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=_tt_workers, thread_name_prefix="to-thread"
            )
        )
        logger.info("default to_thread executor sized to %d workers", _tt_workers)
    except Exception as e:
        logger.warning("could not resize default executor: %s", str(e))

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
            dim = 512 # Buffalo-L embedding dimension
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

    # Warm the API-key registry (master-managed tenant keys).
    try:
        _reload_key_registry(force=True)
        logger.info("api key registry loaded: %d keys", len(_key_registry))
    except Exception as e:
        logger.warning("api key registry warm-up failed: %s", str(e))


@app.on_event("shutdown")
def shutdown_event():
    if hasattr(app.state, "gpu") and app.state.gpu:
        app.state.gpu.close()
    if hasattr(app.state, "executor") and app.state.executor:
        app.state.executor.shutdown(wait=True)
    if hasattr(app.state, "embed_executor") and app.state.embed_executor:
        app.state.embed_executor.shutdown(wait=True)
        


@app.post("/v1/face/search", response_model=FaceSearchResponse, dependencies=[Depends(get_api_key)])
def face_search(
    req: FaceSearchRequest,
    access_key: str = Depends(get_api_key),
) -> FaceSearchResponse:
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
        results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1, access_key=access_key)
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


# ===================== API key registry (master-managed) =====================
# Master key = FACE_SERVICE_LEGACY_API_KEY. It manages tenant keys stored in a
# Qdrant collection and auto-reloaded into an in-memory registry. Each tenant
# key resolves to its own access_key bucket, so every other endpoint scopes its
# data automatically via get_api_key / get_optional_access_key.

_KEYS_COLLECTION = os.environ.get("KEYS_COLLECTION", "api_keys")
_KEY_RELOAD_SEC = float(os.environ.get("KEY_RELOAD_SEC", "15"))
_key_registry: dict[str, dict[str, Any]] = {}
_key_registry_ts: float = 0.0


def _master_key() -> str:
    return str(_get_legacy_api_key() or "").strip()


def _mask_key(raw: str) -> str:
    s = str(raw or "")
    return "****" if len(s) <= 8 else f"{s[:6]}…{s[-4:]}"


def _key_point_id(raw: str) -> str:
    return _uuid5_from_name(f"apikey:{raw}")


def _keys_qdrant():
    return getattr(app.state, "qdrant", None)


def _ensure_keys_collection(client) -> None:
    _ensure_qdrant_collection(client, _KEYS_COLLECTION, 1)


def _scroll_key_payloads(client) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    _ensure_keys_collection(client)
    next_off = None
    while True:
        batch, next_off = client.scroll(
            collection_name=_KEYS_COLLECTION, limit=1000,
            with_payload=True, with_vectors=False, offset=next_off,
        )
        for pnt in batch or []:
            out.append(getattr(pnt, "payload", None) or {})
        if not next_off:
            break
    return out


def _reload_key_registry(force: bool = False) -> None:
    global _key_registry, _key_registry_ts
    now = time.time()
    if not force and (now - _key_registry_ts) < _KEY_RELOAD_SEC:
        return
    client = _keys_qdrant()
    if client is None:
        _key_registry_ts = now
        return
    reg: dict[str, dict[str, Any]] = {}
    try:
        for p in _scroll_key_payloads(client):
            raw = str(p.get("key") or "")
            if not raw or not bool(p.get("active", True)):
                continue
            reg[raw] = {
                "key_id": str(p.get("key_id") or ""),
                "access_key": str(p.get("access_key") or ""),
                "name": p.get("name"),
                "active": bool(p.get("active", True)),
                "created_at": p.get("created_at"),
            }
        _key_registry = reg
        _key_registry_ts = now
    except Exception as e:
        logger.warning(f"key registry reload failed: {e}")


def _registry_lookup(raw: str) -> str | None:
    _reload_key_registry(False)
    info = _key_registry.get(str(raw or "").strip())
    return info["access_key"] if info else None


async def require_master_key(header_value: str = Security(api_key_header)):
    master = _master_key()
    if not master:
        raise HTTPException(status_code=503, detail="master key (legacy_api_key) not configured")
    if str(header_value or "").strip() != master:
        raise HTTPException(status_code=403, detail="master key required")
    return True


@app.post("/v1/keys", response_model=KeyInfo, dependencies=[Depends(require_master_key)])
def create_key(req: KeyCreateRequest) -> KeyInfo:
    client = _keys_qdrant()
    if client is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    raw = str(req.api_key or "").strip() or ("fs_" + uuid.uuid4().hex + uuid.uuid4().hex[:8])
    reserved = {str(_get_api_key()).strip(), _master_key()}
    if raw in reserved:
        raise HTTPException(status_code=400, detail="key conflicts with a reserved key")
    access_key = "t_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    kid = _key_point_id(raw)
    rec = {
        "key_id": kid, "key": raw, "name": req.name, "access_key": access_key,
        "active": True, "created_at": _iso_now(),
    }
    try:
        from qdrant_client.http.models import PointStruct
        _ensure_keys_collection(client)
        client.upsert(
            collection_name=_KEYS_COLLECTION,
            points=[PointStruct(id=kid, vector=[0.0], payload=rec)],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"key store failed: {str(e)}")
    _reload_key_registry(force=True)
    # Raw key returned ONCE here; subsequent listings only show a masked value.
    return KeyInfo(
        key_id=kid, name=req.name, access_key=access_key, created_at=rec["created_at"],
        active=True, api_key=raw, api_key_masked=_mask_key(raw),
    )


@app.get("/v1/keys", response_model=KeyListResponse, dependencies=[Depends(require_master_key)])
def list_keys() -> KeyListResponse:
    client = _keys_qdrant()
    if client is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    try:
        recs = _scroll_key_payloads(client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"key list failed: {str(e)}")
    keys = [
        KeyInfo(
            key_id=str(r.get("key_id") or ""), name=r.get("name"),
            access_key=str(r.get("access_key") or ""), created_at=r.get("created_at"),
            active=bool(r.get("active", True)), api_key_masked=_mask_key(str(r.get("key") or "")),
        )
        for r in recs
    ]
    return KeyListResponse(keys=keys)


@app.delete("/v1/keys/{key_id}", response_model=KeyDeleteResponse, dependencies=[Depends(require_master_key)])
def delete_key(key_id: str) -> KeyDeleteResponse:
    client = _keys_qdrant()
    if client is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    ok = False
    try:
        from qdrant_client.http.models import PointIdsList
        _ensure_keys_collection(client)
        client.delete(collection_name=_KEYS_COLLECTION, points_selector=PointIdsList(points=[str(key_id)]))
        ok = True
    except Exception as e:
        logger.error(f"delete key failed: {str(e)}")
    _reload_key_registry(force=True)
    return KeyDeleteResponse(key_id=str(key_id), deleted=ok)


@app.post("/v1/groups", response_model=GroupResponse, dependencies=[Depends(get_api_key)])
async def create_group(
    req: GroupCreateRequest,
    access_key: str = Depends(get_api_key),
) -> GroupResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    group_collection = os.environ.get("QDRANT_GROUPS_COLLECTION", "face_groups")
    try:
        from qdrant_client.http.models import Distance, VectorParams
        if not q.collection_exists(group_collection):
            q.create_collection(
                collection_name=group_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE), # Dummy vector
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to ensure groups collection: {str(e)}")

    try:
        from qdrant_client.http.models import PointStruct
        q.upsert(
            collection_name=group_collection,
            points=[
                PointStruct(
                    id=_uuid5_from_name(req.group_id),
                    vector=[0.0],
                    payload={
                        "group_id": req.group_id,
                        "name": req.name or req.group_id,
                        "meta": req.meta or {},
                        "access_key": access_key,
                        "created_at": _iso_now()
                    }
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to create group: {str(e)}")

    return GroupResponse(group_id=req.group_id, name=req.name, meta=req.meta)

@app.get("/v1/groups", response_model=GroupListResponse, dependencies=[Depends(get_api_key)])
async def list_groups(access_key: str = Depends(get_optional_access_key)) -> GroupListResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    group_collection = os.environ.get("QDRANT_GROUPS_COLLECTION", "face_groups")
    try:
        if not q.collection_exists(group_collection):
            return GroupListResponse(groups=[])
        
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        points, _ = q.scroll(
            collection_name=group_collection, 
            limit=100, 
            with_payload=True,
            scroll_filter=Filter(must=[FieldCondition(key="access_key", match=MatchValue(value=access_key))])
        )
        groups = []
        for p in points:
            payload = p.payload or {}
            groups.append(GroupResponse(
                group_id=payload.get("group_id", ""),
                name=payload.get("name"),
                meta=payload.get("meta")
            ))
        return GroupListResponse(groups=groups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to list groups: {str(e)}")

@app.delete("/v1/groups/{group_id}", dependencies=[Depends(get_api_key)])
async def delete_group(
    group_id: str,
    access_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    group_collection = os.environ.get("QDRANT_GROUPS_COLLECTION", "face_groups")
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        q.delete(
            collection_name=group_collection,
            points_selector=Filter(must=[
                FieldCondition(key="group_id", match=MatchValue(value=group_id)),
                FieldCondition(key="access_key", match=MatchValue(value=access_key))
            ])
        )
        return {"deleted": True, "group_id": group_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to delete group: {str(e)}")


# Cache of validated branch ids -> monotonic expiry. Branches change rarely,
# so skip the two per-request Qdrant round-trips once a branch is confirmed.
_BRANCH_OK_CACHE: dict[str, float] = {}
_BRANCH_CACHE_TTL = float(os.environ.get("BRANCH_CACHE_TTL", "300") or 300)


def _ensure_branch_exists(q, branch_id: str | None, access_key: str = "standard") -> None:
    if not branch_id:
        return
    bid = str(branch_id).strip()
    if not bid:
        return

    exp = _BRANCH_OK_CACHE.get(bid)
    if exp is not None and exp > _pc():
        return

    branch_collection = os.environ.get("QDRANT_BRANCHES_COLLECTION", "face_branches")
    try:
        if not q.collection_exists(branch_collection):
             raise HTTPException(status_code=400, detail=f"branch '{bid}' not found (no branches registered)")

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        res, _ = q.scroll(
            collection_name=branch_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="branch_id", match=MatchValue(value=bid)),
                FieldCondition(key="access_key", match=MatchValue(value=access_key))
            ]),
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        if not res:
            raise HTTPException(status_code=400, detail=f"branch '{bid}' is not available for your access key. please create it first.")
        _BRANCH_OK_CACHE[bid] = _pc() + _BRANCH_CACHE_TTL
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"failed to verify branch existence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"error verifying branch '{bid}'")


@app.post("/v1/branches", response_model=BranchResponse, dependencies=[Depends(get_api_key)])
async def create_branch(
    req: BranchCreateRequest,
    access_key: str = Depends(get_api_key),
) -> BranchResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    branch_collection = os.environ.get("QDRANT_BRANCHES_COLLECTION", "face_branches")
    try:
        from qdrant_client.http.models import Distance, VectorParams
        if not q.collection_exists(branch_collection):
            q.create_collection(
                collection_name=branch_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE), # Dummy vector
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to ensure branches collection: {str(e)}")

    try:
        from qdrant_client.http.models import PointStruct
        q.upsert(
            collection_name=branch_collection,
            points=[
                PointStruct(
                    id=_uuid5_from_name(req.branch_id),
                    vector=[0.0],
                    payload={
                        "branch_id": req.branch_id,
                        "name": req.name or req.branch_id,
                        "meta": req.meta or {},
                        "access_key": access_key,
                        "created_at": _iso_now()
                    }
                )
            ]
        )
        _BRANCH_OK_CACHE[str(req.branch_id).strip()] = _pc() + _BRANCH_CACHE_TTL
        return BranchResponse(
            branch_id=req.branch_id,
            name=req.name or req.branch_id,
            meta=req.meta or {}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to create branch: {str(e)}")


@app.get("/v1/branches", response_model=BranchListResponse, dependencies=[Depends(get_api_key)])
async def list_branches(access_key: str = Depends(get_optional_access_key)) -> BranchListResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    branch_collection = os.environ.get("QDRANT_BRANCHES_COLLECTION", "face_branches")
    try:
        if not q.collection_exists(branch_collection):
            return BranchListResponse(branches=[])
        
        branches = []
        # Scroll all points from branch collection
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        res, _ = q.scroll(
            collection_name=branch_collection, 
            limit=1000, 
            with_payload=True, 
            with_vectors=False,
            scroll_filter=Filter(must=[FieldCondition(key="access_key", match=MatchValue(value=access_key))])
        )
        
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        except Exception:
            Filter = None

        for p in res:
            pl = p.payload or {}
            bid = str(pl.get("branch_id") or "")
            
            e_count = 0
            s_count = 0
            if Filter and bid:
                try:
                    # enrollment_count
                    cnt_res = q.count(
                        collection_name=app.state.qdrant_collection,
                        count_filter=Filter(must=[
                            FieldCondition(key="branch", match=MatchValue(value=bid)),
                            FieldCondition(key="access_key", match=MatchValue(value=access_key))
                        ]),
                        exact=True
                    )
                    e_count = int(getattr(cnt_res, "count", 0) or 0)

                    # subject_count
                    sids = set()
                    next_offset = None
                    while True:
                        batch, next_offset = q.scroll(
                            collection_name=app.state.qdrant_collection,
                            scroll_filter=Filter(must=[
                                FieldCondition(key="branch", match=MatchValue(value=bid)),
                                FieldCondition(key="access_key", match=MatchValue(value=access_key))
                            ]),
                            limit=1000,
                            with_payload=["subject_id"],
                            with_vectors=False,
                            offset=next_offset
                        )
                        for b in batch:
                            sid = (b.payload or {}).get("subject_id")
                            if sid: sids.add(sid)
                        if not next_offset: break
                    s_count = len(sids)
                except Exception:
                    pass

            branches.append(BranchResponse(
                branch_id=bid,
                name=str(pl.get("name") or ""),
                meta=dict(pl.get("meta") or {}),
                enrollment_count=e_count,
                subject_count=s_count
            ))

        # Also surface branches that exist only as a payload value on enrolled
        # data for this key (no Branch entity was ever created). Lets the UI
        # offer real branches to filter by without requiring POST /v1/branches.
        try:
            known = {str(b.branch_id) for b in branches}
            if Filter:
                next_off = None
                scanned = 0
                while scanned < 20000:  # cap scan for safety
                    batch, next_off = q.scroll(
                        collection_name=app.state.qdrant_collection,
                        scroll_filter=Filter(must=[FieldCondition(key="access_key", match=MatchValue(value=access_key))]),
                        limit=1000,
                        with_payload=["branch"],
                        with_vectors=False,
                        offset=next_off,
                    )
                    for b in batch or []:
                        bid = str((b.payload or {}).get("branch") or "").strip()
                        if bid and bid not in known:
                            known.add(bid)
                            branches.append(BranchResponse(branch_id=bid, name=bid, meta={}))
                    scanned += len(batch or [])
                    if not next_off:
                        break
        except Exception:
            pass

        return BranchListResponse(branches=branches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to list branches: {str(e)}")


@app.delete("/v1/branches/{branch_id}", dependencies=[Depends(get_api_key)])
async def delete_branch(
    branch_id: str,
    access_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    
    branch_collection = os.environ.get("QDRANT_BRANCHES_COLLECTION", "face_branches")
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        q.delete(
            collection_name=branch_collection,
            points_selector=Filter(must=[
                FieldCondition(key="branch_id", match=MatchValue(value=branch_id)),
                FieldCondition(key="access_key", match=MatchValue(value=access_key))
            ])
        )
        _BRANCH_OK_CACHE.pop(str(branch_id).strip(), None)
        return {"deleted": True, "branch_id": branch_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to delete branch: {str(e)}")


@app.post("/v1/faces/add", response_model=FaceAddResponse, dependencies=[Depends(get_api_key)])
def faces_add(
    req: FaceAddRequest,
    access_key: str = Depends(get_api_key),
) -> FaceAddResponse:
    if not req.subject_id or not str(req.subject_id).strip():
        raise HTTPException(status_code=400, detail="subject_id is required")

    # Merge images_b64 and image_urls into a single list
    all_images: list[str] = list(req.images_b64 or []) + list(req.image_urls or [])
    if not all_images:
        raise HTTPException(status_code=400, detail="images_b64 or image_urls must be non-empty")

    subject_id = str(req.subject_id).strip()
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    cap = _subject_embedding_cap()
    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id, access_key=access_key)
    if existing >= cap:
        raise HTTPException(status_code=409, detail=f"subject embedding cap reached ({existing}/{cap})")

    num_embedded = 0
    emb_dim: int | None = None
    last_meta: dict[str, Any] | None = None
    
    # Use provided date or default to now
    enroll_ts = _normalize_created_at(req.created_at, req.ts)

    for i, img_b64 in enumerate(all_images):
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
                hits = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1, access_key=access_key)
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
                            "branch": req.branch,
                            "access_key": access_key,
                            "image_id": image_id,
                            "created_at": enroll_ts,
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


@app.post("/v1/faces/add_upload", response_model=FaceAddResponse, dependencies=[Depends(get_api_key)])
async def faces_add_upload(
    subject_id: str = Form(...),
    branch: str | None = Form(None),
    created_at: str | None = Form(None),
    ts: float | None = Form(None),
    files: list[UploadFile] = File(default=[]),
    image_urls: list[str] = Form(default=[]),
    access_key: str = Depends(get_api_key),
) -> FaceAddResponse:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    if not files and not image_urls:
        raise HTTPException(status_code=400, detail="files or image_urls must be non-empty")

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    cap = _subject_embedding_cap()
    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id, access_key=access_key)
    if existing >= cap:
        raise HTTPException(status_code=409, detail=f"subject embedding cap reached ({existing}/{cap})")

    num_embedded = 0
    emb_dim: int | None = None
    last_meta: dict[str, Any] | None = None
    
    # Use provided date or default to now
    enroll_ts = _normalize_created_at(created_at, ts)

    for i, f in enumerate(files):
        if existing >= cap:
            break
        image_bytes = await f.read()
        bgr = await _decode_image_bytes_offloaded(image_bytes)
        try:
            emb, meta = await _quality_check_and_embed_async(bgr)
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
                hits = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1, access_key=access_key)
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
                            "branch": branch,
                            "access_key": access_key,
                            "image_id": image_id,
                            "created_at": enroll_ts,
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

    # Process image URLs
    for i, url in enumerate(image_urls):
        if existing >= cap:
            break
        url = str(url or "").strip()
        if not url:
            continue
        image_bytes = await asyncio.to_thread(_decode_b64_bytes, url)
        bgr = await _decode_image_bytes_offloaded(image_bytes)
        try:
            emb, meta = await _quality_check_and_embed_async(bgr)
        except HTTPException as e:
            _debug(f"add_skip subject_id={subject_id} url_idx={i} reason={e.detail}")
            continue
        last_meta = meta
        emb_dim = int(emb.reshape(-1).shape[0])
        try:
            _ensure_qdrant_collection(q, app.state.qdrant_collection, vector_size=emb_dim)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant init failed: {str(e)}")

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_id = image_hash[:16]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{subject_id}:{image_hash}"))
        thumb_path = _save_thumb(bgr, app.state.thumbs_dir, image_id)
        image_path = _save_image(bgr, os.environ.get("IMAGES_DIR", "/data/images"), subject_id, image_id)

        if _enroll_dup_check_enabled():
            try:
                thr = float(_enroll_dup_min_similarity())
                hits = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=1, access_key=access_key)
                if hits:
                    best = hits[0]
                    best_sid = str(best.get("subject_id") or "").strip()
                    best_sim = float(best.get("similarity") or 0.0)
                    if best_sid and best_sid != subject_id and best_sim >= thr:
                        extra = {
                            "matched_subject_id": best_sid,
                            "similarity": float(best_sim),
                            "threshold": float(thr),
                            "source_url": url,
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
                            "branch": branch,
                            "access_key": access_key,
                            "image_id": image_id,
                            "created_at": enroll_ts,
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
        num_images=len(files) + len(image_urls),
        num_embedded=num_embedded,
        embedding_dim=emb_dim,
        meta=last_meta,
    )


@app.post("/v1/faces/search", response_model=FaceSearchTopKResponse)
def faces_search(
    req: FaceSearchTopKRequest,
    access_key: str = Depends(get_api_key)
) -> FaceSearchTopKResponse:
    top_k = int(req.top_k or 5)
    top_k = max(1, min(top_k, 50))

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    s_ts, e_ts = _date_window_from_params(
        day=req.day,
        from_day=req.from_day,
        to_day=req.to_day,
        since_ts=req.since_ts,
        until_ts=req.until_ts
    )

    bgr = _decode_b64_image(req.image_b64)
    emb, _meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k, branch=req.branch, since_ts=s_ts, until_ts=e_ts, access_key=access_key)

    thumb_path = ""
    try:

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
                access_key=access_key,
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


@app.post("/v1/faces/search_upload", response_model=FaceSearchTopKResponse, dependencies=[Depends(get_api_key)])
async def faces_search_upload(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    branch: Optional[str] = Form(None),
    day: Optional[str] = Form(None),
    from_day: Optional[str] = Form(None),
    to_day: Optional[str] = Form(None),
    since_ts: Optional[float] = Form(None),
    until_ts: Optional[float] = Form(None),
    access_key: str = Depends(get_api_key),
) -> FaceSearchTopKResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    s_ts, e_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts
    )

    top_k = int(top_k or 5)
    top_k = max(1, min(top_k, 50))
    image_bytes = await file.read()
    bgr = await _decode_image_bytes_offloaded(image_bytes)
    emb, _meta = await _quality_check_and_embed_async(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k, branch=branch, since_ts=s_ts, until_ts=e_ts, access_key=access_key)

    thumb_path = None
    # Persistent Logging
    try:
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
            access_key=access_key,
        )
        if app.state.events:
            app.state.events.insert_search_event(ev)
            logger.info("Search Event Logged: %s", event_id)
    except Exception as e:
        logger.error("failed to log search event: %s", str(e))

    _record_event(app.state.search_events)
    return FaceSearchTopKResponse(results=[FaceSearchTopKItem(**r) for r in results], query_thumb_path=thumb_path or None)


@app.post("/v1/faces/recognize", response_model=FaceRecognizeResponse)
def faces_recognize(
    req: FaceRecognizeRequest,
    access_key: str = Depends(get_api_key)
) -> FaceRecognizeResponse:
    top_k = int(req.top_k or 5)
    top_k = max(1, min(top_k, 50))
    min_sim = float(req.min_similarity) if req.min_similarity is not None else float(app.state.min_similarity)

    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    _ensure_branch_exists(q, req.branch, access_key=access_key)

    s_ts, e_ts = _date_window_from_params(
        day=req.day,
        from_day=req.from_day,
        to_day=req.to_day,
        since_ts=req.since_ts,
        until_ts=req.until_ts
    )

    bgr = _decode_b64_image(req.image_b64)
    emb, meta = _quality_check_and_embed(bgr)
    results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k, branch=req.branch, since_ts=s_ts, until_ts=e_ts, access_key=access_key)
    items = [FaceSearchTopKItem(**r) for r in results]

    # Persistent Logging
    try:
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
            meta=meta,
            access_key=access_key,
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
async def faces_cross_match(
    subject_id: str,
    top_k: int = 20,
    access_key: str = Depends(get_optional_access_key),
) -> FaceSearchTopKResponse:
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
                scroll_filter=Filter(must=[
                    FieldCondition(key="subject_id", match=MatchValue(value=subject_id)),
                    FieldCondition(key="access_key", match=MatchValue(value=access_key))
                ]),
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
    hits = _qdrant_search(q, app.state.qdrant_collection, vector, top_k=100, access_key=access_key)
    
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
    camera: str = Form(""),
    source_path: str = Form(""),
    ts: float | None = Form(None),
    top_k: int = Form(5),
    min_similarity: float | None = Form(None),
    process_all_faces: bool = Form(False),
    branch: str | None = Form(None),
    access_key: str = Depends(get_optional_access_key),
) -> RecognitionEventResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    camera = str(camera or "").strip()
    branch = str(branch).strip() if branch else None

    t_req0 = _t()
    t_total0 = _pc()
    t_decode0 = _pc()
    image_bytes = await file.read()
    bgr = await _decode_image_bytes_offloaded(image_bytes)
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

        # Detection + embedding blocks (GPU queue wait can be seconds). Run it on
        # a worker thread so the event loop stays free to serve /health etc.
        def _run_detect() -> tuple[list[Any], dict[str, Any]]:
            _timing: dict[str, Any] = {}
            if process_all_faces:
                if hasattr(infer, "detect_all_timed"):
                    _faces, tinfo = infer.detect_all_timed(bgr)
                    _timing = dict(tinfo or {})
                else:
                    _faces = list(infer.detect_all(bgr))
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
                        _timing = dict(tinfo or {})
                        _faces = [_pick_best_face_by_area(list(faces_all or []))]
                    else:
                        _faces = [
                            _pick_best_face_by_area(list(infer.detect_all(bgr)))
                            if hasattr(infer, "detect_all")
                            else infer.detect_best(bgr)
                        ]
                else:
                    if hasattr(infer, "detect_best_timed"):
                        face0, tinfo = infer.detect_best_timed(bgr)
                        _faces = [face0]
                        _timing = dict(tinfo or {})
                    else:
                        _faces = [infer.detect_best(bgr)]
            return _faces, _timing

        faces, infer_timing = await asyncio.to_thread(_run_detect)
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
                    branch=branch,
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
                    access_key=access_key,
                )
            )
            return RecognitionEventResponse(
                event_id=event_id,
                ts=ts_val,
                camera=camera,
                branch=branch,
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
                emb = await asyncio.to_thread(_embed_from_face, face)
                if emb is not None:
                    try:
                        quality_meta = await asyncio.to_thread(evaluator.evaluate, bgr, face)
                    except Exception:
                        quality_meta = None
                    if isinstance(quality_meta, dict) and quality_meta.get("status") == "rejected":
                        raise ValueError(f"quality_reject:{str(quality_meta.get('reason') or 'unknown')}")
                else:
                    bgr_face = _crop_face(bgr, face)
                    emb, quality_meta = await _quality_check_and_embed_async(bgr_face)
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
                        branch=branch,
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
                        access_key=access_key,
                    )
                )
                faces_processed += 1
                if primary_resp is None:
                    primary_resp = RecognitionEventResponse(
                        event_id=ev_id,
                        ts=ts_val,
                        camera=camera,
                        branch=branch,
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
                        access_key=access_key,
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
                    branch=branch,
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
                    access_key=access_key,
                )
            )
            faces_processed += 1
            if primary_resp is None:
                primary_resp = RecognitionEventResponse(
                    event_id=ev_id,
                    ts=ts_val,
                    camera=camera,
                    branch=branch,
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
                    access_key=access_key,
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
        results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=top_k, branch=branch, access_key=access_key)
        
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
                                results = _qdrant_search(q, app.state.qdrant_collection, emb, top_k=5, access_key=access_key)
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
                                                "access_key": access_key,
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
                    existing = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id, access_key=access_key)
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
                                            "access_key": access_key,
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
                branch=branch,
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
                access_key=access_key,
            )
        )

        faces_processed += 1
        if primary_resp is None:
            primary_resp = RecognitionEventResponse(
                event_id=ev_id,
                ts=ts_val,
                camera=camera,
                branch=branch,
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
    branch: str | None = None,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    limit: int = 100,
    cursor: float | None = None,
    access_key: str = Depends(get_optional_access_key),
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
        branch=branch,
        limit=limit,
        cursor_ts=cursor,
        access_key=access_key,
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
    access_key: str = Depends(get_optional_access_key),
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
        access_key=access_key,
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
    access_key: str = Depends(get_optional_access_key),
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
        access_key=access_key,
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
def list_recognition_cameras(
    limit: int = 5000,
    access_key: str = Depends(get_optional_access_key),
) -> list[str]:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    out: set[str] = set()
    try:
        for c in store.list_cameras(limit=int(limit or 5000), access_key=access_key):
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
async def forward_recognition_event(
    req: ForwardEventRequest,
    access_key: str = Depends(get_optional_access_key),
) -> dict[str, Any]:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    event_id = str(req.event_id or "").strip()
    if not event_id:
        raise HTTPException(status_code=400, detail="event_id is required")

    it = store.get_event(event_id, access_key=access_key)
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
    access_key: str = Depends(get_optional_access_key),
) -> FeedbackStatsResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    st = store.feedback_stats(since_ts=since_ts, until_ts=until_ts, camera=camera, access_key=access_key)
    return FeedbackStatsResponse(**st)


@app.get("/v1/events/recognition/stats", response_model=RecognitionStatsResponse)
def recognition_stats(
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    camera: str | None = None,
    access_key: str = Depends(get_optional_access_key),
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
        access_key=access_key,
    )
    return RecognitionStatsResponse(**res)


@app.get("/v1/events/recognition/{event_id}", response_model=RecognitionEventResponse)
def get_recognition_event(
    event_id: str,
    access_key: str = Depends(get_optional_access_key),
) -> RecognitionEventResponse:
    store: EventsStore | None = getattr(app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")
    it = store.get_event(event_id, access_key=access_key)
    if not it:
        raise HTTPException(status_code=404, detail="event not found")
    return RecognitionEventResponse(**it)


@app.post("/v1/events/recognition/{event_id}/feedback", response_model=EventFeedbackResponse)
def set_recognition_event_feedback(
    event_id: str,
    req: EventFeedbackRequest,
    access_key: str = Depends(get_optional_access_key),
) -> EventFeedbackResponse:
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
    updated = bool(store.set_feedback(event_id, label=label, note=note, updated_at=updated_at, access_key=access_key))
    if not updated:
        raise HTTPException(status_code=404, detail="event not found")

    return EventFeedbackResponse(
        event_id=event_id,
        updated=True,
        feedback_label=label,
        feedback_note=note,
        feedback_updated_at=updated_at,
    )


def _log_recognition_event_background(
    bgr: np.ndarray,
    results: list[dict[str, Any]],
    meta: dict[str, Any],
    subject_id: str | None,
    similarity: float | None,
    items: list[Any],
    events_store: Any,
    thumbs_dir: str,
    access_key: str = "standard",
    branch: str | None = None,
):
    logger.info(f"Background logging task started (RE_LOGGING={os.environ.get('RECOGNITION_LOGGING_ENABLED')})")
    if not _recognition_logging_enabled():
        logger.info("Background logging disabled via check")
        return

    try:
        event_id = str(uuid.uuid4())
        logger.info(f"Saving assets for event {event_id}")
        img_url, thumb_path = _save_search_query_assets(bgr, event_id)
        
        if not img_url:
            logger.warning(f"Failed to save assets for event {event_id}")

        top_sid = subject_id
        top_sim = similarity
        
        # 1. Save to Search History
        ev_search = SearchEvent(
            event_id=event_id,
            ts=_t(),
            query_image_path=img_url,
            query_thumb_path=thumb_path,
            top_subject_id=top_sid,
            top_similarity=top_sim,
            results=results,
            meta=meta,
            access_key=access_key,
        )
        if events_store:
            events_store.insert_search_event(ev_search)
            logger.info(f"Inserted search event {event_id}")

        # 2. Save to Recognition History (so it shows up in Recognition.tsx)
        decision = "no_match"
        if top_sid and top_sim is not None:
             # Logic to determine decision status
             # We don't have min_sim here easily, but we can infer from items/results
             # Or just mark it as match if we have a top_sid and it was matched in main thread
             # Actually, the items list might tell us if it was a match.
             # For now, let's just mark it based on presence.
             decision = "match" if top_sid else "no_match"

        ev_rec = RecognitionEvent(
            event_id=event_id,
            ts=_t(),
            camera="api_recognize_upload",
            branch=branch,
            source_path="api_upload",
            decision=decision,
            subject_id=top_sid,
            similarity=top_sim,
            processing_ms=None,
            model_ms=None,
            rejected_reason=None,
            bbox=None,
            det_score=None,
            image_path=img_url,
            thumb_path=thumb_path,
            image_saved_at=_t(),
            meta=meta,
            access_key=access_key,
        )
        if events_store:
            events_store.insert_event(ev_rec)
            logger.info(f"Inserted recognition event {event_id}")

    except Exception as e:
        logger.error("failed to log recognize upload event in background: %s", str(e))


@app.post("/v1/faces/recognize_upload", response_model=FaceRecognizeResponse, dependencies=[Depends(get_api_key)])
async def faces_recognize_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    top_k: int = Form(5),
    min_similarity: float | None = Form(None),
    branch: str | None = Form(None),
    day: Optional[str] = Form(None),
    from_day: Optional[str] = Form(None),
    to_day: Optional[str] = Form(None),
    since_ts: Optional[float] = Form(None),
    until_ts: Optional[float] = Form(None),
    access_key: str = Depends(get_api_key),
) -> FaceRecognizeResponse:
    t_start = _pc()
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    _ensure_branch_exists(q, branch, access_key=access_key)

    s_ts, e_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts
    )

    top_k = int(top_k or 5)
    top_k = max(1, min(top_k, 50))
    min_sim = float(min_similarity) if min_similarity is not None else float(app.state.min_similarity)

    image_bytes = await file.read()

    t_decode_start = _pc()
    bgr = await _decode_image_bytes_offloaded(image_bytes)
    if bgr is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    decode_ms = int((_pc() - t_decode_start) * 1000.0)

    loop = asyncio.get_event_loop()
    t_embed_start = _pc()
    # Run model inference off the event loop so concurrent camera traffic
    # doesn't block frontend API calls (stats/events) from being served.
    emb, meta = await loop.run_in_executor(app.state.embed_executor, _quality_check_and_embed, bgr)
    embed_ms = int((_pc() - t_embed_start) * 1000.0)

    t_search_start = _pc()
    # Execute Qdrant search in executor to avoid blocking the event loop
    results = await loop.run_in_executor(
        None,
        _qdrant_search,
        q,
        app.state.qdrant_collection,
        emb,
        top_k,
        branch,
        s_ts,
        e_ts,
        access_key
    )
    search_ms = int((_pc() - t_search_start) * 1000.0)    
    items = [FaceSearchTopKItem(**r) for r in results]

    # Enhanced timing metadata
    full_meta = (meta or {}).copy()
    timing = full_meta.get("timing", {})
    timing.update({
        "decode_ms": decode_ms,
        "model_ms": embed_ms, 
        "search_ms": search_ms,
        "total_ms": int((_pc() - t_start) * 1000.0)
    })
    full_meta["timing"] = timing

    # Offload logging to background task
    background_tasks.add_task(
        _log_recognition_event_background,
        bgr,
        results,
        meta,
        results[0]["subject_id"] if results else None,
        results[0]["similarity"] if results else None,
        items,
        app.state.events,
        app.state.search_thumbs_dir,
        access_key,
        branch
    )

    if not items:
        return FaceRecognizeResponse(matched=False, results=[], meta=full_meta)

    best = items[0]
    if float(best.similarity) >= float(min_sim) and str(best.subject_id).strip():
        return FaceRecognizeResponse(
            matched=True,
            subject_id=best.subject_id,
            similarity=float(best.similarity),
            results=items,
            meta=full_meta,
        )
    return FaceRecognizeResponse(matched=False, results=items, meta=full_meta)


@app.post("/v1/quality/check_upload", response_model=QualityCheckResponse)
async def quality_check_upload(
    file: UploadFile = File(...),
    annotate: bool = Form(default=True),
) -> QualityCheckResponse:
    t0 = _pc()
    image_bytes = await file.read()
    decode_t0 = _pc()
    bgr = await _decode_image_bytes_offloaded(image_bytes)
    decode_ms = int(max(0.0, (_pc() - decode_t0)) * 1000.0)

    q_t0 = _pc()
    results, meta, annotated = await asyncio.to_thread(_quality_check_all, bgr)
    quality_ms = int(max(0.0, (_pc() - q_t0)) * 1000.0)

    # The annotated JPEG (boxes drawn) is base64'd into the response — it is the
    # single biggest part of the payload (~30 KB). Callers that only need the
    # quality verdict can pass annotate=false to skip the encode + download.
    annotated_image = ""
    if annotate:
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    total_ok = all(r["ok"] for r in results) if results else False
    total_quality = "pass" if total_ok else "fail"

    total_ms = int(max(0.0, (_pc() - t0)) * 1000.0)
    timing = dict(meta.get("timing") or {})
    timing.update({"decode_ms": int(decode_ms), "quality_ms": int(quality_ms), "total_ms": int(total_ms)})

    return QualityCheckResponse(
        ok=bool(total_ok),
        total_quality=str(total_quality),
        faces=[FaceQualityResult(**r) for r in results],
        annotated_image=annotated_image,
        timing=timing,
    )


@app.post("/v1/faces/privacy_extract", response_model=PrivacyExtractResponse)
async def privacy_extract(
    req: PrivacyExtractRequest,
    request: Request,
    response: Response,
    access_key: str = Depends(get_optional_access_key),
) -> PrivacyExtractResponse:
    t0 = _pc()
    _tm = {}
    # queue-wait = handler start - gate arrival (time spent waiting for a slot).
    _gate_enter = request.scope.get("_gate_enter")
    _queue_wait_ms = int(max(0.0, (t0 - _gate_enter)) * 1000) if _gate_enter else 0
    bgr = await _decode_b64_image_async(req.image_b64)
    _tm["decode_ms"] = int((_pc() - t0) * 1000); _t1 = _pc()

    # Use GPU manager if available, else direct embedder
    infer = getattr(app.state, "gpu", None) or app.state.embedder
    try:
        faces = await _detect_all_async(infer, bgr)
    except Exception as e:
        if "no face" in str(e).lower():
            return PrivacyExtractResponse(results=[])
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    _tm["detect_ms"] = int((_pc() - _t1) * 1000)
    _tm["quality_ms"] = 0; _tm["blur_ms"] = 0; _tm["encode_ms"] = 0; _tm["rec_ms"] = 0

    evaluator = getattr(app.state, "quality", None)
    results = []
    
    # Compute date window for recognition filtering
    since_ts, until_ts = _date_window_from_params(
        day=req.day,
        from_day=req.from_day,
        to_day=req.to_day,
        since_ts=req.since_ts,
        until_ts=req.until_ts
    )

    # We need bboxes for all faces to blur them
    all_bboxes = []
    for f in faces:
        bbox_arr = np.asarray(getattr(f, "bbox", None), dtype=np.float32).reshape(-1)
        if bbox_arr.size == 4:
            all_bboxes.append(bbox_arr.tolist())

    for i, target_face in enumerate(faces):
        # 1. Quality filtering BEFORE any processing (on original image)
        quality_meta = None
        if evaluator:
            _q0 = _pc()
            try:
                quality_meta = evaluator.evaluate(bgr, target_face)
            except Exception:
                pass
            _tm["quality_ms"] += int((_pc() - _q0) * 1000)
        
        # 2. Define crop region with 150px padding
        target_bbox = np.asarray(getattr(target_face, "bbox", None), dtype=np.float32).reshape(-1)
        x1_t, y1_t, x2_t, y2_t = [int(v) for v in target_bbox]
        h, w = bgr.shape[:2]
        
        # Add 150px padding
        crop_x1 = max(0, x1_t - 150)
        crop_y1 = max(0, y1_t - 150)
        crop_x2 = min(w, x2_t + 150)
        crop_y2 = min(h, y2_t + 150)
        
        # Extract the crop
        crop_bgr = bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        # 3. Privacy blurring within the crop
        for j, other_bbox in enumerate(all_bboxes):
            if i == j:
                continue # Don't blur the target face
            
            ox1, oy1, ox2, oy2 = [int(v) for v in other_bbox]
            
            # Map original coordinates to crop coordinates
            lx1 = max(0, ox1 - crop_x1)
            ly1 = max(0, oy1 - crop_y1)
            lx2 = min(crop_bgr.shape[1], ox2 - crop_x1)
            ly2 = min(crop_bgr.shape[0], oy2 - crop_y1)
            
            # Check if other face is even partially within the crop
            if lx2 > lx1 and ly2 > ly1:
                face_region = crop_bgr[ly1:ly2, lx1:lx2]
                fw = lx2 - lx1
                fh = ly2 - ly1
                k_size = int(max(fw, fh) / 3) | 1
                if k_size < 3: k_size = 3
                
                blurred_face = cv2.GaussianBlur(face_region, (k_size, k_size), 30)
                crop_bgr[ly1:ly2, lx1:lx2] = blurred_face

        # 4. Optional Recognition
        rec_res = None
        if req.recognition:
            s_ts, e_ts = _date_window_from_params(
                day=req.day,
                from_day=req.from_day,
                to_day=req.to_day,
                since_ts=req.since_ts,
                until_ts=req.until_ts
            )
            emb = getattr(target_face, "normed_embedding", None)
            if emb is None:
                emb = getattr(target_face, "embedding", None)
            
            if emb is not None:
                emb = _l2_normalize(np.asarray(emb, dtype=np.float32))
                q = getattr(app.state, "qdrant", None)
                if q:
                    search_results = _qdrant_search(
                        q, 
                        app.state.qdrant_collection, 
                        emb, 
                        top_k=req.top_k or 1,
                        branch=req.branch,
                        since_ts=s_ts,
                        until_ts=e_ts,
                        access_key=access_key
                    )
                    items = [FaceSearchTopKItem(**r) for r in search_results]
                    
                    matched = False
                    subject_id = None
                    similarity = None
                    
                    if items:
                        best = items[0]
                        min_sim = float(app.state.min_similarity)
                        if float(best.similarity) >= min_sim and str(best.subject_id).strip():
                            ok, second, margin, req_m = _passes_top2_margin(search_results, float(best.similarity))
                            if ok:
                                matched = True
                                subject_id = best.subject_id
                                similarity = float(best.similarity)
                    
                    rec_res = FaceRecognizeResponse(
                        matched=matched,
                        subject_id=subject_id,
                        similarity=similarity,
                        results=items
                    )

        # 5. Encode to base64
        _e0 = _pc()
        _, buffer = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_b64 = base64.b64encode(buffer).decode('utf-8')
        _tm["encode_ms"] += int((_pc() - _e0) * 1000)

        results.append(PrivacyCropItem(
            bbox=target_bbox.tolist() if target_bbox.size == 4 else None,
            quality=quality_meta,
            image_b64=f"data:image/jpeg;base64,{crop_b64}",
            recognition=rec_res
        ))

    _tm["total_ms"] = int((_pc() - t0) * 1000)
    # Expose the per-layer split so a client/benchmark can attribute latency
    # without correlating logs: queue wait vs model processing.
    response.headers["X-Queue-Ms"] = str(_queue_wait_ms)
    response.headers["X-Model-Ms"] = str(_tm["total_ms"])
    response.headers["X-Detect-Ms"] = str(_tm.get("detect_ms", 0))
    logger.info("privacy_extract timing: queue=%dms %s faces=%d", _queue_wait_ms, _tm, len(faces))
    return PrivacyExtractResponse(results=results)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@app.post("/v1/faces/privacy_blur", response_model=PrivacyBlurResponse, dependencies=[Depends(get_api_key)])
async def privacy_blur(req: PrivacyBlurRequest) -> PrivacyBlurResponse:
    """Privacy v2: blur every detected face on the full frame EXCEPT the one at
    `bbox`. `blur_all=true` blurs all faces (including the bbox target).
    Returns the full image as base64. Input accepts base64 or http(s) URL."""
    bgr = await _decode_b64_image_async(req.image_b64)
    h, w = bgr.shape[:2]

    infer = getattr(app.state, "gpu", None) or app.state.embedder
    try:
        faces = await _detect_all_async(infer, bgr)
    except Exception as e:
        if "no face" in str(e).lower():
            faces = []
        else:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # Collect detected face boxes.
    boxes: list[list[float]] = []
    for f in faces:
        arr = np.asarray(getattr(f, "bbox", None), dtype=np.float32).reshape(-1)
        if arr.size == 4:
            boxes.append(arr.tolist())

    # Pick the keep-index = detected face best overlapping the supplied bbox.
    keep_idx = -1
    kept_bbox = None
    if not req.blur_all and req.bbox is not None:
        target = [float(v) for v in req.bbox][:4]
        if len(target) == 4:
            best_iou = 0.0
            for i, b in enumerate(boxes):
                iou = _bbox_iou(target, b)
                if iou > best_iou:
                    best_iou, keep_idx = iou, i
            # No detected face overlaps the bbox → keep that raw region anyway.
            if keep_idx >= 0:
                kept_bbox = boxes[keep_idx]
            else:
                kept_bbox = target

    blurred = 0
    for i, b in enumerate(boxes):
        if not req.blur_all and i == keep_idx:
            continue
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(w, int(b[2])), min(h, int(b[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        region = bgr[y1:y2, x1:x2]
        fw, fh = x2 - x1, y2 - y1
        k = int(max(fw, fh) / 3) | 1
        if k < 3:
            k = 3
        bgr[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 30)
        blurred += 1

    # When keeping by raw bbox (no detected overlap), un-blur nothing extra —
    # the target region was never blurred since it isn't in `boxes`.
    ok, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode output image")
    out_b64 = base64.b64encode(buffer).decode('utf-8')

    return PrivacyBlurResponse(
        image_b64=f"data:image/jpeg;base64,{out_b64}",
        faces_total=len(boxes),
        blurred_count=blurred,
        kept_bbox=kept_bbox,
    )


def _compare_two_faces(bgr1: np.ndarray, bgr2: np.ndarray) -> FaceCompareResponse:
    t0 = _pc()
    emb1, meta1 = _quality_check_and_embed(bgr1)
    emb2, meta2 = _quality_check_and_embed(bgr2)
    similarity = float(np.dot(emb1, emb2))
    is_match = similarity > 0.45
    if similarity > 0.45:
        confidence = "High"
    elif similarity > 0.35:
        confidence = "Medium"
    else:
        confidence = "Low"
    return FaceCompareResponse(
        similarity=similarity,
        match=is_match,
        confidence=confidence,
        meta={
            "timing_ms": int((_pc() - t0) * 1000),
            "image1_meta": meta1,
            "image2_meta": meta2,
        },
    )


@app.post("/v1/face/compare", response_model=FaceCompareResponse, dependencies=[Depends(get_api_key)])
def face_compare_json(req: FaceCompareRequest) -> FaceCompareResponse:
    src1 = req.image1_b64 or req.image1_url
    src2 = req.image2_b64 or req.image2_url
    if not src1 or not src2:
        raise HTTPException(status_code=400, detail="two images required: provide image1_b64/image1_url and image2_b64/image2_url")
    bgr1 = _decode_image_bytes(_decode_b64_bytes(src1))
    bgr2 = _decode_image_bytes(_decode_b64_bytes(src2))
    return _compare_two_faces(bgr1, bgr2)


@app.post("/v1/face/compare_upload", response_model=FaceCompareResponse, dependencies=[Depends(get_api_key)])
async def face_compare_upload(
    file1: UploadFile | None = File(default=None),
    file2: UploadFile | None = File(default=None),
    image1_url: str | None = Form(default=None),
    image2_url: str | None = Form(default=None),
) -> FaceCompareResponse:
    # Resolve image 1: file takes priority, then URL
    if file1 and file1.size:
        bgr1 = await _decode_image_bytes_offloaded(await file1.read())
    elif image1_url:
        bgr1 = await _decode_image_bytes_offloaded(await asyncio.to_thread(_decode_b64_bytes, image1_url))
    else:
        raise HTTPException(status_code=400, detail="image 1 required: provide file1 or image1_url")

    # Resolve image 2: file takes priority, then URL
    if file2 and file2.size:
        bgr2 = await _decode_image_bytes_offloaded(await file2.read())
    elif image2_url:
        bgr2 = await _decode_image_bytes_offloaded(await asyncio.to_thread(_decode_b64_bytes, image2_url))
    else:
        raise HTTPException(status_code=400, detail="image 2 required: provide file2 or image2_url")

    return await asyncio.to_thread(_compare_two_faces, bgr1, bgr2)


@app.post("/v1/face/search_upload", response_model=FaceSearchResponse, dependencies=[Depends(get_api_key)])
async def face_search_upload(file: UploadFile = File(...)) -> FaceSearchResponse:
    image_bytes = await file.read()
    req = FaceSearchRequest(image_b64=base64.b64encode(image_bytes).decode("ascii"))
    # face_search is a sync handler (decode + GPU embed); run it off the event loop.
    return await asyncio.to_thread(face_search, req)


@app.get("/ui")
def ui() -> Response:
    return Response(content=ui_html(), media_type="text/html")


@app.get("/v1/faces/subjects", response_model=FaceSubjectsResponse)
def faces_subjects(access_key: str = Depends(get_optional_access_key)) -> FaceSubjectsResponse:
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    subjects = _qdrant_list_subjects(q, app.state.qdrant_collection, access_key=access_key)
    return FaceSubjectsResponse(subjects=subjects)


@app.delete("/v1/faces/subjects/{subject_id}", response_model=FaceDeleteSubjectResponse)
def faces_delete_subject(
    subject_id: str,
    access_key: str = Depends(get_api_key),
) -> FaceDeleteSubjectResponse:
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
                    ),
                    FieldCondition(
                        key="access_key",
                        match=MatchValue(value=access_key),
                    )
                ]
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant delete failed: {str(e)}")

    return FaceDeleteSubjectResponse(subject_id=subject_id, deleted=True)


@app.get("/v1/stats")
def stats(access_key: str = Depends(get_optional_access_key)) -> dict[str, Any]:
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
            subjects_total = len(_qdrant_list_subjects(q, collection, access_key=access_key))
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

    last_24h_searches = len(app.state.search_events or [])
    store: EventsStore | None = getattr(app.state, "events", None)
    if store:
        try:
            # Use the store to get accurate segregated counts for the last 24h
            s_stats = store.search_events_stats(match_threshold=0.0, since_ts=cutoff, access_key=access_key)
            last_24h_searches = s_stats.get("total", 0)
        except Exception:
            pass

    return {
        "subjects_total": subjects_total,
        "embeddings_total": embeddings_total,
        "last_24h_enrolls": len(app.state.enroll_events or []),
        "last_24h_searches": last_24h_searches,
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
    branch: str | None = None,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    access_key: str = Depends(get_optional_access_key),
) -> SubjectsListResponse:
    client = getattr(app.state, "qdrant", None)
    if client is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    # Increase max limit to 20000
    limit = max(1, min(int(limit or 50), 20000))

    qstr = str(q or "").strip().lower()
    want_filter = bool(qstr)
    branch_filter = str(branch or "").strip()

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range, DatetimeRange
    except Exception:
        Filter = None

    # Pre-construct filter outside the loop
    must_filters = [FieldCondition(key="access_key", match=MatchValue(value=access_key))]
    if branch_filter and Filter:
        must_filters.append(FieldCondition(key="branch", match=MatchValue(value=branch_filter)))
    
    if (since_ts or until_ts) and Filter:
        r_kwargs = {}
        if since_ts:
            r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
        if until_ts:
            r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
        must_filters.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))

    search_filter = None
    if must_filters and Filter:
        search_filter = Filter(must=must_filters)

    try:
        # Reduced scan_limit for better stability across environments
        scan_limit = 2000
        next_cur: Any = cursor
        uniq: dict[str, None] = {}
        
        # Safety counter to prevent infinite scanning
        max_iterations = 500 
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            scroll_kwargs: dict[str, Any] = {
                "collection_name": app.state.qdrant_collection,
                "limit": int(scan_limit),
                "with_payload": True,
                "with_vectors": False,
            }
            if next_cur:
                scroll_kwargs["offset"] = next_cur
            
            if search_filter:
                scroll_kwargs["scroll_filter"] = search_filter

            try:
                batch, new_next = client.scroll(**scroll_kwargs)
            except TypeError:
                if "scroll_filter" in scroll_kwargs:
                    scroll_kwargs["filter"] = scroll_kwargs.pop("scroll_filter")
                batch, new_next = client.scroll(**scroll_kwargs)
            
            next_cur = new_next

            for pnt in batch or []:
                try:
                    payload = getattr(pnt, "payload", None) or {}
                    
                    if branch_filter:
                        p_branch = str(payload.get("branch") or "").strip()
                        if p_branch != branch_filter:
                            continue

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
    except Exception as e:
        logger.error(f"list_subjects scroll failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"qdrant scroll failed: {str(e)}")

    cap = _subject_embedding_cap()
    items: list[SubjectItem] = []
    if with_counts:
        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range, DatetimeRange
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")
        for sid in uniq.keys():
            try:
                must_conditions = [FieldCondition(key="subject_id", match=MatchValue(value=sid))]
                if branch_filter:
                    must_conditions.append(FieldCondition(key="branch", match=MatchValue(value=branch_filter)))
                if since_ts or until_ts:
                    r_kwargs = {}
                    if since_ts:
                        # ISO8601 string comparison matches enrollment 'created_at' format
                        r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
                    if until_ts:
                        r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
                    must_conditions.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))
                cnt = client.count(
                    collection_name=app.state.qdrant_collection,
                    exact=True,
                    count_filter=Filter(must=must_conditions),
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
def list_subject_images(
    subject_id: str, 
    cursor: str | None = None, 
    limit: int = 50,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    access_key: str = Depends(get_optional_access_key),
) -> SubjectImagesResponse:
    subject_id = str(subject_id or '').strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")
    limit = max(1, min(int(limit or 50), 500))

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    try:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range, DatetimeRange
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

    must_filters = [
        FieldCondition(key="subject_id", match=MatchValue(value=subject_id)),
        FieldCondition(key="access_key", match=MatchValue(value=access_key)),
    ]
    if since_ts or until_ts:
        r_kwargs = {}
        if since_ts:
            # ISO8601 string comparison in Qdrant correctly filters 'created_at'
            r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
        if until_ts:
            r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
        must_filters.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))
    
    scroll_filter = Filter(must=must_filters)

    try:
        points, next_cur = q.scroll(
            collection_name=app.state.qdrant_collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            offset=cursor,
            scroll_filter=scroll_filter,
        )
    except TypeError:
        # older qdrant_client versions use 'filter' parameter name
        points, next_cur = q.scroll(
            collection_name=app.state.qdrant_collection,
            limit=int(limit),
            with_payload=True,
            with_vectors=False,
            offset=cursor,
            filter=scroll_filter,
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
def get_subject(
    subject_id: str,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    access_key: str = Depends(get_optional_access_key),
) -> SubjectItem:
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")
    q = getattr(app.state, "qdrant", None)
    if q is None:
        raise HTTPException(status_code=501, detail="qdrant not configured")

    since_ts, until_ts = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range, DatetimeRange
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"qdrant client error: {str(e)}")

    must_filters = [
        FieldCondition(key="subject_id", match=MatchValue(value=subject_id)),
        FieldCondition(key="access_key", match=MatchValue(value=access_key)),
    ]
    if since_ts or until_ts:
        r_kwargs = {}
        if since_ts:
            # ISO8601 string comparison in Qdrant correctly filters 'created_at'
            r_kwargs["gte"] = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
        if until_ts:
            r_kwargs["lte"] = datetime.fromtimestamp(until_ts, tz=timezone.utc).isoformat()
        must_filters.append(FieldCondition(key="created_at", range=DatetimeRange(**r_kwargs)))

    try:
        res = q.count(
            collection_name=app.state.qdrant_collection,
            count_filter=Filter(must=must_filters),
            exact=True
        )
        n = int(getattr(res, "count", 0) or 0)
    except Exception:
        # Fallback to without range if needed or just return 0
        n = _qdrant_count_subject_embeddings(q, app.state.qdrant_collection, subject_id, access_key=access_key)

    cap = _subject_embedding_cap()
    return SubjectItem(subject_id=subject_id, embeddings_count=n, embeddings_cap=cap, embeddings_capped=bool(n >= cap))
@app.api_route("/health", methods=["GET", "POST"])
def health() -> dict[str, Any]:
    q = getattr(app.state, "qdrant", None)
    gpu = getattr(app.state, "gpu", None)
    
    subjects_count = 0
    groups_count = 0
    if q is not None:
        try:
            subjects_count = len(_qdrant_list_subjects(q, getattr(app.state, "qdrant_collection", None), access_key="standard"))
        except Exception:
            subjects_count = 0
        
        try:
            group_collection = os.environ.get("QDRANT_GROUPS_COLLECTION", "face_groups")
            if q.collection_exists(group_collection):
                res = q.count(collection_name=group_collection)
                groups_count = getattr(res, "count", 0)
        except Exception:
            groups_count = 0

    gpu_status = {}
    if gpu is not None:
        try:
            gpu_status = {
                "queue_size": gpu._queue.qsize(),
                "max_queue": gpu._queue.maxsize,
                "workers": len(gpu._workers),
                "batch_window_s": gpu._batch_window_s,
            }
        except Exception:
            gpu_status = {"error": "failed to get gpu status"}

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Proactive GC if memory is high (e.g., > 1GB)
    if mem_info.rss > 1024 * 1024 * 1024:
        gc.collect()
        mem_info = process.memory_info()

    return {
        "ok": True,
        "subjects": subjects_count,
        "groups": groups_count,
        "qdrant_enabled": q is not None,
        "qdrant_collection": getattr(app.state, "qdrant_collection", None),
        "gpu_inference": gpu_status,
        "system": {
            "memory_rss_mb": mem_info.rss / (1024 * 1024),
            "memory_vms_mb": mem_info.vms / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "gc_objects": len(gc.get_objects()),
        }
    }


# ---------------------------------------------------------------------------
# Async job queue — "202 + poll" facade over the heavy image endpoints.
#
# Burst-friendly ingestion: the client submits a job and returns immediately
# (202 + job_id); background workers drain the queue at server capacity by
# self-calling the existing endpoint (reusing ALL its logic including the
# concurrency gate). Results are written to a shared file store so a poll can
# land on EITHER uvicorn worker process (the two workers do not share memory).
#
#   POST /v1/jobs            {endpoint, payload}      -> 202 {job_id}
#   GET  /v1/jobs/{job_id}                            -> {status, result}
#
# Bounded everywhere: per-process queue caps at JOB_QUEUE_MAX (503 past it),
# JOB_WORKERS concurrent executors, results expire after JOB_TTL_SEC.
# ---------------------------------------------------------------------------

_JOBS_DIR = os.environ.get("JOBS_DIR", "/data/jobs")
_JOB_TTL_SEC = int(os.environ.get("JOB_TTL_SEC", "3600") or 3600)
_JOB_WORKERS = max(1, int(os.environ.get("JOB_WORKERS", "4") or 4))
_JOB_QUEUE_MAX = max(1, int(os.environ.get("JOB_QUEUE_MAX", "256") or 256))
_JOB_SELF_BASE = os.environ.get("JOB_SELF_BASE", "http://127.0.0.1:8000")
# Only heavy processing endpoints may be run as jobs.
_JOB_ALLOWED_ENDPOINTS = {
    "/v1/faces/privacy_extract",
    "/v1/faces/privacy_blur",
    "/v1/faces/recognize",
    "/v1/faces/search",
    "/v1/face/search",
}

_job_queue: "asyncio.Queue | None" = None


class JobSubmitRequest(BaseModel):
    endpoint: str
    payload: dict


def _job_path(job_id: str) -> str:
    # job_id is a server-generated uuid4 -> safe as a filename.
    return os.path.join(_JOBS_DIR, f"{job_id}.json")


def _job_write(job_id: str, data: dict) -> None:
    """Atomic write (tmp + rename) so a concurrent poll never reads a torn file."""
    os.makedirs(_JOBS_DIR, exist_ok=True)
    tmp = _job_path(job_id) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, _job_path(job_id))


def _job_read(job_id: str) -> dict | None:
    try:
        with open(_job_path(job_id)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _jobs_cleanup_expired() -> None:
    try:
        now = time.time()
        for name in os.listdir(_JOBS_DIR):
            p = os.path.join(_JOBS_DIR, name)
            try:
                if now - os.path.getmtime(p) > _JOB_TTL_SEC:
                    os.unlink(p)
            except OSError:
                pass
    except FileNotFoundError:
        pass


@app.post("/v1/jobs", status_code=202)
async def submit_job(req: JobSubmitRequest, request: Request):
    if req.endpoint not in _JOB_ALLOWED_ENDPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f"endpoint not allowed for jobs; allowed: {sorted(_JOB_ALLOWED_ENDPOINTS)}",
        )
    if _job_queue is None:
        raise HTTPException(status_code=503, detail="job queue not ready")
    if _job_queue.full():
        raise HTTPException(
            status_code=503,
            detail="job queue full, retry later",
            headers={"Retry-After": "2"},
        )
    job_id = str(uuid.uuid4())
    _job_write(job_id, {
        "job_id": job_id,
        "status": "queued",
        "endpoint": req.endpoint,
        "created_at": time.time(),
    })
    api_key = request.headers.get("x-api-key", "")
    _job_queue.put_nowait((job_id, req.endpoint, req.payload, api_key))
    return {"job_id": job_id, "status": "queued", "poll": f"/v1/jobs/{job_id}"}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    data = _job_read(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="job not found (unknown id or expired)")
    return data


async def _job_worker(worker_idx: int) -> None:
    """Drains the queue by self-calling the real endpoint on localhost.

    Self-call reuses the endpoint's full logic (validation, auth bucket,
    concurrency gate, metrics) with zero duplication; the gate bounds how many
    jobs actually hit the GPU at once.
    """
    async with httpx.AsyncClient(base_url=_JOB_SELF_BASE, timeout=300) as client:
        while True:
            job_id, endpoint, payload, api_key = await _job_queue.get()
            started = time.time()
            base = {"job_id": job_id, "endpoint": endpoint, "started_at": started}
            _job_write(job_id, {**base, "status": "running"})
            try:
                r = await client.post(endpoint, json=payload,
                                      headers={"x-api-key": api_key} if api_key else {})
                try:
                    result = r.json()
                except Exception:
                    result = {"raw": r.text[:10000]}
                _job_write(job_id, {
                    **base,
                    "status": "done" if r.status_code == 200 else "failed",
                    "http_status": r.status_code,
                    "finished_at": time.time(),
                    "duration_ms": int((time.time() - started) * 1000),
                    "result": result,
                })
            except Exception as e:
                _job_write(job_id, {
                    **base,
                    "status": "failed",
                    "error": str(e)[:2000],
                    "finished_at": time.time(),
                })
            finally:
                _job_queue.task_done()


async def _job_ttl_task() -> None:
    while True:
        await asyncio.sleep(600)
        _jobs_cleanup_expired()


@app.on_event("startup")
async def _jobs_startup() -> None:
    global _job_queue
    os.makedirs(_JOBS_DIR, exist_ok=True)
    _jobs_cleanup_expired()
    _job_queue = asyncio.Queue(maxsize=_JOB_QUEUE_MAX)
    for i in range(_JOB_WORKERS):
        asyncio.create_task(_job_worker(i))
    asyncio.create_task(_job_ttl_task())
    logger.info("async jobs ready: workers=%d queue_max=%d dir=%s ttl=%ds",
                _JOB_WORKERS, _JOB_QUEUE_MAX, _JOBS_DIR, _JOB_TTL_SEC)
