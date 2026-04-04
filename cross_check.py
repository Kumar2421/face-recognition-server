from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


cross_check_router = APIRouter()


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


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


def _today_range_ts() -> tuple[float | None, float | None]:
    try:
        tz = _tz()
        d = datetime.now(tz).date()
        start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz).timestamp()
        end = (datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz) + timedelta(days=1)).timestamp()
        return float(start), float(end)
    except Exception:
        return None, None


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


class CrossCheckHit(BaseModel):
    employee_subject_id: str
    visitor_event_id: str
    visitor_subject_id: str
    similarity: float
    top2_second: float | None = None
    top2_margin: float | None = None
    visitor_ts: float | None = None
    visitor_camera: str | None = None
    visitor_image_path: str | None = None
    visitor_thumb_path: str | None = None


class CrossCheckResponse(BaseModel):
    items: list[CrossCheckHit]


def _is_employee_subject_id(sid: str) -> bool:
    s = str(sid or "").strip().lower()
    return bool(s) and s.startswith("employee-")


def _is_visitor_subject_id(sid: str) -> bool:
    s = str(sid or "").strip().lower()
    return bool(s) and (s.startswith("visiter-") or s.startswith("visitor-"))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a) + 1e-9)
    return (a / n).astype(np.float32)


def _now_ts() -> float:
    return float(datetime.now(timezone.utc).timestamp())


def _get_employee_mean_embeddings(request: Request) -> tuple[list[str], np.ndarray]:
    app = request.app
    try:
        cached = getattr(app.state, "_employee_means_cache", None)
    except Exception:
        cached = None

    now = _now_ts()
    if isinstance(cached, dict):
        try:
            if float(cached.get("ts") or 0.0) >= float(now - 60.0):
                sids = list(cached.get("sids") or [])
                vecs = cached.get("vecs")
                if isinstance(vecs, np.ndarray) and vecs.size > 0 and len(sids) == int(vecs.shape[0]):
                    return sids, vecs
        except Exception:
            pass

    q = getattr(app.state, "qdrant", None)
    if q is None:
        return [], np.zeros((0, 0), dtype=np.float32)

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    next_cur: Any = None
    scanned = 0
    max_scan = int(os.environ.get("CROSSCHECK_QDRANT_MAX_SCAN", "20000") or "20000")
    batch_limit = int(os.environ.get("CROSSCHECK_QDRANT_BATCH", "256") or "256")

    while scanned < max_scan:
        kwargs: dict[str, Any] = {
            "collection_name": app.state.qdrant_collection,
            "limit": int(batch_limit),
            "with_payload": True,
            "with_vectors": True,
        }
        if next_cur is not None:
            kwargs["offset"] = next_cur

        batch, next_cur2 = q.scroll(**kwargs)
        next_cur = next_cur2

        if not batch:
            break

        for p in batch:
            scanned += 1
            try:
                payload = getattr(p, "payload", None) or {}
                sid = str(payload.get("subject_id") or "").strip()
                if not _is_employee_subject_id(sid):
                    continue
                v = getattr(p, "vector", None)
                if v is None:
                    continue
                vec = np.asarray(v, dtype=np.float32).reshape(-1)
                if vec.size == 0:
                    continue
            except Exception:
                continue

            if sid not in sums:
                sums[sid] = vec.astype(np.float32)
                counts[sid] = 1
            else:
                sums[sid] = sums[sid] + vec
                counts[sid] = int(counts.get(sid, 0) or 0) + 1

        if next_cur is None:
            break

    sids = sorted(sums.keys())
    if not sids:
        vecs = np.zeros((0, 0), dtype=np.float32)
    else:
        m = []
        for sid in sids:
            c = max(1, int(counts.get(sid, 1) or 1))
            mean = (sums[sid] / float(c)).astype(np.float32)
            m.append(_l2_normalize(mean))
        vecs = np.stack(m, axis=0).astype(np.float32)

    try:
        app.state._employee_means_cache = {"ts": float(now), "sids": sids, "vecs": vecs}
    except Exception:
        pass
    return sids, vecs


@cross_check_router.get("/v1/cross_check/visitors_vs_employees", response_model=CrossCheckResponse)
def cross_check_visitors_vs_employees(
    request: Request,
    camera: str | None = None,
    day: str | None = None,
    from_day: str | None = None,
    to_day: str | None = None,
    since_ts: float | None = None,
    until_ts: float | None = None,
    limit: int = 500,
) -> CrossCheckResponse:
    store = getattr(request.app.state, "events", None)
    if store is None:
        raise HTTPException(status_code=500, detail="events store not configured")

    since_ts2, until_ts2 = _date_window_from_params(
        day=day,
        from_day=from_day,
        to_day=to_day,
        since_ts=since_ts,
        until_ts=until_ts,
    )

    # If caller didn't provide any window, default to today.
    if since_ts2 is None and until_ts2 is None:
        since_ts2, until_ts2 = _today_range_ts()

    emp_sids, emp_means = _get_employee_mean_embeddings(request)
    if emp_means.size == 0 or not emp_sids:
        return CrossCheckResponse(items=[])

    limit = max(1, min(int(limit or 500), 5000))
    max_events = int(os.environ.get("CROSSCHECK_MAX_EVENTS", "300") or "300")
    if max_events > 0:
        limit = min(limit, max_events)
    events, _ = store.list_events(
        camera=camera,
        decision="match",
        since_ts=since_ts2,
        until_ts=until_ts2,
        limit=int(limit),
    )

    min_sim = _as_float(os.environ.get("FACE_SERVICE_MIN_SIMILARITY", "0.25"), 0.25)
    try:
        min_top2_margin = float(os.environ.get("FACE_SERVICE_TOP2_MARGIN", "0") or "0")
    except Exception:
        min_top2_margin = 0.0
    try:
        top2_high_conf = float(os.environ.get("FACE_SERVICE_TOP2_HIGH_CONF", "0") or "0")
    except Exception:
        top2_high_conf = 0.0

    out: list[CrossCheckHit] = []
    events_dir = str(os.environ.get("EVENTS_DIR", "/data/events") or "/data/events")

    decode_image_bytes = getattr(request.app.state, "decode_image_bytes", None)
    quality_check_and_embed = getattr(request.app.state, "quality_check_and_embed", None)
    if decode_image_bytes is None or quality_check_and_embed is None:
        raise HTTPException(status_code=500, detail="cross-check dependencies not initialized")

    for it in events or []:
        visitor_sid = str(it.get("subject_id") or "").strip()
        if not _is_visitor_subject_id(visitor_sid):
            continue

        img_path = str(it.get("image_path") or "")
        if not img_path.startswith("/events/"):
            continue
        abs_path = os.path.join(events_dir, img_path.replace("/events/", "", 1).lstrip("/"))
        try:
            with open(abs_path, "rb") as f:
                img_bytes = f.read()
        except Exception:
            continue

        try:
            bgr = decode_image_bytes(img_bytes)
            emb, _meta = quality_check_and_embed(bgr)
        except Exception:
            continue

        sims = (emp_means @ np.asarray(emb, dtype=np.float32).reshape(-1, 1)).reshape(-1)
        if sims.size == 0:
            continue

        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        if best_sim < float(min_sim):
            continue

        try:
            if sims.size >= 2:
                top2 = np.sort(sims)[-2:][::-1]
                second_sim = float(top2[1])
            else:
                second_sim = None
        except Exception:
            second_sim = None

        ok = True
        margin = None
        req = None
        if second_sim is not None:
            try:
                if float(best_sim) >= float(top2_high_conf):
                    req = 0.0
                else:
                    req = float(min_top2_margin)
                margin = float(best_sim) - float(second_sim)
                ok = float(margin) >= float(req or 0.0)
            except Exception:
                ok = True

        if not ok:
            continue

        out.append(
            CrossCheckHit(
                employee_subject_id=str(emp_sids[best_i]),
                visitor_event_id=str(it.get("event_id") or ""),
                visitor_subject_id=visitor_sid,
                similarity=float(best_sim),
                top2_second=second_sim,
                top2_margin=margin,
                visitor_ts=float(it.get("ts") or 0.0) if it.get("ts") is not None else None,
                visitor_camera=str(it.get("camera") or "") or None,
                visitor_image_path=img_path or None,
                visitor_thumb_path=str(it.get("thumb_path") or "") or None,
            )
        )

    return CrossCheckResponse(items=out)
