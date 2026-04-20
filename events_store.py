from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class RecognitionEvent:
    event_id: str
    ts: float
    camera: str
    source_path: str
    decision: str
    subject_id: str | None
    similarity: float | None
    processing_ms: int | None
    model_ms: int | None
    rejected_reason: str | None
    bbox: list[float] | None
    det_score: float | None
    image_path: str
    thumb_path: str
    image_saved_at: float | None
    meta: dict[str, Any] | None
    feedback_label: str | None = None
    feedback_note: str | None = None
    feedback_updated_at: float | None = None


@dataclass
class SearchEvent:
    event_id: str
    ts: float
    query_image_path: str
    query_thumb_path: str
    top_subject_id: str | None
    top_similarity: float | None
    results: list[dict[str, Any]] | None
    meta: dict[str, Any] | None


class EventsStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_events (
                    event_id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    camera TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    subject_id TEXT,
                    similarity REAL,
                    processing_ms INTEGER,
                    model_ms INTEGER,
                    rejected_reason TEXT,
                    bbox_json TEXT,
                    det_score REAL,
                    image_path TEXT NOT NULL,
                    thumb_path TEXT NOT NULL,
                    image_saved_at REAL,
                    meta_json TEXT,
                    feedback_label TEXT,
                    feedback_note TEXT,
                    feedback_updated_at REAL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_events (
                    event_id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    query_image_path TEXT NOT NULL,
                    query_thumb_path TEXT NOT NULL,
                    top_subject_id TEXT,
                    top_similarity REAL,
                    results_json TEXT,
                    meta_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counters (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )

            # Lightweight migration for older DBs
            try:
                cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(recognition_events)").fetchall()]
            except Exception:
                cols = []
            if "image_saved_at" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN image_saved_at REAL")
                except Exception:
                    pass
            if "processing_ms" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN processing_ms INTEGER")
                except Exception:
                    pass
            if "model_ms" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN model_ms INTEGER")
                except Exception:
                    pass
            if "feedback_label" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN feedback_label TEXT")
                except Exception:
                    pass
            if "feedback_note" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN feedback_note TEXT")
                except Exception:
                    pass
            if "feedback_updated_at" not in cols:
                try:
                    conn.execute("ALTER TABLE recognition_events ADD COLUMN feedback_updated_at REAL")
                except Exception:
                    pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_ts ON recognition_events (ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_subject ON recognition_events (subject_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_camera ON recognition_events (camera)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_feedback_label ON recognition_events (feedback_label)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_events_ts ON search_events (ts DESC)"
            )

    def next_counter(self, key: str, start: int = 1) -> int:
        key = str(key or "").strip()
        if not key:
            raise ValueError("counter key is required")
        start = int(start or 1)
        if start < 0:
            start = 0

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO counters (key, value) VALUES (?, ?)",
                    (key, int(start - 1)),
                )
                conn.execute("UPDATE counters SET value = value + 1 WHERE key = ?", (key,))
                row = conn.execute("SELECT value FROM counters WHERE key = ?", (key,)).fetchone()
                if row is None:
                    return int(start)
                return int(row[0])

    def insert_event(self, ev: RecognitionEvent) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO recognition_events (
                        event_id, ts, camera, source_path, decision,
                        subject_id, similarity, processing_ms, model_ms, rejected_reason,
                        bbox_json, det_score,
                        image_path, thumb_path, image_saved_at, meta_json
                        , feedback_label, feedback_note, feedback_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.event_id,
                        float(ev.ts),
                        str(ev.camera),
                        str(ev.source_path),
                        str(ev.decision),
                        ev.subject_id,
                        float(ev.similarity) if ev.similarity is not None else None,
                        int(ev.processing_ms) if ev.processing_ms is not None else None,
                        int(ev.model_ms) if ev.model_ms is not None else None,
                        ev.rejected_reason,
                        json.dumps(ev.bbox) if ev.bbox is not None else None,
                        float(ev.det_score) if ev.det_score is not None else None,
                        str(ev.image_path),
                        str(ev.thumb_path),
                        float(ev.image_saved_at) if ev.image_saved_at is not None else None,
                        json.dumps(ev.meta) if ev.meta is not None else None,
                        ev.feedback_label,
                        ev.feedback_note,
                        float(ev.feedback_updated_at) if ev.feedback_updated_at is not None else None,
                    ),
                )

    def insert_search_event(self, ev: SearchEvent) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO search_events (
                        event_id, ts,
                        query_image_path, query_thumb_path,
                        top_subject_id, top_similarity,
                        results_json, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(ev.event_id),
                        float(ev.ts),
                        str(ev.query_image_path),
                        str(ev.query_thumb_path),
                        ev.top_subject_id,
                        float(ev.top_similarity) if ev.top_similarity is not None else None,
                        json.dumps(ev.results) if ev.results is not None else None,
                        json.dumps(ev.meta) if ev.meta is not None else None,
                    ),
                )

    def list_events(
        self,
        *,
        camera: str | None = None,
        subject_id: str | None = None,
        decision: str | None = None,
        min_similarity: float | None = None,
        max_similarity: float | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        limit: int = 100,
        cursor_ts: float | None = None,
    ) -> tuple[list[dict[str, Any]], float | None]:
        limit = max(1, min(int(limit or 100), 5000))

        where: list[str] = []
        args: list[Any] = []
        if camera:
            where.append("camera = ?")
            args.append(str(camera))
        if subject_id:
            where.append("subject_id = ?")
            args.append(str(subject_id))
        if decision:
            where.append("decision = ?")
            args.append(str(decision))
        if min_similarity is not None:
            where.append("similarity >= ?")
            args.append(float(min_similarity))
        if max_similarity is not None:
            where.append("similarity <= ?")
            args.append(float(max_similarity))
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts <= ?")
            args.append(float(until_ts))
        if cursor_ts is not None:
            # Cursor is based on the same sort key as ORDER BY below.
            where.append("COALESCE(image_saved_at, ts) < ?")
            args.append(float(cursor_ts))

        # Sort primarily by ingestion time (image_saved_at) so newly ingested events
        # show up first even if the producer sends an old `ts`.
        sql = "SELECT *, COALESCE(image_saved_at, ts) AS _sort_ts FROM recognition_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY _sort_ts DESC, ts DESC LIMIT ?"
        args.append(limit)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, args).fetchall()

        items: list[dict[str, Any]] = []
        next_cursor: float | None = None
        for r in rows or []:
            try:
                bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else None
            except Exception:
                bbox = None
            try:
                meta = json.loads(r["meta_json"]) if r["meta_json"] else None
            except Exception:
                meta = None
            it = {
                "event_id": r["event_id"],
                "ts": float(r["ts"]),
                "camera": r["camera"],
                "source_path": r["source_path"],
                "decision": r["decision"],
                "subject_id": r["subject_id"],
                "similarity": r["similarity"],
                "processing_ms": r["processing_ms"],
                "model_ms": r["model_ms"],
                "rejected_reason": r["rejected_reason"],
                "bbox": bbox,
                "det_score": r["det_score"],
                "image_path": r["image_path"],
                "thumb_path": r["thumb_path"],
                "image_saved_at": r["image_saved_at"],
                "meta": meta,
                "feedback_label": r["feedback_label"] if "feedback_label" in r.keys() else None,
                "feedback_note": r["feedback_note"] if "feedback_note" in r.keys() else None,
                "feedback_updated_at": r["feedback_updated_at"] if "feedback_updated_at" in r.keys() else None,
            }
            items.append(it)
            try:
                next_cursor = float(r["_sort_ts"])
            except Exception:
                next_cursor = float(r["ts"])

        return items, next_cursor

    def search_events_stats(
        self,
        *,
        match_threshold: float,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> dict[str, int]:
        thr = float(match_threshold)
        where: list[str] = []
        args: list[Any] = [thr]
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts < ?")
            args.append(float(until_ts))

        sql = (
            "SELECT "
            "SUM(CASE WHEN top_subject_id IS NOT NULL AND top_subject_id != '' AND top_similarity IS NOT NULL AND top_similarity >= ? THEN 1 ELSE 0 END) AS match_count, "
            "SUM(CASE WHEN top_subject_id IS NOT NULL AND top_subject_id != '' AND top_similarity IS NOT NULL AND top_similarity >= ? THEN 0 ELSE 1 END) AS no_match_count "
            "FROM search_events"
        )

        args2 = [thr, thr] + args[1:]
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._lock:
            with self._connect() as conn:
                row = conn.execute(sql, args2).fetchone()

        try:
            mc = int(row["match_count"] or 0) if row is not None else 0
        except Exception:
            mc = 0
        try:
            nmc = int(row["no_match_count"] or 0) if row is not None else 0
        except Exception:
            nmc = 0
        return {"match": mc, "no_match": nmc, "total": int(mc + nmc)}

    def list_search_events(
        self,
        *,
        limit: int = 100,
        cursor_ts: float | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> tuple[list[dict[str, Any]], float | None]:
        limit = max(1, min(int(limit or 100), 5000))

        where: list[str] = []
        args: list[Any] = []
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts < ?")
            args.append(float(until_ts))
        if cursor_ts is not None:
            where.append("ts < ?")
            args.append(float(cursor_ts))

        sql = "SELECT * FROM search_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, args).fetchall()

        items: list[dict[str, Any]] = []
        next_cursor: float | None = None
        for r in rows or []:
            try:
                results = json.loads(r["results_json"]) if r["results_json"] else []
            except Exception:
                results = []
            try:
                meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
            except Exception:
                meta = {}
            it = {
                "event_id": r["event_id"],
                "ts": float(r["ts"]),
                "query_image_path": r["query_image_path"],
                "query_thumb_path": r["query_thumb_path"],
                "top_subject_id": r["top_subject_id"],
                "top_similarity": r["top_similarity"],
                "results": results,
                "meta": meta,
            }
            items.append(it)
            next_cursor = float(r["ts"])

        return items, next_cursor

    def list_cameras(self, *, limit: int = 5000) -> list[str]:
        limit = max(1, min(int(limit or 5000), 50000))
        sql = "SELECT camera FROM recognition_events WHERE camera != '' GROUP BY camera ORDER BY camera ASC LIMIT ?"
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, (int(limit),)).fetchall()
        out: list[str] = []
        for r in rows or []:
            try:
                c = str(r["camera"] or "").strip()
            except Exception:
                c = ""
            if c:
                out.append(c)
        return out

    def recognition_stats(
        self,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
        camera: str | None = None,
    ) -> dict[str, Any]:
        where: list[str] = []
        args: list[Any] = []
        if camera:
            where.append("camera = ?")
            args.append(str(camera))
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts <= ?")
            args.append(float(until_ts))

        sql = "SELECT decision, camera, COUNT(*) AS n FROM recognition_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY decision, camera"

        total = 0
        match = 0
        no_match = 0
        rejection = 0
        by_camera: dict[str, dict[str, int]] = {}

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, args).fetchall()

        for r in rows:
            d = str(r["decision"] or "").strip().lower()
            cam = str(r["camera"] or "").strip() or "unknown"
            n = int(r["n"] or 0)
            
            total += n
            if d == "match":
                match += n
            elif d == "no_match":
                no_match += n
            elif d in ("rejection", "rejected"):
                rejection += n
                
            by_camera.setdefault(cam, {"match": 0, "no_match": 0, "rejection": 0, "total": 0})
            if d == "match":
                by_camera[cam]["match"] += n
            elif d == "no_match":
                by_camera[cam]["no_match"] += n
            elif d in ("rejection", "rejected"):
                by_camera[cam]["rejection"] += n
            by_camera[cam]["total"] += n

        return {
            "total": total,
            "match": match,
            "no_match": no_match,
            "rejection": rejection,
            "by_camera": by_camera
        }

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        event_id = str(event_id or "").strip()
        if not event_id:
            return None
        with self._lock:
            with self._connect() as conn:
                r = conn.execute(
                    "SELECT * FROM recognition_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
        if r is None:
            return None
        try:
            bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else None
        except Exception:
            bbox = None
        try:
            meta = json.loads(r["meta_json"]) if r["meta_json"] else None
        except Exception:
            meta = None
        return {
            "event_id": r["event_id"],
            "ts": float(r["ts"]),
            "camera": r["camera"],
            "source_path": r["source_path"],
            "decision": r["decision"],
            "subject_id": r["subject_id"],
            "similarity": r["similarity"],
            "processing_ms": r["processing_ms"],
            "model_ms": r["model_ms"],
            "rejected_reason": r["rejected_reason"],
            "bbox": bbox,
            "det_score": r["det_score"],
            "image_path": r["image_path"],
            "thumb_path": r["thumb_path"],
            "image_saved_at": r["image_saved_at"],
            "meta": meta,
            "feedback_label": r["feedback_label"] if "feedback_label" in r.keys() else None,
            "feedback_note": r["feedback_note"] if "feedback_note" in r.keys() else None,
            "feedback_updated_at": r["feedback_updated_at"] if "feedback_updated_at" in r.keys() else None,
        }

    def set_feedback(
        self,
        event_id: str,
        *,
        label: str | None,
        note: str | None,
        updated_at: float,
    ) -> bool:
        event_id = str(event_id or "").strip()
        if not event_id:
            return False
        label_v = str(label or "").strip() or None
        note_v = str(note or "").strip() or None
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE recognition_events SET feedback_label = ?, feedback_note = ?, feedback_updated_at = ? WHERE event_id = ?",
                    (label_v, note_v, float(updated_at), event_id),
                )
                return int(getattr(cur, "rowcount", 0) or 0) > 0

    def feedback_stats(
        self,
        *,
        since_ts: float | None = None,
        until_ts: float | None = None,
        camera: str | None = None,
    ) -> dict[str, Any]:
        where: list[str] = []
        args: list[Any] = []
        if camera:
            where.append("camera = ?")
            args.append(str(camera))
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts <= ?")
            args.append(float(until_ts))

        sql = "SELECT decision, feedback_label, COUNT(*) AS n FROM recognition_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY decision, feedback_label"

        counts: dict[str, int] = {}
        by_decision: dict[str, dict[str, int]] = {}
        total = 0

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, args).fetchall()

        for r in rows or []:
            d = str(r["decision"] or "").strip() or "unknown"
            lab = str(r["feedback_label"] or "").strip() or ""
            n = int(r["n"] or 0)
            total += n
            counts[lab] = counts.get(lab, 0) + n
            by_decision.setdefault(d, {})
            by_decision[d][lab] = by_decision[d].get(lab, 0) + n

        def _g(lbl: str) -> int:
            return int(counts.get(lbl, 0) or 0)

        tp = _g("tp")
        fp = _g("fp")
        fn = _g("fn")
        ignore = _g("ignore")
        labeled = tp + fp + fn + ignore
        unlabeled = total - labeled

        # FP rate is most meaningful for match decisions: fp / (tp + fp)
        match_map = by_decision.get("match", {})
        tp_m = int(match_map.get("tp", 0) or 0)
        fp_m = int(match_map.get("fp", 0) or 0)
        denom = tp_m + fp_m
        fp_rate_match = (float(fp_m) / float(denom)) if denom > 0 else None

        return {
            "total": int(total),
            "labeled": int(labeled),
            "unlabeled": int(unlabeled),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "ignore": int(ignore),
            "fp_rate_match": fp_rate_match,
            "by_decision": by_decision,
        }
