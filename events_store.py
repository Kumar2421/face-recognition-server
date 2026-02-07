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
    rejected_reason: str | None
    bbox: list[float] | None
    det_score: float | None
    image_path: str
    thumb_path: str
    image_saved_at: float | None
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
                    rejected_reason TEXT,
                    bbox_json TEXT,
                    det_score REAL,
                    image_path TEXT NOT NULL,
                    thumb_path TEXT NOT NULL,
                    image_saved_at REAL,
                    meta_json TEXT
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_ts ON recognition_events (ts DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_subject ON recognition_events (subject_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_camera ON recognition_events (camera)"
            )

    def insert_event(self, ev: RecognitionEvent) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO recognition_events (
                        event_id, ts, camera, source_path, decision,
                        subject_id, similarity, rejected_reason,
                        bbox_json, det_score,
                        image_path, thumb_path, image_saved_at, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.event_id,
                        float(ev.ts),
                        str(ev.camera),
                        str(ev.source_path),
                        str(ev.decision),
                        ev.subject_id,
                        float(ev.similarity) if ev.similarity is not None else None,
                        ev.rejected_reason,
                        json.dumps(ev.bbox) if ev.bbox is not None else None,
                        float(ev.det_score) if ev.det_score is not None else None,
                        str(ev.image_path),
                        str(ev.thumb_path),
                        float(ev.image_saved_at) if ev.image_saved_at is not None else None,
                        json.dumps(ev.meta) if ev.meta is not None else None,
                    ),
                )

    def list_events(
        self,
        *,
        camera: str | None = None,
        subject_id: str | None = None,
        decision: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        limit: int = 100,
        cursor_ts: float | None = None,
    ) -> tuple[list[dict[str, Any]], float | None]:
        limit = max(1, min(int(limit or 100), 500))

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
        if since_ts is not None:
            where.append("ts >= ?")
            args.append(float(since_ts))
        if until_ts is not None:
            where.append("ts <= ?")
            args.append(float(until_ts))
        if cursor_ts is not None:
            where.append("ts < ?")
            args.append(float(cursor_ts))

        sql = "SELECT * FROM recognition_events"
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
                "rejected_reason": r["rejected_reason"],
                "bbox": bbox,
                "det_score": r["det_score"],
                "image_path": r["image_path"],
                "thumb_path": r["thumb_path"],
                "image_saved_at": r["image_saved_at"],
                "meta": meta,
            }
            items.append(it)
            next_cursor = float(r["ts"])

        return items, next_cursor

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
            "rejected_reason": r["rejected_reason"],
            "bbox": bbox,
            "det_score": r["det_score"],
            "image_path": r["image_path"],
            "thumb_path": r["thumb_path"],
            "image_saved_at": r["image_saved_at"],
            "meta": meta,
        }
