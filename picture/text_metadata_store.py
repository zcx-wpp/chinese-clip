from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .portable_paths import portable_path_text, resolved_path_text


class CaptionMetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS caption_images (
                image_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                image_url TEXT,
                subject TEXT DEFAULT '',
                color TEXT DEFAULT '',
                action TEXT DEFAULT '',
                style TEXT DEFAULT '',
                description TEXT DEFAULT '',
                caption_text TEXT DEFAULT '',
                structured_json TEXT,
                caption_model TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                embedding_blob BLOB,
                status TEXT DEFAULT 'pending',
                error_message TEXT
            );
            """
        )
        self.conn.commit()

    def _run(self, sql: str, params=(), *, commit: bool = False):
        with self._lock:
            cur = self.conn.execute(sql, params)
            if commit:
                self.conn.commit()
            return cur

    def _fetchone(self, sql: str, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params=()):
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_status(self, image_id: str) -> str | None:
        r = self._fetchone("SELECT status FROM caption_images WHERE image_id=?", (image_id,))
        return r["status"] if r else None

    def get_record(self, image_id: str) -> dict | None:
        r = self._fetchone("SELECT * FROM caption_images WHERE image_id=?", (image_id,))
        return dict(r) if r else None

    def list_ids_by_status(self, status: str) -> list[str]:
        rows = self._fetchall(
            "SELECT image_id FROM caption_images WHERE status=? ORDER BY image_id",
            (status,),
        )
        return [str(r["image_id"]) for r in rows]

    def mark_captioning(self, image_id: str, path: str) -> None:
        if self.get_record(image_id):
            self._run(
                "UPDATE caption_images SET status='captioning', path=?, error_message=NULL WHERE image_id=?",
                (portable_path_text(path) or path, image_id),
                commit=True,
            )
        else:
            self.upsert_caption(
                image_id=image_id, path=path, subject="", color="", action="", style="",
                description="", caption_text="", structured_json="{}", caption_model=None, status="captioning",
            )

    def upsert_caption(self, **kw) -> None:
        self._run(
            """
            INSERT INTO caption_images (
                image_id, path, image_url, subject, color, action, style, description,
                caption_text, structured_json, caption_model, status, error_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(image_id) DO UPDATE SET
                path=excluded.path, subject=excluded.subject, color=excluded.color,
                action=excluded.action, style=excluded.style, description=excluded.description,
                caption_text=excluded.caption_text, structured_json=excluded.structured_json,
                caption_model=excluded.caption_model, status=excluded.status, error_message=excluded.error_message
            """,
            (
                kw["image_id"], portable_path_text(kw["path"]) or kw["path"], kw.get("image_url"),
                kw["subject"], kw["color"], kw["action"], kw["style"], kw["description"],
                kw["caption_text"], kw["structured_json"], kw.get("caption_model"),
                kw.get("status", "caption_done"), kw.get("error_message"),
            ),
            commit=True,
        )

    def upsert_embedding(self, image_id: str, *, embedding: np.ndarray, embedding_model: str) -> None:
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        dim = int(np.asarray(embedding).shape[-1])
        self._run(
            """
            UPDATE caption_images SET embedding_model=?, embedding_dim=?, embedding_blob=?, status='done'
            WHERE image_id=?
            """,
            (embedding_model, dim, blob, image_id),
            commit=True,
        )

    def list_records_matching_filters(
        self,
        *,
        subject: str | None = None,
        color: str | None = None,
        action: str | None = None,
        style: str | None = None,
        require_embedding: bool = True,
    ) -> list[dict]:
        clauses = ["status = 'done'"]
        params: list[str] = []
        if require_embedding:
            clauses.append("embedding_blob IS NOT NULL")
        for field, value in (
            ("subject", subject),
            ("color", color),
            ("action", action),
            ("style", style),
        ):
            if value and str(value).strip():
                clauses.append(f"{field} LIKE ?")
                params.append(f"%{str(value).strip()}%")
        sql = f"SELECT * FROM caption_images WHERE {' AND '.join(clauses)} ORDER BY image_id"
        return [dict(r) for r in self._fetchall(sql, params)]

    def list_caption_done(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM caption_images WHERE status='caption_done' ORDER BY image_id"
        )
        return [dict(r) for r in rows]

    def load_embedding(self, record: dict) -> np.ndarray | None:
        blob, dim = record.get("embedding_blob"), record.get("embedding_dim")
        if not blob or not dim:
            return None
        v = np.frombuffer(blob, dtype=np.float32)
        return v if v.size == int(dim) else None

    def list_done_with_embeddings(self):
        rows = self._fetchall(
            "SELECT * FROM caption_images WHERE status='done' AND embedding_blob IS NOT NULL"
        )
        out = []
        for row in rows:
            d = dict(row)
            vec = self.load_embedding(d)
            if vec is not None:
                out.append((d["image_id"], vec, d))
        return out

    def mark_failed(self, image_id: str, path: str, msg: str) -> None:
        self.upsert_caption(
            image_id=image_id, path=path, subject="", color="", action="", style="",
            description="", caption_text="", structured_json="{}", caption_model=None,
            status="failed", error_message=msg,
        )

    def close(self) -> None:
        self.conn.close()
