from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS video (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                duration REAL,
                path TEXT
            );

            CREATE TABLE IF NOT EXISTS segment (
                segment_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                path TEXT,
                FOREIGN KEY(video_id) REFERENCES video(video_id)
            );

            CREATE TABLE IF NOT EXISTS frame_embedding (
                frame_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                timestamp REAL,
                frame_index INTEGER,
                modality TEXT,
                frame_path TEXT,
                extra_json TEXT,
                FOREIGN KEY(video_id) REFERENCES video(video_id),
                FOREIGN KEY(segment_id) REFERENCES segment(segment_id)
            );
            """
        )
        self.conn.commit()

    def upsert_video(self, video_id: str, title: str, duration: float, path: str):
        self.conn.execute(
            """
            INSERT INTO video(video_id, title, duration, path)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title=excluded.title,
                duration=excluded.duration,
                path=excluded.path
            """,
            (video_id, title, duration, path),
        )
        self.conn.commit()

    def upsert_segment(self, segment_id: str, video_id: str, start_time: float, end_time: float, path: str):
        self.conn.execute(
            """
            INSERT INTO segment(segment_id, video_id, start_time, end_time, path)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(segment_id) DO UPDATE SET
                video_id=excluded.video_id,
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                path=excluded.path
            """,
            (segment_id, video_id, start_time, end_time, path),
        )
        self.conn.commit()

    def insert_frame_embedding(
        self,
        frame_id: str,
        video_id: str,
        segment_id: str,
        timestamp: float,
        frame_index: int,
        modality: str,
        frame_path: str,
        extra: dict,
    ):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO frame_embedding(
                frame_id, video_id, segment_id, timestamp, frame_index, modality, frame_path, extra_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (frame_id, video_id, segment_id, timestamp, frame_index, modality, frame_path, json.dumps(extra, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_frame_records(self, frame_ids: list[str]) -> list[dict]:
        if not frame_ids:
            return []
        placeholders = ",".join("?" for _ in frame_ids)
        rows = self.conn.execute(
            f"""
            SELECT
                fe.frame_id,
                fe.video_id,
                fe.segment_id,
                fe.timestamp,
                fe.frame_index,
                fe.modality,
                fe.frame_path,
                fe.extra_json,
                s.start_time AS segment_start,
                s.end_time AS segment_end,
                v.path AS video_path
            FROM frame_embedding fe
            JOIN segment s ON s.segment_id = fe.segment_id
            JOIN video v ON v.video_id = fe.video_id
            WHERE fe.frame_id IN ({placeholders})
            """,
            frame_ids,
        ).fetchall()
        order = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        records = []
        for row in rows:
            item = dict(row)
            item["extra"] = json.loads(item.pop("extra_json") or "{}")
            records.append(item)
        records.sort(key=lambda item: order[item["frame_id"]])
        return records
