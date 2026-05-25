from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TAG_SPLIT_RE = re.compile(r"[,，;；、|\n]+")


@dataclass
class CaptionRecord:
    video_id: str
    video_path: str | None
    duration_seconds: float | None
    caption: str
    description: str
    tags: list[str]
    structured_caption: dict[str, Any] | None
    status: str
    model: str | None
    base_url: str | None
    created_at: str | None


def is_structured_record(record: CaptionRecord) -> bool:
    return bool(record.description.strip() and record.tags)


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = _parse_json_object(value)
        if isinstance(parsed, dict):
            value = parsed.get("tags")
        else:
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed_list = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed_list = None
                if isinstance(parsed_list, list):
                    value = parsed_list
    if isinstance(value, str):
        raw_tags = _TAG_SPLIT_RE.split(value)
    elif isinstance(value, list):
        raw_tags = [str(item) for item in value]
    else:
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def _resolve_structured_caption(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = (
        row.get("structured_caption"),
        row.get("caption_payload"),
        row.get("captionPayload"),
    )
    for candidate in candidates:
        payload = _parse_json_object(candidate)
        if payload:
            return payload
    return None


def _resolve_description(row: dict[str, Any], structured_caption: dict[str, Any] | None) -> str:
    values = (
        row.get("description"),
        None if structured_caption is None else structured_caption.get("description"),
        row.get("caption_description"),
        row.get("captionDescription"),
        row.get("caption"),
    )
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_caption(row: dict[str, Any], description: str) -> str:
    text = str(row.get("caption") or "").strip()
    return text or description


def normalize_caption_record(row: dict[str, Any]) -> CaptionRecord | None:
    video_id = str(row.get("video_id") or row.get("videoID") or "").strip()
    if not video_id:
        return None

    structured_caption = _resolve_structured_caption(row)
    tags = normalize_tags(
        row.get("tags")
        or row.get("caption_tags")
        or row.get("captionTags")
        or (None if structured_caption is None else structured_caption.get("tags"))
    )
    description = _resolve_description(row, structured_caption)
    caption = _resolve_caption(row, description)
    status = str(row.get("status") or ("ok" if caption else "missing")).strip()
    duration_value = (
        row.get("duration_seconds")
        if row.get("duration_seconds") is not None
        else row.get("duration")
    )
    try:
        duration_seconds = float(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    return CaptionRecord(
        video_id=video_id,
        video_path=str(row.get("video_path") or row.get("path") or "").strip() or None,
        duration_seconds=duration_seconds,
        caption=caption,
        description=description or caption,
        tags=tags,
        structured_caption=structured_caption,
        status=status,
        model=str(row.get("model") or row.get("caption_model") or "").strip() or None,
        base_url=str(row.get("base_url") or "").strip() or None,
        created_at=str(row.get("created_at") or row.get("caption_updated_at") or "").strip()
        or None,
    )


def read_caption_jsonl(
    path: Path,
    *,
    include_failed: bool = False,
    require_structured: bool = False,
) -> list[CaptionRecord]:
    rows: list[CaptionRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        record = normalize_caption_record(payload)
        if record is None:
            continue
        if not include_failed and record.status != "ok":
            continue
        if not record.description:
            continue
        if require_structured and not is_structured_record(record):
            continue
        rows.append(record)
    return rows
