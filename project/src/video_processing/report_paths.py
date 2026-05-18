from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .profile_paths import default_logs_dir, resolve_path


def resolve_log_path(profile: str | None, value: str | None, filename: str) -> Path:
    return resolve_path(value, default_logs_dir(profile) / filename)


def build_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamped_log_path(
    profile: str | None,
    value: str | None,
    filename: str,
    *,
    timestamp: str | None = None,
) -> Path:
    if value:
        return Path(value)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    run_timestamp = timestamp or build_run_timestamp()
    return default_logs_dir(profile) / f"{stem}_{run_timestamp}{suffix}"


def latest_log_path(
    profile: str | None,
    value: str | None,
    filename: str,
) -> Path:
    if value:
        return Path(value)
    logs_dir = default_logs_dir(profile)
    exact_path = logs_dir / filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidates: list[Path] = []
    if exact_path.exists():
        candidates.append(exact_path)
    candidates.extend(logs_dir.glob(f"{stem}_*{suffix}"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return exact_path
