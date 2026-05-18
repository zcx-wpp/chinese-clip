from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT


def resolve_path(value: str | Path | None, default: Path) -> Path:
    return Path(value) if value else default


def normalize_profile_name(profile: str | None) -> str | None:
    if profile is None:
        return None
    text = profile.strip()
    return text or None


def profile_root(profile: str | None) -> Path | None:
    name = normalize_profile_name(profile)
    if name is None:
        return None
    return PROJECT_ROOT / "profiles" / name


def default_output_dir(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return PROJECT_ROOT / "output"
    return root / "output"


def default_metadata_db_path(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return PROJECT_ROOT / "metadata" / "metadata.db"
    return root / "metadata.db"


def default_query_bucket_dir(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return PROJECT_ROOT / "metadata" / "query_buckets"
    return root / "query_buckets"


def default_logs_dir(profile: str | None) -> Path:
    return default_output_dir(profile) / "logs"
