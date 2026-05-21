from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ARTIFACT_ROOT, DEFAULT_INDEX_DIR, PIPELINE_ROOT


DEFAULT_SEARCH_PROFILE_NAMES = (
    "apr_media1",
)


@dataclass(frozen=True)
class SearchSource:
    name: str
    metadata_db_path: Path
    index_dir: Path


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
    return PIPELINE_ROOT / "profiles" / name


def default_output_dir(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return ARTIFACT_ROOT / "output"
    return root / "output"


def default_metadata_db_path(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return ARTIFACT_ROOT / "metadata.db"
    return root / "metadata.db"


def default_index_dir(profile: str | None) -> Path:
    root = profile_root(profile)
    if root is None:
        return DEFAULT_INDEX_DIR
    return root / "hybrid_index"


def parse_profile_names(profile: str | None, *, fallback: tuple[str, ...] = DEFAULT_SEARCH_PROFILE_NAMES) -> tuple[str, ...]:
    if profile is None:
        return fallback
    names = tuple(part.strip() for part in str(profile).split(",") if part.strip())
    return names or fallback


def resolve_search_sources(
    profile: str | None,
    metadata_db: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> list[SearchSource]:
    has_explicit_paths = metadata_db is not None or index_dir is not None
    if has_explicit_paths:
        names = parse_profile_names(profile, fallback=())
        if len(names) > 1:
            raise ValueError("Explicit --metadata-db/--index-dir overrides do not support multiple profiles.")
        source_name = names[0] if names else "custom"
        base_profile = names[0] if names else None
        return [
            SearchSource(
                name=source_name,
                metadata_db_path=resolve_path(metadata_db, default_metadata_db_path(base_profile)),
                index_dir=resolve_path(index_dir, default_index_dir(base_profile)),
            )
        ]

    return [
        SearchSource(
            name=name,
            metadata_db_path=default_metadata_db_path(name),
            index_dir=default_index_dir(name),
        )
        for name in parse_profile_names(profile)
    ]
