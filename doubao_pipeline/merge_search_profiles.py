from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .io_utils import write_json
from .metadata_store import MetadataStore
from .profile_paths import default_index_dir, default_metadata_db_path, profile_root


VIDEO_COLUMNS = (
    "video_id",
    "path",
    "duration",
    "caption_tags",
    "caption_description",
    "caption_payload",
    "caption_model",
    "caption_updated_at",
)

SEARCH_DOCUMENT_COLUMNS = (
    "video_id",
    "path",
    "duration",
    "tags_json",
    "description",
    "caption_text",
    "sparse_tags",
    "sparse_description",
    "sparse_caption",
    "search_payload",
    "caption_model",
    "caption_updated_at",
    "embedding_model",
    "embedding_dim",
    "embedding_blob",
    "embedding_updated_at",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple search profiles into one combined profile.")
    parser.add_argument(
        "--source-profile",
        action="append",
        required=True,
        dest="source_profiles",
        help="Source profile name to merge. Repeat this flag for each input profile.",
    )
    parser.add_argument("--dest-profile", required=True, help="Destination profile name for the merged output.")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_source_profiles(values: Iterable[str]) -> list[str]:
    profiles: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip()
            if name:
                profiles.append(name)
    return profiles


def _profile_paths(profile: str) -> tuple[Path, Path]:
    metadata_db = default_metadata_db_path(profile)
    index_dir = default_index_dir(profile)
    if not metadata_db.exists():
        raise FileNotFoundError(f"metadata.db not found for profile '{profile}': {metadata_db}")
    if not index_dir.exists():
        raise FileNotFoundError(f"hybrid_index not found for profile '{profile}': {index_dir}")
    return metadata_db, index_dir


def _validate_source_manifests(source_profiles: list[str]) -> tuple[dict, dict]:
    index_manifests: list[dict] = []
    embedder_manifests: list[dict] = []
    for profile in source_profiles:
        _, index_dir = _profile_paths(profile)
        index_manifests.append(_read_json(index_dir / "index_manifest.json"))
        embedder_manifests.append(_read_json(index_dir / "embedder_manifest.json"))

    reference_embedder = embedder_manifests[0]
    for profile, manifest in zip(source_profiles[1:], embedder_manifests[1:]):
        if manifest != reference_embedder:
            raise ValueError(
                f"Embedder manifest mismatch for profile '{profile}'. "
                "All merged profiles must use the same dense embedding configuration."
            )

    reference_index = index_manifests[0]
    for profile, manifest in zip(source_profiles[1:], index_manifests[1:]):
        for key in ("embedding_backend", "embedding_model", "embedding_dim"):
            if manifest.get(key) != reference_index.get(key):
                raise ValueError(
                    f"Index manifest mismatch for profile '{profile}' on '{key}': "
                    f"{manifest.get(key)!r} != {reference_index.get(key)!r}"
                )
    return reference_index, reference_embedder


def _check_for_overlapping_video_ids(source_profiles: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for profile in source_profiles:
        metadata_db, _ = _profile_paths(profile)
        conn = sqlite3.connect(str(metadata_db))
        try:
            cursor = conn.execute("SELECT video_id FROM search_documents ORDER BY video_id")
            for (video_id,) in cursor:
                if video_id in seen:
                    duplicates.append(str(video_id))
                    if len(duplicates) >= 20:
                        break
                else:
                    seen.add(str(video_id))
            if duplicates:
                break
        finally:
            conn.close()

    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Source profiles contain overlapping video_id values: {preview}")


def _ensure_destination_paths(dest_profile: str) -> tuple[Path, Path]:
    root = profile_root(dest_profile)
    if root is None:
        raise ValueError("Destination profile name is required.")
    if root.exists():
        raise FileExistsError(f"Destination profile already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return default_metadata_db_path(dest_profile), default_index_dir(dest_profile)


def _merge_table(
    dest_conn: sqlite3.Connection,
    sql: str,
    select_sql: str,
    source_metadata_db: Path,
) -> int:
    source_conn = sqlite3.connect(str(source_metadata_db))
    try:
        cursor = source_conn.execute(select_sql)
        dest_conn.executemany(sql, cursor)
        count_row = source_conn.execute(f"SELECT COUNT(*) FROM ({select_sql})").fetchone()
        return int(count_row[0]) if count_row else 0
    finally:
        source_conn.close()


def merge_profiles(source_profiles: list[str], dest_profile: str) -> dict:
    source_profiles = _normalize_source_profiles(source_profiles)
    if len(source_profiles) < 2:
        raise ValueError("Provide at least two non-empty source profile names to merge.")

    reference_index_manifest, reference_embedder_manifest = _validate_source_manifests(source_profiles)
    _check_for_overlapping_video_ids(source_profiles)

    dest_metadata_db, dest_index_dir = _ensure_destination_paths(dest_profile)
    dest_store = MetadataStore(dest_metadata_db)

    insert_video_sql = """
        INSERT INTO videos(
            video_id,
            path,
            duration,
            caption_tags,
            caption_description,
            caption_payload,
            caption_model,
            caption_updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            path=COALESCE(excluded.path, videos.path),
            duration=COALESCE(excluded.duration, videos.duration),
            caption_tags=excluded.caption_tags,
            caption_description=excluded.caption_description,
            caption_payload=excluded.caption_payload,
            caption_model=COALESCE(excluded.caption_model, videos.caption_model),
            caption_updated_at=COALESCE(excluded.caption_updated_at, videos.caption_updated_at)
    """
    insert_search_document_sql = """
        INSERT INTO search_documents(
            video_id,
            path,
            duration,
            tags_json,
            description,
            caption_text,
            sparse_tags,
            sparse_description,
            sparse_caption,
            search_payload,
            caption_model,
            caption_updated_at,
            embedding_model,
            embedding_dim,
            embedding_blob,
            embedding_updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            path=COALESCE(excluded.path, search_documents.path),
            duration=COALESCE(excluded.duration, search_documents.duration),
            tags_json=excluded.tags_json,
            description=excluded.description,
            caption_text=excluded.caption_text,
            sparse_tags=excluded.sparse_tags,
            sparse_description=excluded.sparse_description,
            sparse_caption=excluded.sparse_caption,
            search_payload=COALESCE(excluded.search_payload, search_documents.search_payload),
            caption_model=COALESCE(excluded.caption_model, search_documents.caption_model),
            caption_updated_at=COALESCE(excluded.caption_updated_at, search_documents.caption_updated_at),
            embedding_model=COALESCE(excluded.embedding_model, search_documents.embedding_model),
            embedding_dim=COALESCE(excluded.embedding_dim, search_documents.embedding_dim),
            embedding_blob=COALESCE(excluded.embedding_blob, search_documents.embedding_blob),
            embedding_updated_at=COALESCE(excluded.embedding_updated_at, search_documents.embedding_updated_at)
    """
    select_video_sql = f"SELECT {', '.join(VIDEO_COLUMNS)} FROM videos ORDER BY video_id"
    select_search_document_sql = f"SELECT {', '.join(SEARCH_DOCUMENT_COLUMNS)} FROM search_documents ORDER BY video_id"

    try:
        total_video_rows = 0
        total_search_rows = 0
        dest_conn = dest_store.conn
        dest_conn.execute("BEGIN")
        for profile in source_profiles:
            source_metadata_db, _ = _profile_paths(profile)
            print(f"[merge] source_profile={profile} metadata_db={source_metadata_db}", flush=True)
            total_video_rows += _merge_table(dest_conn, insert_video_sql, select_video_sql, source_metadata_db)
            total_search_rows += _merge_table(
                dest_conn,
                insert_search_document_sql,
                select_search_document_sql,
                source_metadata_db,
            )

        dest_conn.execute("DELETE FROM search_fts")
        dest_conn.execute(
            """
            INSERT INTO search_fts(video_id, tags_terms, description_terms, caption_terms)
            SELECT
                video_id,
                sparse_tags,
                sparse_description,
                sparse_caption
            FROM search_documents
            ORDER BY video_id
            """
        )
        dest_conn.commit()

        video_count = int(dest_conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0])
        search_document_count = int(dest_conn.execute("SELECT COUNT(*) FROM search_documents").fetchone()[0])
        fts_count = int(dest_conn.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0])
    except Exception:
        dest_store.close()
        raise

    dest_index_dir.mkdir(parents=True, exist_ok=True)
    write_json(dest_index_dir / "embedder_manifest.json", reference_embedder_manifest)
    manifest = {
        "records_indexed": search_document_count,
        "embedding_backend": reference_index_manifest.get("embedding_backend"),
        "embedding_model": reference_index_manifest.get("embedding_model"),
        "embedding_dim": reference_index_manifest.get("embedding_dim"),
        "metadata_db": str(dest_metadata_db),
        "index_dir": str(dest_index_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_profiles": list(source_profiles),
        "source_video_rows": total_video_rows,
        "source_search_rows": total_search_rows,
        "videos_merged": video_count,
        "fts_rows": fts_count,
    }
    write_json(dest_index_dir / "index_manifest.json", manifest)
    dest_store.close()
    return manifest


def main():
    args = parse_args()
    manifest = merge_profiles(args.source_profiles, args.dest_profile)
    print(
        f"[done] dest_profile={args.dest_profile} records={manifest['records_indexed']} "
        f"videos={manifest['videos_merged']} source_profiles={','.join(manifest['source_profiles'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
