from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .caption_records import (
    CaptionRecord,
    is_structured_record,
    normalize_caption_record,
    read_caption_jsonl,
)
from .config import DEFAULT_CAPTIONS_JSONL
from .dense_embeddings import (
    DEFAULT_BGE_MODEL_NAME,
    DEFAULT_BGE_QUERY_INSTRUCTION,
    HuggingFaceBgeTextEmbedder,
)
from ..io_utils import write_json
from .metadata_store import MetadataStore
from ..profile_paths import (
    default_hybrid_index_dir,
    default_hybrid_metadata_db_path,
    resolve_path,
)
from .search_text import build_caption_term_text, build_description_term_text, build_tag_term_text


def _progress(iterable, *, total: int | None = None, desc: str, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a local hybrid retrieval index from Doubao captions."
    )
    parser.add_argument("--captions-jsonl", default=str(DEFAULT_CAPTIONS_JSONL))
    parser.add_argument(
        "--profile", help="Optional profile name for metadata.db and hybrid index output."
    )
    parser.add_argument("--metadata-db", help="Optional metadata.db path.")
    parser.add_argument("--index-dir", help="Optional hybrid index output directory.")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_BGE_MODEL_NAME,
        help="Hugging Face model id or local path for dense text embeddings.",
    )
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        help="Embedding device. Defaults to cuda and falls back to cpu when CUDA is unavailable.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument(
        "--embedding-query-instruction",
        default=DEFAULT_BGE_QUERY_INSTRUCTION,
        help="Query prefix used by BGE when encoding search queries.",
    )
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        help="Only load the embedding model from local files; do not download from Hugging Face.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Deprecated legacy local-svd option. Ignored for BGE embeddings.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Deprecated legacy local-svd option. Ignored for BGE embeddings.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Only index the first N caption rows. 0 means all."
    )
    parser.add_argument(
        "--allow-legacy-caption-only",
        action="store_true",
        help="Allow legacy caption-only rows without structured tags/description. Disabled by default.",
    )
    return parser.parse_args()


def _record_from_video_row(row: dict) -> CaptionRecord | None:
    payload = {
        "video_id": row.get("video_id"),
        "path": row.get("path"),
        "duration": row.get("duration"),
        "caption_tags": row.get("caption_tags"),
        "caption_description": row.get("caption_description"),
        "caption_payload": row.get("caption_payload"),
        "caption_model": row.get("caption_model"),
        "caption_updated_at": row.get("caption_updated_at"),
        "status": "ok",
    }
    return normalize_caption_record(payload)


def load_records(
    captions_jsonl: Path, store: MetadataStore, limit: int, allow_legacy_caption_only: bool
) -> list[CaptionRecord]:
    records: list[CaptionRecord]
    if captions_jsonl.exists():
        all_records = read_caption_jsonl(captions_jsonl)
        if allow_legacy_caption_only:
            records = all_records
        else:
            structured_records = [record for record in all_records if is_structured_record(record)]
            legacy_count = len(all_records) - len(structured_records)
            if legacy_count > 0:
                raise RuntimeError(
                    f"Caption file is legacy or partially structured: {captions_jsonl} "
                    f"(total={len(all_records)}, structured={len(structured_records)}, legacy={legacy_count}). "
                    "Please regenerate captions with doubao_batch_caption.py so every row includes tags and description, "
                    "or pass --allow-legacy-caption-only to index the old format anyway."
                )
            records = structured_records
    else:
        records = [
            record
            for row in store.list_video_records()
            if (record := _record_from_video_row(row)) is not None
        ]
        if not allow_legacy_caption_only:
            records = [record for record in records if is_structured_record(record)]
    if limit > 0:
        records = records[:limit]
    if not records:
        raise RuntimeError(
            f"No usable caption records found. captions_jsonl={captions_jsonl} db={store.db_path}"
        )
    return records


def sync_records_to_store(store: MetadataStore, records: list[CaptionRecord]) -> None:
    iterator = _progress(records, total=len(records), desc="sync metadata", unit="video")
    for record in iterator:
        payload = record.structured_caption or {
            "tags": record.tags,
            "description": record.description,
        }
        store.upsert_video_caption_metadata(
            video_id=record.video_id,
            path=record.video_path,
            duration=record.duration_seconds,
            tags_json=json.dumps(record.tags, ensure_ascii=False),
            description=record.description,
            payload_json=json.dumps(payload, ensure_ascii=False),
            caption_model=record.model,
            caption_updated_at=record.created_at,
        )


def build_dense_source_text(record: CaptionRecord) -> str:
    return record.description or record.caption


def _caption_is_newer_than_embedding(
    caption_updated_at: str | None, embedding_updated_at: str | None
) -> bool:
    if not caption_updated_at or not embedding_updated_at:
        return False
    return caption_updated_at > embedding_updated_at


def record_needs_dense_embedding(store: MetadataStore, record: CaptionRecord) -> bool:
    state = store.get_search_document_embedding_state(record.video_id)
    if not state or not state.get("embedding_blob"):
        return True
    caption_at = record.created_at or state.get("caption_updated_at")
    return _caption_is_newer_than_embedding(caption_at, state.get("embedding_updated_at"))


def partition_records_for_dense_embedding(
    store: MetadataStore, records: list[CaptionRecord]
) -> tuple[list[CaptionRecord], list[CaptionRecord]]:
    needing: list[CaptionRecord] = []
    cached: list[CaptionRecord] = []
    for record in records:
        if record_needs_dense_embedding(store, record):
            needing.append(record)
        else:
            cached.append(record)
    return needing, cached


def build_index(
    records: list[CaptionRecord],
    store: MetadataStore,
    index_dir: Path,
    *,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
    embedding_max_length: int,
    embedding_query_instruction: str,
    embedding_local_files_only: bool,
) -> dict:
    needing_records, cached_records = partition_records_for_dense_embedding(store, records)
    print(
        f"[step] dense embedding plan total={len(records)} encode={len(needing_records)} "
        f"reuse={len(cached_records)}",
        flush=True,
    )

    embedder: HuggingFaceBgeTextEmbedder | None = None
    vectors_by_video_id: dict[str, np.ndarray] = {}
    if needing_records:
        dense_texts = [build_dense_source_text(record) for record in needing_records]
        print(
            f"[step] loading embedding model model={embedding_model} device={embedding_device}",
            flush=True,
        )
        embedder = HuggingFaceBgeTextEmbedder(
            model_name=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
            query_instruction=embedding_query_instruction,
            local_files_only=embedding_local_files_only,
        )
        print(
            f"[step] encoding dense vectors records={len(dense_texts)} batch_size={embedding_batch_size}",
            flush=True,
        )
        encoded_vectors = embedder.encode_passages(dense_texts, progress_desc="encode BGE")
        vectors_by_video_id = {
            record.video_id: vector
            for record, vector in zip(needing_records, encoded_vectors, strict=True)
        }
        embedder.save(index_dir)
    elif cached_records:
        print("[step] all dense embeddings up to date; skipping BGE model load", flush=True)

    print(f"[step] writing search documents records={len(records)}", flush=True)
    iterator = _progress(records, total=len(records), desc="write search docs", unit="video")
    now_iso = datetime.now(timezone.utc).isoformat()
    for record in iterator:
        payload = record.structured_caption or {
            "tags": record.tags,
            "description": record.description,
        }
        vector = vectors_by_video_id.get(record.video_id)
        store.upsert_search_document(
            video_id=record.video_id,
            path=record.video_path,
            duration=record.duration_seconds,
            tags_json=json.dumps(record.tags, ensure_ascii=False),
            description=record.description,
            caption_text=record.caption,
            sparse_tags=build_tag_term_text(record.tags),
            sparse_description=build_description_term_text(record.description),
            sparse_caption=build_caption_term_text(record.tags, record.description, record.caption),
            search_payload=json.dumps(payload, ensure_ascii=False),
            caption_model=record.model,
            caption_updated_at=record.created_at,
            embedding_model=embedder.model_name if embedder is not None and vector is not None else None,
            embedding_vector=vector,
            embedding_updated_at=now_iso if vector is not None else None,
        )

    manifest = {
        "records_indexed": len(records),
        "records_encoded": len(needing_records),
        "records_reused_embeddings": len(cached_records),
        "embedding_backend": embedder.backend if embedder is not None else None,
        "embedding_model": embedder.model_name if embedder is not None else embedding_model,
        "embedding_dim": embedder.vector_dim if embedder is not None else None,
        "metadata_db": str(store.db_path),
        "index_dir": str(index_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(index_dir / "index_manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    metadata_db = resolve_path(args.metadata_db, default_hybrid_metadata_db_path(args.profile))
    index_dir = resolve_path(args.index_dir, default_hybrid_index_dir(args.profile))
    captions_jsonl = Path(args.captions_jsonl)

    if args.embedding_dim != 256 or args.max_features != 20000:
        print(
            "[warn] --embedding-dim and --max-features are legacy local-svd options and are ignored for BGE embeddings.",
            flush=True,
        )

    store = MetadataStore(metadata_db)
    try:
        records = load_records(captions_jsonl, store, args.limit, args.allow_legacy_caption_only)
        print(f"[step] loaded caption records={len(records)} source={captions_jsonl}", flush=True)
        sync_records_to_store(store, records)
        manifest = build_index(
            records,
            store,
            index_dir,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
            embedding_query_instruction=args.embedding_query_instruction,
            embedding_local_files_only=args.embedding_local_files_only,
        )
    finally:
        store.close()

    print(
        f"[done] records={manifest['records_indexed']} metadata_db={metadata_db} "
        f"index_dir={index_dir} embedding_model={manifest['embedding_model']} "
        f"embedding_dim={manifest['embedding_dim']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
