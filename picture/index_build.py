from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .faiss_store import PictureFaissIndex
from .vector_utils import l2_normalize

DEFAULT_INCREMENTAL_ADDS_BEFORE_FULL_REBUILD = 50


def read_build_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_build_manifest(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_faiss_index(
    *,
    faiss_dir: Path,
    index_filename: str,
    meta_filename: str,
    index_kind: str,
    dim: int,
    model_name: str,
    all_ids: list[str],
    all_vectors: np.ndarray,
    new_ids: list[str],
    force_full_rebuild: bool = False,
    incremental_adds_before_full: int = DEFAULT_INCREMENTAL_ADDS_BEFORE_FULL_REBUILD,
    manifest_path: Path | None = None,
    extra_manifest: dict | None = None,
) -> str:
    all_vectors = l2_normalize(all_vectors.astype(np.float32))
    index_path = faiss_dir / index_filename
    meta_path = faiss_dir / meta_filename
    manifest_path = manifest_path or (faiss_dir.parent / "index_manifest.json")
    prior = read_build_manifest(manifest_path)
    incremental_adds = int(prior.get("incremental_adds") or 0)
    index_exists = index_path.exists() and meta_path.exists()
    new_id_set = set(new_ids)

    if not new_id_set and index_exists and not force_full_rebuild:
        write_build_manifest(
            manifest_path,
            {
                "index_kind": index_kind,
                "image_count": len(all_ids),
                "model_name": model_name,
                "last_build_mode": "noop",
                "incremental_adds": incremental_adds,
                **(extra_manifest or {}),
            },
        )
        return "noop"

    need_full = (
        force_full_rebuild or not index_exists or incremental_adds >= incremental_adds_before_full
    )
    existing: PictureFaissIndex | None = None
    if index_exists:
        existing = PictureFaissIndex.load(index_path, meta_path, expected_index_kind=index_kind)
        if not need_full and new_id_set & set(existing.item_ids):
            need_full = True

    if not need_full and existing is not None:
        id_to_row = dict(zip(all_ids, all_vectors, strict=False))
        append_ids = [i for i in new_ids if i not in set(existing.item_ids)]
        if append_ids:
            matrix = np.vstack([id_to_row[i] for i in append_ids]).astype(np.float32)
            existing.add(append_ids, matrix)
            existing.save(index_path, meta_path, model_name=model_name)
            incremental_adds += 1
            mode = "incremental"
        else:
            mode = "incremental"
    else:
        faiss_index = PictureFaissIndex(dim=dim, index_kind=index_kind)
        faiss_index.add(all_ids, all_vectors)
        faiss_index.save(index_path, meta_path, model_name=model_name)
        incremental_adds = 0
        mode = "full"

    write_build_manifest(
        manifest_path,
        {
            "index_kind": index_kind,
            "image_count": len(all_ids),
            "model_name": model_name,
            "last_built_at": datetime.now(timezone.utc).isoformat(),
            "last_build_mode": mode,
            "incremental_adds": incremental_adds,
            **(extra_manifest or {}),
        },
    )
    return mode
