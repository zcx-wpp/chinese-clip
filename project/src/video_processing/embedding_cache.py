from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import write_json
from .logging_utils import utc_now_iso
from .portable_paths import portable_path_text


def embedding_metadata_path(embedding_path: Path) -> Path:
    return embedding_path.with_suffix(".json")


def write_embedding_cache(
    embedding_path: Path,
    embedding: np.ndarray,
    item_id: str,
    model_name: str,
    model_path: str,
    model_revision: str,
    embedding_dim: int,
    embedding_dtype: str,
    embedding_norm: float,
    item_id_key: str = "frame_id",
) -> Path:
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embedding.astype(np.float32))

    metadata_path = embedding_metadata_path(embedding_path)
    metadata = {
        item_id_key: item_id,
        "model": model_name,
        "model_path": portable_path_text(model_path) or model_path,
        "model_revision": model_revision,
        "dim": int(embedding_dim),
        "embedding_dtype": embedding_dtype,
        "created_at": utc_now_iso(),
        "norm": float(embedding_norm),
        "embedding_path": portable_path_text(embedding_path) or str(embedding_path.resolve()),
    }
    write_json(metadata_path, metadata)
    return metadata_path
