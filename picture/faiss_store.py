from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .config import INDEX_KIND, INDEX_KIND_CAPTION, MODALITY
from .io_utils import read_json, write_json


class PictureFaissIndex:
    def __init__(self, dim: int = 512, *, index_kind: str = INDEX_KIND):
        self.dim = dim
        self.index_kind = index_kind
        self.index = faiss.IndexFlatIP(dim)
        self.item_ids: list[str] = []

    def add(self, image_ids: list[str], embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if len(image_ids) != embeddings.shape[0]:
            raise ValueError("image_ids length must match embedding rows")
        self.index.add(embeddings)
        self.item_ids.extend(image_ids)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        scores, indices = self.index.search(query_embeddings, top_k)
        results = []
        for row_scores, row_indices in zip(scores, indices):
            items = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0 or idx >= len(self.item_ids):
                    continue
                items.append((self.item_ids[idx], float(score)))
            results.append(items)
        return results

    def save(self, index_path: Path, meta_path: Path, *, model_name: str = "") -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        write_json(
            meta_path,
            {
                "dim": self.dim,
                "item_ids": self.item_ids,
                "index_kind": self.index_kind,
                "modality": MODALITY,
                "model_name": model_name,
            },
        )

    @classmethod
    def load(
        cls,
        index_path: Path,
        meta_path: Path,
        *,
        expected_index_kind: str | None = INDEX_KIND,
    ) -> "PictureFaissIndex":
        meta = read_json(meta_path)
        meta_kind = meta.get("index_kind")
        if expected_index_kind and meta_kind not in {None, expected_index_kind}:
            raise ValueError(f"index_kind mismatch: {meta_kind!r} vs {expected_index_kind!r}")
        instance = cls(dim=int(meta["dim"]), index_kind=str(meta_kind or expected_index_kind or INDEX_KIND))
        instance.index = faiss.read_index(str(index_path))
        instance.item_ids = list(meta.get("item_ids") or [])
        return instance
