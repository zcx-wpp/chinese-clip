from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .io_utils import read_json, write_json
from .vector_store import VectorStore


class FaissFrameIndex(VectorStore):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.item_ids: list[str] = []

    def add(self, frame_ids: list[str], embeddings: np.ndarray):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.item_ids.extend(frame_ids)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
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

    def save(self, index_path: Path | None = None, meta_path: Path | None = None):
        if index_path is None or meta_path is None:
            return None
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        write_json(meta_path, {"dim": self.dim, "item_ids": self.item_ids})

    def persist(self, index_path: Path, meta_path: Path):
        self.save(index_path, meta_path)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissFrameIndex":
        meta = read_json(meta_path)
        instance = cls(dim=meta["dim"])
        instance.index = faiss.read_index(str(index_path))
        instance.item_ids = list(meta.get("item_ids") or meta.get("frame_ids") or [])
        return instance
