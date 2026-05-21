from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .faiss_store import PictureFaissIndex
from .vector_utils import l2_normalize


def compute_fetch_k(top_k: int, index_size: int, *, multiplier: int = 5, extra: int = 40) -> int:
    """FAISS 预取条数：保证去重后仍能凑满 top_k 条不同画面。"""
    if index_size <= 0 or top_k <= 0:
        return 0
    return min(index_size, max(top_k, top_k * multiplier, top_k + extra))


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def dedupe_hits(
    hits: list[tuple[str, float]],
    *,
    top_k: int,
    resolve_path: Callable[[str], Path | None],
    method: str = "md5",
    similarity_threshold: float = 0.99,
    faiss_index: PictureFaissIndex | None = None,
) -> list[tuple[str, float]]:
    """
    在 FAISS 粗排结果上按内容去重，保留分数最高的一条。

    method:
      - md5: 字节完全相同（MUGE 重复图）
      - embedding: 向量余弦 >= threshold（略慢，可合并近重复）
    """
    if top_k <= 0 or not hits:
        return []

    method = (method or "md5").strip().lower()
    seen_md5: set[str] = set()
    kept_vectors: list[np.ndarray] = []
    id_to_idx: dict[str, int] | None = None
    index = None

    if method == "embedding":
        if faiss_index is None:
            raise ValueError("embedding dedupe requires faiss_index")
        index = faiss_index.index
        id_to_idx = {iid: i for i, iid in enumerate(faiss_index.item_ids)}

    out: list[tuple[str, float]] = []
    for image_id, score in hits:
        if method == "md5":
            path = resolve_path(image_id)
            if path is not None:
                key = _file_md5(path)
                if key in seen_md5:
                    continue
                seen_md5.add(key)
        elif method == "embedding":
            assert id_to_idx is not None and index is not None
            idx = id_to_idx.get(image_id)
            if idx is None:
                continue
            vec = l2_normalize(np.asarray(index.reconstruct(int(idx)), dtype=np.float32))
            if kept_vectors and max(float(np.dot(vec, k)) for k in kept_vectors) >= similarity_threshold:
                continue
            kept_vectors.append(vec)
        else:
            raise ValueError(f"unknown dedupe method: {method}")

        out.append((image_id, score))
        if len(out) >= top_k:
            break
    return out
