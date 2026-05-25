from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import INDEX_KIND_CAPTION
from .faiss_store import PictureFaissIndex
from .image_resolve import resolve_image_file
from .profile_paths import default_caption_metadata_db_path, default_output_dir, resolve_path
from .search_dedup import compute_fetch_k, dedupe_hits
from .text_metadata_store import CaptionMetadataStore
from .vector_utils import l2_normalize


@dataclass
class TextSearchFilters:
    subject: str | None = None
    color: str | None = None
    action: str | None = None
    style: str | None = None

    def active(self) -> bool:
        return any(
            str(v or "").strip() for v in (self.subject, self.color, self.action, self.style)
        )


class PictureTextRetriever:
    def __init__(
        self,
        *,
        output_dir: Path,
        metadata_db: Path,
        bge_device: str = "cuda",
        bge_batch_size: int = 16,
        search_roots: list[Path] | None = None,
    ):
        self.output_dir = output_dir
        self.search_roots = search_roots or []
        self.store = CaptionMetadataStore(metadata_db)
        self.bge_device = bge_device
        self.bge_batch_size = bge_batch_size
        self.faiss_index = PictureFaissIndex.load(
            output_dir / "faiss" / "caption_index.faiss",
            output_dir / "faiss" / "caption_index.meta.json",
            expected_index_kind=INDEX_KIND_CAPTION,
        )
        self._embedder = None
        self._cache = {item[0]: item[2] for item in self.store.list_done_with_embeddings()}

    def _get_embedder(self):
        if self._embedder is None:
            from video_retrieval.hybrid.dense_embeddings import HuggingFaceBgeTextEmbedder

            from .config import DEFAULT_BGE_MODEL_NAME

            bge_dir = self.output_dir / "bge_embedder"
            if (bge_dir / "embedder_manifest.json").exists():
                self._embedder = HuggingFaceBgeTextEmbedder.load(
                    bge_dir, device=self.bge_device, batch_size=self.bge_batch_size
                )
            else:
                self._embedder = HuggingFaceBgeTextEmbedder(
                    model_name=DEFAULT_BGE_MODEL_NAME, device=self.bge_device
                )
        return self._embedder

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: TextSearchFilters | None = None,
    ) -> tuple[list[dict], float]:
        filters = filters or TextSearchFilters()
        embedder = self._get_embedder()
        query_vec = l2_normalize(embedder.encode_queries([query])[0])
        started = time.perf_counter()

        if filters.active():
            candidates = self.store.list_records_matching_filters(
                subject=filters.subject,
                color=filters.color,
                action=filters.action,
                style=filters.style,
            )
            scored = []
            for record in candidates:
                vec = self.store.load_embedding(record)
                if vec is None:
                    continue
                scored.append((float(np.dot(l2_normalize(vec), query_vec)), record))
            scored.sort(key=lambda x: x[0], reverse=True)
            hits = scored[:top_k]
            results = [self._record_to_dict(rec, sc) for sc, rec in hits]
        else:
            fetch_k = compute_fetch_k(top_k, len(self.faiss_index.item_ids))
            if fetch_k <= 0:
                return [], 0.0
            raw = self.faiss_index.search(query_vec.reshape(1, -1), fetch_k)[0]
            raw = dedupe_hits(
                raw,
                top_k=top_k,
                resolve_path=self.resolve_path,
                method="md5",
            )
            results = []
            for image_id, score in raw:
                record = self._cache.get(image_id) or self.store.get_record(image_id)
                if not record:
                    continue
                results.append(self._record_to_dict(record, float(score)))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return results, elapsed_ms

    def _record_to_dict(self, record: dict, score: float) -> dict:
        return {
            "image_id": record["image_id"],
            "score": score,
            "path": str(record.get("path") or ""),
            "subject": str(record.get("subject") or ""),
            "color": str(record.get("color") or ""),
            "action": str(record.get("action") or ""),
            "style": str(record.get("style") or ""),
            "description": str(record.get("description") or ""),
            "display_line": (
                f"主体: [{record.get('subject') or '-'}] | 颜色: [{record.get('color') or '-'}] | "
                f"动作: [{record.get('action') or '-'}] | 风格: [{record.get('style') or '-'}]"
            ),
        }

    def resolve_path(self, image_id: str, image_root: Path | None = None) -> Path | None:
        record = self.store.get_record(image_id)
        if not record:
            return None
        roots: list[Path] = []
        if image_root:
            roots.append(Path(image_root))
        roots.extend(self.search_roots)
        return resolve_image_file(image_id, str(record.get("path") or ""), roots)

    def index_count(self) -> int:
        return len(self.faiss_index.item_ids)

    def close(self) -> None:
        self.store.close()


def build_text_retriever(
    *,
    profile: str | None,
    output_dir: Path | None = None,
    metadata_db: Path | None = None,
    bge_device: str = "cuda",
    search_roots: list[Path] | None = None,
) -> PictureTextRetriever:
    return PictureTextRetriever(
        output_dir=resolve_path(output_dir, default_output_dir(profile)),
        metadata_db=resolve_path(metadata_db, default_caption_metadata_db_path(profile)),
        bge_device=bge_device,
        search_roots=search_roots,
    )
