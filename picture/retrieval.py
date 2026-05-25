from __future__ import annotations

import time
from pathlib import Path

from .encoder import ChineseClipEncoder, encode_pil_images
from .faiss_store import PictureFaissIndex
from .image_io import load_image_rgb
from .image_resolve import resolve_image_file
from .metadata_store import PictureMetadataStore
from .portable_paths import resolve_portable_path
from .profile_paths import default_metadata_db_path, default_output_dir
from .search_dedup import compute_fetch_k, dedupe_hits
from .vector_utils import l2_normalize


class PictureRetriever:
    def __init__(
        self,
        *,
        encoder: ChineseClipEncoder,
        index: PictureFaissIndex,
        store: PictureMetadataStore,
        search_roots: list[Path] | None = None,
    ):
        self.encoder = encoder
        self.index = index
        self.store = store
        self.search_roots = search_roots or []

    @classmethod
    def load(
        cls,
        *,
        output_dir: Path,
        metadata_db_path: Path,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> PictureRetriever:
        encoder = ChineseClipEncoder(model_path=model_path, device=device, batch_size=batch_size)
        index = PictureFaissIndex.load(
            output_dir / "faiss" / "image_index.faiss",
            output_dir / "faiss" / "image_index.meta.json",
        )
        store = PictureMetadataStore(metadata_db_path)
        return cls(encoder=encoder, index=index, store=store, search_roots=[])

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        dedupe: bool = True,
        dedupe_method: str = "md5",
        dedupe_similarity: float = 0.99,
    ) -> tuple[list[dict], float]:
        vectors = self.encoder.encode_texts([query])
        if vectors.size == 0:
            return [], 0.0
        query_vec = l2_normalize(vectors[0])
        started = time.perf_counter()
        n = len(self.index.item_ids)
        fetch_k = compute_fetch_k(top_k, n) if dedupe else min(top_k, n)
        raw_hits = self.index.search(query_vec.reshape(1, -1), fetch_k)[0] if fetch_k else []
        if dedupe:
            raw_hits = dedupe_hits(
                raw_hits,
                top_k=top_k,
                resolve_path=self.resolve_path,
                method=dedupe_method,
                similarity_threshold=dedupe_similarity,
                faiss_index=self.index,
            )
        else:
            raw_hits = raw_hits[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [self._hit_to_dict(image_id, score) for image_id, score in raw_hits], elapsed_ms

    def search_image(
        self,
        source: str | Path,
        *,
        top_k: int = 10,
        dedupe: bool = True,
        dedupe_method: str = "md5",
        dedupe_similarity: float = 0.99,
    ) -> tuple[list[dict], float]:
        image = load_image_rgb(source)
        query_vec = l2_normalize(encode_pil_images(self.encoder, [image])[0])
        started = time.perf_counter()
        n = len(self.index.item_ids)
        fetch_k = compute_fetch_k(top_k, n) if dedupe else min(top_k, n)
        raw_hits = self.index.search(query_vec.reshape(1, -1), fetch_k)[0] if fetch_k else []
        if dedupe:
            raw_hits = dedupe_hits(
                raw_hits,
                top_k=top_k,
                resolve_path=self.resolve_path,
                method=dedupe_method,
                similarity_threshold=dedupe_similarity,
                faiss_index=self.index,
            )
        else:
            raw_hits = raw_hits[:top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [self._hit_to_dict(image_id, score) for image_id, score in raw_hits], elapsed_ms

    def _hit_to_dict(self, image_id: str, score: float) -> dict:
        row = self.store.get_image(image_id) or {}
        path_text = str(row.get("path") or image_id)
        return {
            "image_id": image_id,
            "score": float(score),
            "path": path_text,
            "resolved_path": str(resolve_portable_path(path_text)),
        }

    def resolve_path(self, image_id: str, image_root: Path | None = None) -> Path | None:
        row = self.store.get_image(image_id)
        if not row:
            return None
        roots: list[Path] = []
        if image_root:
            roots.append(Path(image_root))
        roots.extend(self.search_roots)
        return resolve_image_file(image_id, str(row.get("path") or ""), roots)

    def close(self) -> None:
        self.store.close()


def build_retriever(
    *,
    profile: str | None,
    model_path: str,
    output_dir: Path | None = None,
    metadata_db: Path | None = None,
    device: str = "cuda",
    search_roots: list[Path] | None = None,
) -> PictureRetriever:
    resolved_output = output_dir or default_output_dir(profile)
    resolved_db = metadata_db or default_metadata_db_path(profile)
    retriever = PictureRetriever.load(
        output_dir=resolved_output,
        metadata_db_path=resolved_db,
        model_path=model_path,
        device=device,
    )
    if search_roots:
        retriever.search_roots = [Path(p).resolve() for p in search_roots]
    return retriever
