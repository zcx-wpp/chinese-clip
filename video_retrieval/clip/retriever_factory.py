from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import DEFAULT_MODEL_PATH
from ..profile_paths import ProfileLayout
from .config import RetrievalConfig
from .embedding import ChineseClipEncoder
from .faiss_store import FaissFrameIndex
from .metadata_store import MetadataStore
from .retrieval import VideoRetriever


def build_retriever(
    *,
    output_dir: str | Path,
    metadata_db_path: str | Path,
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    video_recall_top_k: int | None = None,
    segment_recall_top_k: int | None = None,
    video_recall_candidate_pool_size: int | None = None,
    segment_recall_candidate_pool_size: int | None = None,
    rerank_top_k_average: int | None = None,
    rerank_smoothmax_beta: float | None = None,
    clip_score_weight: float | None = None,
    motion_score_weight: float | None = None,
    rerank_segment_support_weight: float | None = None,
    rerank_genericness_penalty_weight: float | None = None,
) -> VideoRetriever:
    output_dir = Path(output_dir)
    metadata_db_path = Path(metadata_db_path)
    resolved_model_path = model_path or str(DEFAULT_MODEL_PATH)

    faiss_dir = output_dir / "faiss"
    frame_index = FaissFrameIndex.load(
        faiss_dir / "frame_index.faiss", faiss_dir / "frame_index.meta.json"
    )
    segment_index = FaissFrameIndex.load(
        faiss_dir / "segment_index.faiss", faiss_dir / "segment_index.meta.json"
    )
    video_index = None
    video_index_path = faiss_dir / "video_index.faiss"
    video_meta_path = faiss_dir / "video_index.meta.json"
    if video_index_path.exists() and video_meta_path.exists():
        video_index = FaissFrameIndex.load(video_index_path, video_meta_path)

    overrides = {
        "video_recall_top_k": video_recall_top_k,
        "segment_recall_top_k": segment_recall_top_k,
        "video_recall_candidate_pool_size": video_recall_candidate_pool_size,
        "segment_recall_candidate_pool_size": segment_recall_candidate_pool_size,
        "rerank_top_k_average": rerank_top_k_average,
        "rerank_smoothmax_beta": rerank_smoothmax_beta,
        "clip_score_weight": clip_score_weight,
        "motion_score_weight": motion_score_weight,
        "rerank_segment_support_weight": rerank_segment_support_weight,
        "rerank_genericness_penalty_weight": rerank_genericness_penalty_weight,
    }
    retrieval_config = replace(
        RetrievalConfig(), **{k: v for k, v in overrides.items() if v is not None}
    )

    encoder = ChineseClipEncoder(
        model_path=resolved_model_path, device=device, batch_size=batch_size
    )
    store = MetadataStore(metadata_db_path)
    return VideoRetriever(
        encoder=encoder,
        index=frame_index,
        segment_index=segment_index,
        video_index=video_index,
        metadata_store=store,
        retrieval_config=retrieval_config,
    )


def build_retriever_from_layout(
    layout: ProfileLayout,
    *,
    model_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    **kwargs,
) -> VideoRetriever:
    """Build a retriever from unified profile layout (preferred entry for services)."""
    return build_retriever(
        output_dir=layout.clip_output_dir,
        metadata_db_path=layout.clip_metadata_db,
        model_path=str(model_path or DEFAULT_MODEL_PATH),
        device=device,
        batch_size=batch_size,
        **kwargs,
    )
