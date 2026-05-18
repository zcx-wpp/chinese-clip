from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .faiss_store import FaissFrameIndex
from .frame_selector import extract_candidate_frames
from .logging_utils import append_error_log
from .metadata_store import MetadataStore
from .video_representation import (
    SegmentRepresentation,
    compute_global_genericness,
    compute_pairwise_segment_similarity,
    select_top_representative_segments,
)


def probe_video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(total_frames / fps) if fps > 0 else 0.0
    finally:
        capture.release()


def extract_frames_for_segment(
    segment_path: Path,
    output_dir: Path,
    frames_per_second: float,
    min_side: int,
    image_format: str,
    jpg_quality: int,
):
    return extract_candidate_frames(
        segment_path=segment_path,
        output_dir=output_dir,
        frames_per_second=frames_per_second,
        min_side=min_side,
        image_format=image_format,
        jpg_quality=jpg_quality,
    )


def existing_segment_paths(segments_dir: Path, video_id: str) -> list[Path]:
    return sorted((segments_dir / video_id).glob(f"{video_id}_seg_*.mp4"))


def append_stage_error(logs_dir: Path, stage: str, **payload):
    append_error_log(logs_dir, {"stage": stage, **payload})


def load_existing_embedding(path_value: str | None) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    return np.load(path).astype(np.float32) if path.exists() else None


def persist_faiss_index(dim: int, item_ids: list[str], embedding_rows: list[np.ndarray], index_path: Path, meta_path: Path):
    index = FaissFrameIndex(dim=dim)
    if item_ids:
        index.add(frame_ids=item_ids, embeddings=np.vstack(embedding_rows).astype(np.float32))
    index.persist(index_path, meta_path)


def refresh_representative_segments(
    store: MetadataStore,
    video_ids: list[str],
    top_n: int,
    genericness_weight: float,
    diversity_penalty_weight: float,
) -> int:
    segment_records = store.get_segment_records_by_video_ids(video_ids)
    segment_embeddings: dict[str, np.ndarray] = {}
    segment_representations_by_video: dict[str, list[SegmentRepresentation]] = {}

    for item in segment_records:
        embedding = load_existing_embedding(item.get("embedding_path"))
        if embedding is None:
            continue
        segment_embeddings[item["segment_id"]] = embedding
        segment_representations_by_video.setdefault(item["video_id"], []).append(
            SegmentRepresentation(
                segment_id=item["segment_id"],
                video_id=item["video_id"],
                importance_score=float(item.get("importance_score") or 0.0),
                motion_score=float(item.get("motion_score") or 0.0),
                visual_diversity_score=float(item.get("visual_diversity_score") or 0.0),
                embedding_norm_score=float(item.get("embedding_norm") or 0.0),
            )
        )

    genericness_by_segment_id = compute_global_genericness(segment_embeddings)
    pairwise_similarity = compute_pairwise_segment_similarity(segment_embeddings)
    for segment_id, genericness_score in genericness_by_segment_id.items():
        store.update_segment_genericness(segment_id=segment_id, genericness_score=genericness_score)

    refreshed_videos = 0
    for video_id in video_ids:
        representations = segment_representations_by_video.get(video_id, [])
        selected = select_top_representative_segments(
            representations,
            top_n=top_n,
            pairwise_similarity=pairwise_similarity,
            genericness_by_segment_id=genericness_by_segment_id,
            genericness_weight=genericness_weight,
            diversity_penalty_weight=diversity_penalty_weight,
        ) if representations else []
        store.update_segment_representative_flags(video_id=video_id, selected_segment_ids=[item.segment_id for item in selected])
        refreshed_videos += int(bool(selected))
    return refreshed_videos
