"""Rebuild segment embeddings and FAISS indexes from cached frame vectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .embedding_cache import write_embedding_cache
from .metadata_store import MetadataStore
from .pipeline_utils import persist_faiss_index
from .segment_embedding import softmax_attention_pooling
from .video_representation import (
    SegmentRepresentation,
    compute_frame_diff_motion_score,
    compute_segment_importance,
    select_top_representative_segments,
)

ProgressCallback = Callable[[str], None] | None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _log(callback: ProgressCallback, message: str) -> None:
    if callback is not None:
        callback(message)
    else:
        print(message, flush=True)


def rebuild_segment_embeddings_from_frames(
    store: MetadataStore,
    output_dir: Path,
    *,
    model_name: str,
    model_path: str,
    model_revision: str,
    representative_top_n: int,
    on_progress: ProgressCallback = None,
) -> tuple[int, int, int]:
    """Aggregate cached frame .npy files into segment embeddings and pick representatives."""
    frame_records = store.get_all_frame_records()
    if not frame_records:
        return 0, 0, 0
    _log(on_progress, f"[index] loaded frame records: {len(frame_records)}")

    frames_by_segment: dict[str, list[dict]] = defaultdict(list)
    video_ids: list[str] = []
    seen_video_ids: set[str] = set()
    for item in frame_records:
        frames_by_segment[item["segment_id"]].append(item)
        video_id = item["video_id"]
        if video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            video_ids.append(video_id)
    _log(
        on_progress,
        f"[index] grouped into segments={len(frames_by_segment)} videos={len(video_ids)}",
    )

    embedding_dim = 0
    rebuilt_segments = 0
    total_segments = len(frames_by_segment)
    segment_representations_by_video: dict[str, list[SegmentRepresentation]] = defaultdict(list)
    for segment_id, records in frames_by_segment.items():
        records.sort(key=lambda item: (float(item["timestamp"]), int(item.get("frame_index") or 0)))
        frame_embeddings = []
        frame_scores = []
        for item in records:
            frame_embedding = normalize_vector(np.load(Path(item["embedding_path"])).astype(np.float32))
            embedding_dim = embedding_dim or int(frame_embedding.shape[-1])
            frame_embeddings.append(frame_embedding)
            frame_scores.append(float(item.get("embedding_norm") or 1.0))
        if not frame_embeddings:
            continue

        segment_matrix = np.vstack(frame_embeddings).astype(np.float32)
        frame_score_array = np.asarray(frame_scores, dtype=np.float32)
        segment_start = float(records[0].get("segment_start") or 0.0)
        segment_end = float(records[0].get("segment_end") or 0.0)
        frame_timestamps = np.asarray(
            [float(item.get("timestamp") or 0.0) - segment_start for item in records],
            dtype=np.float32,
        )
        frame_paths = [str(item["frame_path"]) for item in records if item.get("frame_path")]
        frame_diff_motion_score = compute_frame_diff_motion_score(frame_paths)
        importance_score, motion_score, visual_diversity_score, embedding_norm_score = (
            compute_segment_importance(
                frame_embeddings=segment_matrix,
                frame_norms=frame_score_array,
                frame_timestamps=frame_timestamps,
                segment_duration_seconds=max(segment_end - segment_start, 0.0),
                frame_diff_motion_score=frame_diff_motion_score,
                embedding_norm_weight=1.0,
                motion_score_weight=1.0,
                visual_diversity_weight=1.0,
            )
        )
        segment_embedding, _ = softmax_attention_pooling(segment_matrix, frame_score_array)
        video_id = records[0]["video_id"]
        segment_embedding_path = output_dir / "embeddings" / video_id / f"{segment_id}.npy"
        segment_metadata_path = write_embedding_cache(
            embedding_path=segment_embedding_path,
            embedding=segment_embedding,
            item_id=segment_id,
            item_id_key="segment_id",
            model_name=model_name,
            model_path=model_path,
            model_revision=model_revision,
            embedding_dim=embedding_dim,
            embedding_dtype="float32",
            embedding_norm=float(np.linalg.norm(segment_embedding)),
        )
        store.update_segment_embedding(
            segment_id=segment_id,
            embedding_path=str(segment_embedding_path.resolve()),
            embedding_metadata_path=str(segment_metadata_path.resolve()),
            embedding_norm=float(np.linalg.norm(segment_embedding)),
            importance_score=importance_score,
            motion_score=motion_score,
            visual_diversity_score=visual_diversity_score,
        )
        segment_representations_by_video[video_id].append(
            SegmentRepresentation(
                segment_id=segment_id,
                video_id=video_id,
                importance_score=importance_score,
                motion_score=motion_score,
                visual_diversity_score=visual_diversity_score,
                embedding_norm_score=embedding_norm_score,
            )
        )
        rebuilt_segments += 1
        if (
            rebuilt_segments <= 5
            or rebuilt_segments % 1000 == 0
            or rebuilt_segments == total_segments
        ):
            _log(
                on_progress,
                f"[index] rebuilt segments: {rebuilt_segments}/{total_segments}",
            )

    rebuilt_videos = 0
    total_videos = len(video_ids)
    _log(
        on_progress,
        f"[index] selecting representative segments: videos={total_videos} top_n={representative_top_n}",
    )
    for idx, video_id in enumerate(video_ids, start=1):
        selected = (
            select_top_representative_segments(
                segment_representations_by_video.get(video_id, []),
                top_n=representative_top_n,
            )
            if segment_representations_by_video.get(video_id)
            else []
        )
        store.update_segment_representative_flags(
            video_id=video_id,
            selected_segment_ids=[item.segment_id for item in selected],
        )
        if selected:
            rebuilt_videos += 1
        if idx <= 5 or idx % 500 == 0 or idx == total_videos:
            _log(on_progress, f"[index] representative selection: {idx}/{total_videos}")

    return len(frame_records), rebuilt_segments, rebuilt_videos


def _load_embedding(path: Path, *, normalize: bool) -> np.ndarray:
    vector = np.load(path).astype(np.float32)
    return normalize_vector(vector) if normalize else vector


def rebuild_faiss_indexes(
    store: MetadataStore,
    faiss_dir: Path,
    *,
    normalize_embeddings: bool = True,
    on_progress: ProgressCallback = None,
) -> tuple[int, int, int]:
    """Build frame / segment / video FAISS indexes from rows in metadata.db."""
    faiss_dir.mkdir(parents=True, exist_ok=True)
    _log(on_progress, f"[index] rebuilding FAISS indexes under: {faiss_dir}")

    frame_records = store.get_all_frame_records()
    frame_ids: list[str] = []
    frame_embeddings: list[np.ndarray] = []
    total_frame_records = len(frame_records)
    for item in frame_records:
        path = Path(item["embedding_path"])
        if not path.exists():
            continue
        frame_ids.append(item["frame_id"])
        frame_embeddings.append(_load_embedding(path, normalize=normalize_embeddings))
        if (
            len(frame_embeddings) <= 5
            or len(frame_embeddings) % 10000 == 0
            or len(frame_embeddings) == total_frame_records
        ):
            _log(
                on_progress,
                f"[index] frame embeddings for FAISS: {len(frame_embeddings)}/{total_frame_records}",
            )

    if not frame_embeddings:
        raise RuntimeError("No cached frame embeddings found.")
    dim = int(frame_embeddings[0].shape[-1])

    persist_faiss_index(
        dim,
        frame_ids,
        frame_embeddings,
        faiss_dir / "frame_index.faiss",
        faiss_dir / "frame_index.meta.json",
    )
    _log(on_progress, f"[index] saved frame index: count={len(frame_ids)}")

    video_ids: list[str] = []
    seen_video_ids: set[str] = set()
    for item in frame_records:
        video_id = item["video_id"]
        if video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            video_ids.append(video_id)

    segment_records = store.get_segment_records_by_video_ids(video_ids)
    segment_ids: list[str] = []
    segment_embeddings: list[np.ndarray] = []
    total_segment_records = len(segment_records)
    for item in segment_records:
        embedding_path = item.get("embedding_path")
        if not embedding_path:
            continue
        path = Path(embedding_path)
        if not path.exists():
            continue
        segment_ids.append(item["segment_id"])
        segment_embeddings.append(_load_embedding(path, normalize=normalize_embeddings))
        if (
            len(segment_embeddings) <= 5
            or len(segment_embeddings) % 5000 == 0
            or len(segment_embeddings) == total_segment_records
        ):
            _log(
                on_progress,
                f"[index] segment embeddings for FAISS: {len(segment_embeddings)}/{total_segment_records}",
            )

    persist_faiss_index(
        dim,
        segment_ids,
        segment_embeddings,
        faiss_dir / "segment_index.faiss",
        faiss_dir / "segment_index.meta.json",
    )
    _log(on_progress, f"[index] saved segment index: count={len(segment_ids)}")

    representative_segment_records = store.get_representative_segment_records_by_video_ids(
        video_ids
    )
    indexed_video_ids: list[str] = []
    video_embeddings: list[np.ndarray] = []
    total_representative_records = len(representative_segment_records)
    for item in representative_segment_records:
        embedding_path = item.get("embedding_path")
        if not embedding_path:
            continue
        path = Path(embedding_path)
        if not path.exists():
            continue
        indexed_video_ids.append(f"{item['video_id']}::{item['segment_id']}")
        video_embeddings.append(_load_embedding(path, normalize=normalize_embeddings))
        if (
            len(video_embeddings) <= 5
            or len(video_embeddings) % 5000 == 0
            or len(video_embeddings) == total_representative_records
        ):
            _log(
                on_progress,
                f"[index] video-level embeddings for FAISS: {len(video_embeddings)}/{total_representative_records}",
            )

    persist_faiss_index(
        dim,
        indexed_video_ids,
        video_embeddings,
        faiss_dir / "video_index.faiss",
        faiss_dir / "video_index.meta.json",
    )
    _log(on_progress, f"[index] saved video index: count={len(indexed_video_ids)}")

    return len(frame_ids), len(segment_ids), len(indexed_video_ids)
