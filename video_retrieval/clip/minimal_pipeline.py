from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import DEFAULT_CLIP_PROFILE, DEFAULT_MODEL_PATH, DEFAULT_VIDEO_DIR
from ..logging_utils import build_logger
from .embedding_cache import write_embedding_cache
from .index_builder import rebuild_faiss_indexes
from .metadata_store import MetadataStore
from .pipeline_utils import (
    append_stage_error,
    existing_segment_paths,
    extract_frames_for_segment,
    probe_video_duration,
    refresh_representative_segments,
    segment_representations_from_store,
    video_embedding_outputs_ready,
)
from ..profile_paths import (
    default_clip_metadata_db_path,
    default_clip_output_dir,
    resolve_path,
)
from .scheduler import GLOBAL_FAISS_STAGE, GLOBAL_FAISS_VIDEO_ID, PipelineScheduler
from .segmenter import iter_videos, segment_video

if TYPE_CHECKING:
    from .embedding import ChineseClipEncoder
    from .video_representation import SegmentRepresentation


@dataclass
class MinimalConfig:
    video_dir: Path
    output_dir: Path
    metadata_db_path: Path
    model_path: str
    device: str
    limit: int
    segment_seconds: int
    frames_per_second: float
    top_k_per_segment: int
    dedupe_threshold: float
    min_side: int
    batch_size: int
    ffmpeg_binary: str
    image_format: str
    jpg_quality: int
    num_workers: int
    video_workers: int
    representative_segments_top_n: int
    importance_embedding_norm_weight: float
    importance_motion_score_weight: float
    importance_visual_diversity_weight: float
    genericness_weight: float
    diversity_penalty_weight: float

    @property
    def segments_dir(self) -> Path:
        return self.output_dir / "segments"

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "embeddings"

    @property
    def faiss_dir(self) -> Path:
        return self.output_dir / "faiss"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def faiss_index_path(self) -> Path:
        return self.faiss_dir / "frame_index.faiss"

    @property
    def faiss_meta_path(self) -> Path:
        return self.faiss_dir / "frame_index.meta.json"

    def faiss_paths(self, name: str) -> tuple[Path, Path]:
        return self.faiss_dir / f"{name}.faiss", self.faiss_dir / f"{name}.meta.json"

    @property
    def embeddings_npy_path(self) -> Path:
        return self.embeddings_dir / "frame_embeddings.npy"

    @property
    def embeddings_manifest_path(self) -> Path:
        return self.embeddings_dir / "frame_embeddings.jsonl"

    def frame_embedding_path(self, video_id: str, segment_id: str, frame_id: str) -> Path:
        return self.embeddings_dir / video_id / segment_id / f"{frame_id}.npy"

    def segment_embedding_path(self, video_id: str, segment_id: str) -> Path:
        return self.embeddings_dir / video_id / f"{segment_id}.npy"

    def video_embedding_path(self, video_id: str) -> Path:
        return self.embeddings_dir / f"{video_id}.npy"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal runnable video pipeline.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument(
        "--profile",
        default=DEFAULT_CLIP_PROFILE,
        help=f"CLIP storage profile (default: {DEFAULT_CLIP_PROFILE}).",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process the first N videos. 0 means all."
    )
    parser.add_argument("--segment-seconds", type=int, default=4)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--top-k-per-segment", type=int, default=4)
    parser.add_argument("--dedupe-threshold", type=float, default=0.98)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-workers", type=int, default=1)
    return parser.parse_args()


def build_config(args) -> MinimalConfig:
    return MinimalConfig(
        video_dir=Path(args.video_dir),
        output_dir=resolve_path(args.output_dir, default_clip_output_dir(args.profile)),
        metadata_db_path=resolve_path(
            args.metadata_db, default_clip_metadata_db_path(args.profile)
        ),
        model_path=args.model_path,
        device=args.device,
        limit=args.limit,
        segment_seconds=args.segment_seconds,
        frames_per_second=args.frames_per_second,
        top_k_per_segment=args.top_k_per_segment,
        dedupe_threshold=args.dedupe_threshold,
        min_side=args.min_side,
        batch_size=args.batch_size,
        ffmpeg_binary=args.ffmpeg_binary,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
        num_workers=args.num_workers,
        video_workers=args.video_workers,
        representative_segments_top_n=8,
        importance_embedding_norm_weight=1.0,
        importance_motion_score_weight=1.0,
        importance_visual_diversity_weight=1.0,
        genericness_weight=0.35,
        diversity_penalty_weight=0.25,
    )


def prepare_dirs(config: MinimalConfig):
    for path in (
        config.output_dir,
        config.segments_dir,
        config.frames_dir,
        config.embeddings_dir,
        config.faiss_dir,
        config.metadata_db_path.parent,
        config.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


_WORKER_STATE = threading.local()
_SHARED_ENCODER: ChineseClipEncoder | None = None
_ENCODER_LOCK = threading.RLock()


def use_shared_encoder(config: MinimalConfig) -> bool:
    return config.device.lower().startswith("cuda")


def load_encoder_class():
    from .embedding import ChineseClipEncoder

    return ChineseClipEncoder


def load_frame_selector_funcs():
    from .frame_selector import load_candidate_frames_from_dir, select_top_k_frames

    return load_candidate_frames_from_dir, select_top_k_frames


def load_segment_pooling():
    from .segment_embedding import softmax_attention_pooling

    return softmax_attention_pooling


def load_video_representation_funcs():
    from .video_representation import (
        SegmentRepresentation,
        compute_frame_diff_motion_score,
        compute_segment_importance,
        select_top_representative_segments,
    )

    return (
        SegmentRepresentation,
        compute_frame_diff_motion_score,
        compute_segment_importance,
        select_top_representative_segments,
    )


def _build_encoder(config: MinimalConfig) -> ChineseClipEncoder:
    return load_encoder_class()(
        model_path=config.model_path,
        device=config.device,
        batch_size=config.batch_size,
    )


def get_shared_encoder(config: MinimalConfig) -> ChineseClipEncoder:
    global _SHARED_ENCODER
    with _ENCODER_LOCK:
        if _SHARED_ENCODER is None:
            _SHARED_ENCODER = _build_encoder(config)
    return _SHARED_ENCODER


def get_encoder(config: MinimalConfig) -> ChineseClipEncoder:
    if use_shared_encoder(config):
        return get_shared_encoder(config)
    encoder = getattr(_WORKER_STATE, "encoder", None)
    if encoder is None:
        encoder = _build_encoder(config)
        _WORKER_STATE.encoder = encoder
    return encoder


def get_worker_runtime(config: MinimalConfig):
    store = getattr(_WORKER_STATE, "store", None)
    if store is None or getattr(_WORKER_STATE, "db_path", None) != config.metadata_db_path:
        store = MetadataStore(config.metadata_db_path)
        _WORKER_STATE.store = store
        _WORKER_STATE.db_path = config.metadata_db_path
        _WORKER_STATE.scheduler = PipelineScheduler(store)
        _WORKER_STATE.logger = build_logger(config.logs_dir, name="video_processing.minimal")
    return _WORKER_STATE.store, _WORKER_STATE.scheduler, _WORKER_STATE.logger


def frame_extract_workers(config: MinimalConfig) -> int:
    return max(1, config.num_workers // max(1, config.video_workers))


def encode_images(config: MinimalConfig, image_paths: list[str]):
    encoder = get_encoder(config)
    if use_shared_encoder(config):
        with _ENCODER_LOCK:
            return encoder.encode_images(image_paths)
    return encoder.encode_images(image_paths)


def stage_log(video_id: str, stage: str, **payload):
    parts = [f"video_id={video_id}", f"stage={stage}"]
    parts.extend(f"{key}={value}" for key, value in payload.items())
    print("[stage] " + " ".join(parts), flush=True)


def process_video(
    config: MinimalConfig, video_path: Path, video_idx: int, total_videos: int
) -> tuple[str, bool]:
    store, scheduler, logger = get_worker_runtime(config)
    print(f"[start] video {video_idx}/{total_videos}: {video_path.name}", flush=True)
    video_id = video_path.stem
    current_status = store.get_video_status(video_id)
    if current_status == "done":
        logger.info("skip done video_id=%s path=%s", video_id, video_path)
        return video_id, True
    stage_log(
        video_id,
        "init",
        device=config.device,
        video_workers=config.video_workers,
        frame_workers=frame_extract_workers(config),
    )
    encoder = get_encoder(config)
    stage_log(
        video_id, "encoder_ready", embedding_dim=encoder.embedding_dim, batch_size=config.batch_size
    )
    load_candidate_frames_from_dir, select_top_k_frames = load_frame_selector_funcs()
    softmax_attention_pooling = load_segment_pooling()
    (
        SegmentRepresentation,
        compute_frame_diff_motion_score,
        compute_segment_importance,
        select_top_representative_segments,
    ) = load_video_representation_funcs()

    duration = probe_video_duration(video_path)
    scheduler.start_video(video_id=video_id, path=str(video_path.resolve()), duration=duration)
    logger.info("start video_id=%s path=%s", video_id, video_path)
    try:
        segment_task_status = scheduler.task_status(video_id, "segment")
        if segment_task_status == "done":
            segment_paths = existing_segment_paths(config.segments_dir, video_id)
            if not scheduler.segment_outputs_ready(config.segments_dir, video_id):
                store.update_task(
                    video_id,
                    "segment",
                    "pending",
                    updated_at=None,
                    error_message="segment files missing",
                )
                segment_task_status = "pending"
        if segment_task_status != "done":
            store.clear_video_segments_and_frames(video_id)
            scheduler.reset_downstream_tasks(video_id, ("frame_extract", "embedding", "topk"))
            scheduler.start_task(video_id, "segment")
            try:
                segment_paths = segment_video(
                    video_path=video_path,
                    output_dir=config.segments_dir / video_id,
                    segment_seconds=config.segment_seconds,
                    ffmpeg_binary=config.ffmpeg_binary,
                )
                scheduler.complete_task(video_id, "segment")
                stage_log(
                    video_id, "segment", segments=len(segment_paths), seconds=config.segment_seconds
                )
            except Exception as exc:
                logger.exception("ffmpeg failed video_id=%s", video_id)
                append_stage_error(config.logs_dir, "ffmpeg", video_id=video_id, message=str(exc))
                scheduler.fail_task(video_id, "segment", str(exc))
                raise

        frame_task_status = scheduler.task_status(video_id, "frame_extract")
        ordered_segments = []
        if frame_task_status == "done":
            segment_ids = [segment_path.stem for segment_path in segment_paths]
            if not scheduler.frame_outputs_ready(
                config.frames_dir, video_id, segment_ids, config.image_format
            ):
                store.update_task(
                    video_id,
                    "frame_extract",
                    "pending",
                    updated_at=None,
                    error_message="frame files missing",
                )
                frame_task_status = "pending"
            for segment_idx, segment_path in enumerate(segment_paths):
                segment_id = segment_path.stem
                start_time = segment_idx * config.segment_seconds
                candidates = load_candidate_frames_from_dir(
                    segment_path=segment_path,
                    frames_dir=config.frames_dir / video_id / segment_id,
                    image_format=config.image_format,
                )
                ordered_segments.append((segment_idx, segment_path, start_time, candidates))
        if frame_task_status != "done":
            scheduler.reset_downstream_tasks(video_id, ("embedding", "topk"))
            scheduler.start_task(video_id, "frame_extract")
            future_to_segment = {}
            worker_count = frame_extract_workers(config)
            total_segments = len(segment_paths)
            extracted_segments = 0
            try:
                if worker_count == 1:
                    for segment_idx, segment_path in enumerate(segment_paths):
                        segment_id = segment_path.stem
                        start_time = segment_idx * config.segment_seconds
                        end_time = (
                            min(duration, start_time + config.segment_seconds)
                            if duration > 0
                            else start_time + config.segment_seconds
                        )
                        store.upsert_segment(
                            segment_id=segment_id,
                            video_id=video_id,
                            start_time=start_time,
                            end_time=end_time,
                            path=str(segment_path.resolve()),
                        )
                        candidates = extract_frames_for_segment(
                            segment_path,
                            config.frames_dir / video_id / segment_id,
                            config.frames_per_second,
                            config.min_side,
                            config.image_format,
                            config.jpg_quality,
                        )
                        ordered_segments.append((segment_idx, segment_path, start_time, candidates))
                        extracted_segments += 1
                        stage_log(
                            video_id,
                            "frame_extract",
                            progress=f"{extracted_segments}/{total_segments}",
                            segment_id=segment_id,
                            candidates=len(candidates),
                        )
                else:
                    with ProcessPoolExecutor(max_workers=worker_count) as executor:
                        for segment_idx, segment_path in enumerate(segment_paths):
                            segment_id = segment_path.stem
                            start_time = segment_idx * config.segment_seconds
                            end_time = (
                                min(duration, start_time + config.segment_seconds)
                                if duration > 0
                                else start_time + config.segment_seconds
                            )
                            store.upsert_segment(
                                segment_id=segment_id,
                                video_id=video_id,
                                start_time=start_time,
                                end_time=end_time,
                                path=str(segment_path.resolve()),
                            )
                            future = executor.submit(
                                extract_frames_for_segment,
                                segment_path,
                                config.frames_dir / video_id / segment_id,
                                config.frames_per_second,
                                config.min_side,
                                config.image_format,
                                config.jpg_quality,
                            )
                            future_to_segment[future] = (segment_idx, segment_path, start_time)

                        for future in as_completed(future_to_segment):
                            segment_idx, segment_path, start_time = future_to_segment[future]
                            candidates = future.result()
                            ordered_segments.append(
                                (segment_idx, segment_path, start_time, candidates)
                            )
                            extracted_segments += 1
                            stage_log(
                                video_id,
                                "frame_extract",
                                progress=f"{extracted_segments}/{total_segments}",
                                segment_id=segment_path.stem,
                                candidates=len(candidates),
                            )
                scheduler.complete_task(video_id, "frame_extract")
            except Exception as exc:
                scheduler.fail_task(video_id, "frame_extract", str(exc))
                raise

        ordered_segments.sort(key=lambda item: item[0])
        embedding_task_status = scheduler.task_status(video_id, "embedding")
        skip_embedding = False
        if embedding_task_status == "done":
            if video_embedding_outputs_ready(store, ordered_segments):
                skip_embedding = True
            else:
                store.update_task(
                    video_id,
                    "embedding",
                    "pending",
                    updated_at=None,
                    error_message="embedding files missing",
                )
                embedding_task_status = "pending"

        scheduler.start_task(video_id, "embedding")
        scheduler.start_task(video_id, "topk")
        segment_representations: list[SegmentRepresentation] = []
        if skip_embedding:
            segment_representations = segment_representations_from_store(
                store, video_id, ordered_segments
            )
            stage_log(
                video_id,
                "embedding",
                skipped="reuse_cached",
                segments=len(segment_representations),
            )
        else:
            total_segments = len(ordered_segments)
            embedded_segments = 0
            for _, segment_path, start_time, candidates in ordered_segments:
                segment_id = segment_path.stem
                if not candidates:
                    embedded_segments += 1
                    stage_log(
                        video_id,
                        "embedding",
                        progress=f"{embedded_segments}/{total_segments}",
                        segment_id=segment_id,
                        skipped="no_candidates",
                    )
                    continue

                try:
                    encoded = encode_images(config, [item.frame_path for item in candidates])
                except Exception as exc:
                    message = f"embedding failed for segment {segment_id}: {exc}"
                    logger.exception(message)
                    append_stage_error(
                        config.logs_dir,
                        "embedding",
                        video_id=video_id,
                        segment_id=segment_id,
                        message=str(exc),
                    )
                    scheduler.fail_task(video_id, "embedding", message)
                    raise RuntimeError(message) from exc

                selected = select_top_k_frames(
                    candidates=candidates,
                    embeddings=encoded.embeddings,
                    norms=encoded.norms,
                    top_k=config.top_k_per_segment,
                    dedupe_threshold=config.dedupe_threshold,
                )
                if not selected:
                    continue

                candidate_frame_paths = [
                    item.frame_path
                    for item in sorted(
                        candidates, key=lambda candidate: candidate.timestamp_seconds
                    )
                ]
                frame_diff_motion_score = compute_frame_diff_motion_score(candidate_frame_paths)

                for item in selected:
                    frame_id = f"{segment_id}_f{item.frame_index:06d}"
                    absolute_timestamp = start_time + item.timestamp_seconds
                    frame_embedding = item.embedding.astype(np.float32)
                    embedding_path = config.frame_embedding_path(video_id, segment_id, frame_id)
                    metadata_path = write_embedding_cache(
                        embedding_path=embedding_path,
                        embedding=frame_embedding,
                        item_id=frame_id,
                        model_name=encoder.model_name,
                        model_path=encoder.model_source_path,
                        model_revision=encoder.model_revision,
                        embedding_dim=encoder.embedding_dim,
                        embedding_dtype=encoder.embedding_dtype,
                        embedding_norm=item.embedding_norm,
                    )
                    store.upsert_frame(
                        frame_id=frame_id,
                        segment_id=segment_id,
                        timestamp=absolute_timestamp,
                        frame_path=item.frame_path,
                        embedding_path=str(embedding_path.resolve()),
                        embedding_metadata_path=str(metadata_path.resolve()),
                        frame_index=item.frame_index,
                        width=item.width,
                        height=item.height,
                        embedding_norm=item.embedding_norm,
                    )

                segment_matrix = np.vstack(
                    [item.embedding.astype(np.float32) for item in selected]
                ).astype(np.float32)
                segment_scores = np.array(
                    [item.embedding_norm for item in selected], dtype=np.float32
                )
                segment_timestamps = np.array(
                    [item.timestamp_seconds for item in selected], dtype=np.float32
                )
                importance_score, motion_score, visual_diversity_score, embedding_norm_score = (
                    compute_segment_importance(
                        frame_embeddings=segment_matrix,
                        frame_norms=segment_scores,
                        frame_timestamps=segment_timestamps,
                        segment_duration_seconds=float(config.segment_seconds),
                        frame_diff_motion_score=frame_diff_motion_score,
                        embedding_norm_weight=config.importance_embedding_norm_weight,
                        motion_score_weight=config.importance_motion_score_weight,
                        visual_diversity_weight=config.importance_visual_diversity_weight,
                    )
                )
                segment_embedding, _ = softmax_attention_pooling(segment_matrix, segment_scores)
                segment_embedding_path = config.segment_embedding_path(video_id, segment_id)
                segment_metadata_path = write_embedding_cache(
                    embedding_path=segment_embedding_path,
                    embedding=segment_embedding,
                    item_id=segment_id,
                    item_id_key="segment_id",
                    model_name=encoder.model_name,
                    model_path=encoder.model_source_path,
                    model_revision=encoder.model_revision,
                    embedding_dim=encoder.embedding_dim,
                    embedding_dtype=encoder.embedding_dtype,
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
                segment_representations.append(
                    SegmentRepresentation(
                        segment_id=segment_id,
                        video_id=video_id,
                        importance_score=importance_score,
                        motion_score=motion_score,
                        visual_diversity_score=visual_diversity_score,
                        embedding_norm_score=embedding_norm_score,
                    )
                )
                embedded_segments += 1
                stage_log(
                    video_id,
                    "embedding",
                    progress=f"{embedded_segments}/{total_segments}",
                    segment_id=segment_id,
                    selected_frames=len(selected),
                )

        selected_representations = (
            select_top_representative_segments(
                segment_representations,
                top_n=config.representative_segments_top_n,
            )
            if segment_representations
            else []
        )
        store.update_segment_representative_flags(
            video_id=video_id,
            selected_segment_ids=[item.segment_id for item in selected_representations],
        )

        scheduler.complete_task(video_id, "embedding")
        scheduler.complete_task(video_id, "topk")
        scheduler.complete_video(video_id)
        logger.info("done video_id=%s", video_id)
        print(
            f"[done] video_id={video_id} representative_segments={len(selected_representations)}",
            flush=True,
        )
        return video_id, True
    except Exception as exc:
        if scheduler.task_status(video_id, "topk") == "processing":
            scheduler.fail_task(video_id, "topk", str(exc))
        scheduler.fail_video(video_id, str(exc))
        logger.exception("failed video_id=%s", video_id)
        append_stage_error(config.logs_dir, "pipeline", video_id=video_id, message=str(exc))
        print(f"[failed] video_id={video_id} message={exc}", flush=True)
        return video_id, False


def finalize_indexes(config: MinimalConfig, video_ids: list[str]):
    store = MetadataStore(config.metadata_db_path)
    scheduler = PipelineScheduler(store)
    logger = build_logger(config.logs_dir, name="video_processing.minimal")
    encoder = get_encoder(config)
    scheduler.start_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE)
    refreshed_videos = 0
    try:
        refreshed_videos = refresh_representative_segments(
            store,
            video_ids,
            config.representative_segments_top_n,
            config.genericness_weight,
            config.diversity_penalty_weight,
        )
        records = store.get_all_frame_records()
        manifest_rows = []
        embedding_rows = []
        for item in records:
            embedding_path = Path(item["embedding_path"])
            if not embedding_path.exists():
                logger.warning(
                    "missing embedding file frame_id=%s path=%s", item["frame_id"], embedding_path
                )
                append_stage_error(
                    config.logs_dir,
                    "finalize",
                    video_id=item["video_id"],
                    frame_id=item["frame_id"],
                    message=f"missing embedding file: {embedding_path}",
                )
                continue
            embedding_rows.append(np.load(embedding_path).astype(np.float32))
            manifest_rows.append(
                {
                    "frame_id": item["frame_id"],
                    "video_id": item["video_id"],
                    "segment_id": item["segment_id"],
                    "timestamp": item["timestamp"],
                    "frame_index": item["frame_index"],
                    "frame_path": item["frame_path"],
                    "embedding_path": item["embedding_path"],
                    "embedding_metadata_path": item["embedding_metadata_path"],
                    "embedding_norm": item["embedding_norm"],
                }
            )

        full_embedding_matrix = (
            np.vstack(embedding_rows).astype(np.float32)
            if embedding_rows
            else np.zeros((0, encoder.embedding_dim), dtype=np.float32)
        )
        np.save(config.embeddings_npy_path, full_embedding_matrix)

        with config.embeddings_manifest_path.open("w", encoding="utf-8") as writer:
            for row in manifest_rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")

        rebuild_faiss_indexes(
            store,
            config.faiss_dir,
            normalize_embeddings=False,
        )
        scheduler.complete_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE)
    except Exception as exc:
        scheduler.fail_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE, str(exc))
        raise
    finally:
        store.close()

    print(f"Saved embeddings: {config.embeddings_npy_path}", flush=True)
    print(f"Saved embedding metadata: {config.embeddings_manifest_path}", flush=True)
    print(f"Saved FAISS index: {config.faiss_index_path}", flush=True)
    print(f"Refreshed representative segments: {refreshed_videos}", flush=True)
    print(f"Saved metadata DB: {config.metadata_db_path}", flush=True)
    print(f"Config: {asdict(config)}", flush=True)


def run_pipeline(config: MinimalConfig):
    prepare_dirs(config)
    store = MetadataStore(config.metadata_db_path)
    scheduler = PipelineScheduler(store)
    video_paths = list(iter_videos(config.video_dir))
    if config.limit > 0:
        video_paths = video_paths[: config.limit]
    video_ids = [video_path.stem for video_path in video_paths]
    total_videos = len(video_paths)
    print(
        f"[queue] total_videos={total_videos} profile_dir={config.output_dir} "
        f"device={config.device} video_workers={config.video_workers} frame_workers={frame_extract_workers(config)}",
        flush=True,
    )

    for video_path in video_paths:
        video_id = video_path.stem
        duration = probe_video_duration(video_path)
        scheduler.ensure_video(video_id=video_id, path=str(video_path.resolve()), duration=duration)
    scheduler.ensure_global_tasks()
    store.close()

    processed_videos = 0
    with ThreadPoolExecutor(max_workers=max(1, config.video_workers)) as executor:
        futures = [
            executor.submit(process_video, config, video_path, video_idx, total_videos)
            for video_idx, video_path in enumerate(video_paths, start=1)
        ]
        print(f"[queue] submitted_videos={len(futures)}", flush=True)
        for future in as_completed(futures):
            future.result()
            processed_videos += 1
            print(f"[progress] processed_videos={processed_videos}/{total_videos}", flush=True)

    finalize_indexes(config, video_ids)


def main():
    args = parse_args()
    config = build_config(args)
    start_ts = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[run] start_time={start_time}", flush=True)
    try:
        run_pipeline(config)
    finally:
        end_ts = time.time()
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_seconds = int(end_ts - start_ts)
        print(f"[run] end_time={end_time}", flush=True)
        print(f"[run] elapsed_seconds={elapsed_seconds}", flush=True)


if __name__ == "__main__":
    main()
