from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import (
    EmbeddingConfig,
    FrameSamplingConfig,
    MultimodalConfig,
    PROJECT_ROOT,
    PipelineConfig,
    RetrievalConfig,
    SegmentConfig,
    VectorStoreConfig,
)
from .embedding_cache import write_embedding_cache
from .embedding import ChineseClipEncoder
from .faiss_store import FaissFrameIndex
from .frame_selector import load_candidate_frames_from_dir, select_top_k_frames
from .logging_utils import build_logger
from .metadata_store import MetadataStore
from .milvus_store import MilvusFrameIndex
from .multimodal import MultimodalSignals, build_asr_engine, build_ocr_engine
from .pipeline_utils import (
    append_stage_error,
    existing_segment_paths,
    extract_frames_for_segment,
    load_existing_embedding,
    persist_faiss_index,
    probe_video_duration,
    refresh_representative_segments,
)
from .scheduler import GLOBAL_FAISS_STAGE, GLOBAL_FAISS_VIDEO_ID, PipelineScheduler
from .segment_embedding import softmax_attention_pooling
from .segmenter import iter_videos, segment_video
from .video_representation import (
    SegmentRepresentation,
    compute_frame_diff_motion_score,
    compute_segment_importance,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline video vectorization pipeline.")
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "videos"), help="Directory with source videos.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output"), help="Directory for segments, frames, faiss and logs.")
    parser.add_argument("--metadata-dir", default=str(PROJECT_ROOT / "metadata"), help="Directory for metadata.db.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"), help="Local Chinese-CLIP model directory.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos. 0 means all.")
    parser.add_argument("--segment-seconds", type=int, default=4)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--top-k-per-segment", type=int, default=4)
    parser.add_argument("--dedupe-threshold", type=float, default=0.98)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--ocr-lang", default="ch")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    return PipelineConfig(
        project_root=PROJECT_ROOT,
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
        metadata_dir=Path(args.metadata_dir),
        models_dir=Path(args.model_path),
        num_workers=args.num_workers,
        segment=SegmentConfig(segment_seconds=args.segment_seconds),
        sampling=FrameSamplingConfig(
            frames_per_second=args.frames_per_second,
            top_k_per_segment=args.top_k_per_segment,
            dedupe_threshold=args.dedupe_threshold,
            min_side=args.min_side,
        ),
        embedding=EmbeddingConfig(
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
        ),
        multimodal=MultimodalConfig(
            enable_ocr=args.enable_ocr,
            enable_asr=args.enable_asr,
            ocr_lang=args.ocr_lang,
            whisper_model=args.whisper_model,
        ),
        retrieval=RetrievalConfig(),
        vector_store=VectorStoreConfig(
            backend=args.vector_backend,
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
            milvus_collection=args.milvus_collection,
        ),
    )


def build_vector_index(config: PipelineConfig, embedding_dim: int):
    if config.vector_store.backend == "milvus":
        return MilvusFrameIndex(
            uri=config.vector_store.milvus_uri,
            token=config.vector_store.milvus_token,
            collection_name=config.vector_store.milvus_collection,
            dim=embedding_dim,
            index_type=config.vector_store.milvus_index_type,
            metric_type=config.vector_store.milvus_metric_type,
            m=config.vector_store.milvus_m,
            ef_construction=config.vector_store.milvus_ef_construction,
        )
    return FaissFrameIndex(dim=embedding_dim)


def probe_segment_durations(segment_paths: list[Path]) -> list[float]:
    return [probe_video_duration(segment_path) for segment_path in segment_paths]


def merge_modalities(vision_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    if text_embeddings.size == 0:
        return vision_embedding
    merged = np.vstack([vision_embedding[None, :], text_embeddings]).mean(axis=0)
    norm = np.linalg.norm(merged)
    return (merged / max(norm, 1e-12)).astype(np.float32)


def frame_embedding_path(config: PipelineConfig, video_id: str, segment_id: str, frame_id: str) -> Path:
    return config.embeddings_dir / video_id / segment_id / f"{frame_id}.npy"


def segment_embedding_path(config: PipelineConfig, video_id: str, segment_id: str) -> Path:
    return config.embeddings_dir / video_id / f"{segment_id}.npy"


def build_index(config: PipelineConfig, limit: int = 0):
    for path in (
        config.output_dir,
        config.metadata_dir,
        config.segments_dir,
        config.frames_dir,
        config.embeddings_dir,
        config.faiss_dir,
        config.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    store = MetadataStore(config.metadata_db_path)
    scheduler = PipelineScheduler(store)
    logger = build_logger(config.logs_dir, name="video_processing.offline")
    encoder = ChineseClipEncoder(
        model_path=config.embedding.model_path,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )
    index = build_vector_index(config, embedding_dim=encoder.embedding_dim)
    ocr_engine = build_ocr_engine(
        enable_ocr=config.multimodal.enable_ocr,
        lang=config.multimodal.ocr_lang,
    )
    asr_engine = build_asr_engine(
        enable_asr=config.multimodal.enable_asr,
        model_name=config.multimodal.whisper_model,
    )
    video_paths = list(iter_videos(config.video_dir))
    if limit and limit > 0:
        video_paths = video_paths[:limit]
    video_ids = [video_path.stem for video_path in video_paths]
    total_videos = len(video_paths)
    total_indexed_frames = 0
    num_workers = getattr(config, "num_workers", 2)

    for video_path in video_paths:
        video_id = video_path.stem
        duration = probe_video_duration(video_path)
        scheduler.ensure_video(video_id=video_id, path=str(video_path.resolve()), duration=duration)
    scheduler.ensure_global_tasks()

    for video_idx, video_path in enumerate(video_paths, start=1):
        video_id = video_path.stem
        current_status = store.get_video_status(video_id)
        if current_status == "done":
            logger.info("skip done video_id=%s path=%s", video_id, video_path)
            print(f"[progress] processed_videos={video_idx}/{total_videos}", flush=True)
            continue
        print(
            f"[start] video {video_idx}/{total_videos}: {video_path.name}",
            flush=True,
        )
        duration = probe_video_duration(video_path)
        scheduler.start_video(video_id=video_id, path=str(video_path.resolve()), duration=duration)
        logger.info("start video_id=%s path=%s", video_id, video_path)
        try:
            video_segment_dir = config.segments_dir / video_id
            segment_task_status = scheduler.task_status(video_id, "segment")
            if segment_task_status == "done":
                segments = existing_segment_paths(config.segments_dir, video_id)
                if not scheduler.segment_outputs_ready(config.segments_dir, video_id):
                    store.update_task(video_id, "segment", "pending", updated_at=None, error_message="segment files missing")
                    segment_task_status = "pending"
            if segment_task_status != "done":
                store.clear_video_segments_and_frames(video_id)
                scheduler.reset_downstream_tasks(video_id, ("frame_extract", "embedding", "topk"))
                scheduler.start_task(video_id, "segment")
                try:
                    segments = segment_video(
                        video_path=video_path,
                        output_dir=video_segment_dir,
                        segment_seconds=config.segment.segment_seconds,
                        ffmpeg_binary=config.segment.ffmpeg_binary,
                    )
                    scheduler.complete_task(video_id, "segment")
                except Exception as exc:
                    logger.exception("ffmpeg failed video_id=%s", video_id)
                    append_stage_error(config.logs_dir, "ffmpeg", video_id=video_id, message=str(exc))
                    scheduler.fail_task(video_id, "segment", str(exc))
                    raise
            segment_durations = probe_segment_durations(segments)
            asr_text = asr_engine.transcribe(video_path)

            frame_task_status = scheduler.task_status(video_id, "frame_extract")
            ordered_segments = []
            cumulative_start_time = 0.0
            if frame_task_status == "done":
                segment_ids = [segment_path.stem for segment_path in segments]
                if not scheduler.frame_outputs_ready(config.frames_dir, video_id, segment_ids, config.sampling.image_format):
                    store.update_task(video_id, "frame_extract", "pending", updated_at=None, error_message="frame files missing")
                    frame_task_status = "pending"
                for segment_idx, segment_path in enumerate(segments):
                    segment_duration = segment_durations[segment_idx] if segment_idx < len(segment_durations) else 0.0
                    start_time = cumulative_start_time
                    end_time = start_time + (segment_duration if segment_duration > 0 else config.segment.segment_seconds)
                    if duration > 0:
                        end_time = min(duration, end_time)
                    segment_id = segment_path.stem
                    candidates = load_candidate_frames_from_dir(
                        segment_path=segment_path,
                        frames_dir=config.frames_dir / video_id / segment_id,
                        image_format=config.sampling.image_format,
                    )
                    ordered_segments.append((segment_idx, segment_path, start_time, end_time, candidates))
                    cumulative_start_time = end_time
            if frame_task_status != "done":
                scheduler.reset_downstream_tasks(video_id, ("embedding", "topk"))
                scheduler.start_task(video_id, "frame_extract")
                future_to_segment = {}
                cumulative_start_time = 0.0
                try:
                    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
                        for segment_idx, segment_path in enumerate(segments):
                            segment_id = segment_path.stem
                            segment_duration = segment_durations[segment_idx] if segment_idx < len(segment_durations) else 0.0
                            start_time = cumulative_start_time
                            end_time = start_time + (segment_duration if segment_duration > 0 else config.segment.segment_seconds)
                            if duration > 0:
                                end_time = min(duration, end_time)

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
                                config.sampling.frames_per_second,
                                config.sampling.min_side,
                                config.sampling.image_format,
                                config.sampling.jpg_quality,
                            )
                            future_to_segment[future] = (segment_idx, segment_path, start_time, end_time)
                            cumulative_start_time = end_time

                        for future in as_completed(future_to_segment):
                            segment_idx, segment_path, start_time, end_time = future_to_segment[future]
                            candidates = future.result()
                            ordered_segments.append((segment_idx, segment_path, start_time, end_time, candidates))
                    scheduler.complete_task(video_id, "frame_extract")
                except Exception as exc:
                    scheduler.fail_task(video_id, "frame_extract", str(exc))
                    raise

            ordered_segments.sort(key=lambda item: item[0])
            scheduler.start_task(video_id, "embedding")
            scheduler.start_task(video_id, "topk")
            segment_representations: list[SegmentRepresentation] = []
            for _, segment_path, start_time, end_time, candidates in ordered_segments:
                segment_id = segment_path.stem
                if not candidates:
                    continue

                try:
                    encoded = encoder.encode_images([item.frame_path for item in candidates])
                except Exception as exc:
                    message = f"embedding failed for segment {segment_id}: {exc}"
                    logger.exception(message)
                    append_stage_error(config.logs_dir, "embedding", video_id=video_id, segment_id=segment_id, message=str(exc))
                    scheduler.fail_task(video_id, "embedding", message)
                    raise RuntimeError(message) from exc

                selected = select_top_k_frames(
                    candidates=candidates,
                    embeddings=encoded.embeddings,
                    norms=encoded.norms,
                    top_k=config.sampling.top_k_per_segment,
                    dedupe_threshold=config.sampling.dedupe_threshold,
                )
                if not selected:
                    continue

                candidate_frame_paths = [item.frame_path for item in sorted(candidates, key=lambda candidate: candidate.timestamp_seconds)]
                frame_diff_motion_score = compute_frame_diff_motion_score(candidate_frame_paths)

                frame_ids = []
                merged_embeddings = []
                for item in selected:
                    absolute_timestamp = start_time + item.timestamp_seconds
                    ocr_text = ocr_engine.extract(Path(item.frame_path))
                    signals = MultimodalSignals(ocr_text=ocr_text, asr_text=asr_text)
                    merged_text = signals.merged_text()
                    text_embeddings = encoder.encode_texts([merged_text]) if merged_text else np.zeros((0, encoder.embedding_dim), dtype=np.float32)
                    final_embedding = merge_modalities(item.embedding, text_embeddings)
                    frame_id = f"{segment_id}_f{item.frame_index:06d}"
                    frame_ids.append(frame_id)
                    merged_embeddings.append(final_embedding)
                    embedding_path = frame_embedding_path(config, video_id, segment_id, frame_id)
                    metadata_path = write_embedding_cache(
                        embedding_path=embedding_path,
                        embedding=final_embedding.astype(np.float32),
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

                segment_matrix = np.vstack(merged_embeddings).astype(np.float32)
                segment_scores = np.array([item.embedding_norm for item in selected], dtype=np.float32)
                segment_timestamps = np.array([item.timestamp_seconds for item in selected], dtype=np.float32)
                importance_score, motion_score, visual_diversity_score, embedding_norm_score = compute_segment_importance(
                    frame_embeddings=segment_matrix,
                    frame_norms=segment_scores,
                    frame_timestamps=segment_timestamps,
                    segment_duration_seconds=float(max(end_time - start_time, 0.0)),
                    frame_diff_motion_score=frame_diff_motion_score,
                    embedding_norm_weight=config.video_representation.importance_embedding_norm_weight,
                    motion_score_weight=config.video_representation.importance_motion_score_weight,
                    visual_diversity_weight=config.video_representation.importance_visual_diversity_weight,
                )
                segment_embedding, _ = softmax_attention_pooling(segment_matrix, segment_scores)
                current_segment_embedding_path = segment_embedding_path(config, video_id, segment_id)
                current_segment_metadata_path = write_embedding_cache(
                    embedding_path=current_segment_embedding_path,
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
                    embedding_path=str(current_segment_embedding_path.resolve()),
                    embedding_metadata_path=str(current_segment_metadata_path.resolve()),
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

                index.add(frame_ids=frame_ids, embeddings=np.vstack(merged_embeddings).astype(np.float32))
                total_indexed_frames += len(frame_ids)

            selected_representations = select_top_representative_segments(
                segment_representations,
                top_n=config.video_representation.representative_segments_top_n,
            ) if segment_representations else []
            store.update_segment_representative_flags(
                video_id=video_id,
                selected_segment_ids=[item.segment_id for item in selected_representations],
            )

            scheduler.complete_task(video_id, "embedding")
            scheduler.complete_task(video_id, "topk")
            scheduler.complete_video(video_id)
            logger.info("done video_id=%s", video_id)
        except Exception as exc:
            if scheduler.task_status(video_id, "topk") == "processing":
                scheduler.fail_task(video_id, "topk", str(exc))
            scheduler.fail_video(video_id, str(exc))
            logger.exception("failed video_id=%s", video_id)
            append_stage_error(config.logs_dir, "pipeline", video_id=video_id, message=str(exc))
        print(f"[progress] processed_videos={video_idx}/{total_videos}", flush=True)

    if isinstance(index, FaissFrameIndex):
        scheduler.start_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE)
        refreshed_videos = 0
        try:
            refreshed_videos = refresh_representative_segments(
                store,
                video_ids,
                config.video_representation.representative_segments_top_n,
                config.video_representation.genericness_weight,
                config.video_representation.diversity_penalty_weight,
            )
            records = store.get_all_frame_records()
            unique_segment_ids = list(dict.fromkeys(item["segment_id"] for item in records))
            segment_records = store.get_segment_records(unique_segment_ids)
            frame_ids = []
            embedding_rows = []
            for item in records:
                embedding_path = Path(item["embedding_path"])
                if not embedding_path.exists():
                    logger.warning("missing embedding file frame_id=%s path=%s", item["frame_id"], embedding_path)
                    append_stage_error(
                        config.logs_dir,
                        "finalize",
                        video_id=item["video_id"],
                        frame_id=item["frame_id"],
                        message=f"missing embedding file: {embedding_path}",
                    )
                    continue
                frame_ids.append(item["frame_id"])
                embedding_rows.append(np.load(embedding_path).astype(np.float32))

            persist_faiss_index(
                encoder.embedding_dim,
                frame_ids,
                embedding_rows,
                config.faiss_index_path,
                config.faiss_meta_path,
            )

            segment_ids = []
            segment_embedding_rows = []
            for item in segment_records:
                embedding = load_existing_embedding(item.get("embedding_path"))
                if embedding is None:
                    continue
                segment_ids.append(item["segment_id"])
                segment_embedding_rows.append(embedding)
            persist_faiss_index(
                encoder.embedding_dim,
                segment_ids,
                segment_embedding_rows,
                config.segment_index_path,
                config.segment_meta_path,
            )

            representative_segment_records = store.get_representative_segment_records_by_video_ids(video_ids)
            video_item_ids = []
            video_embedding_rows = []
            for item in representative_segment_records:
                embedding = load_existing_embedding(item.get("embedding_path"))
                if embedding is None:
                    continue
                video_item_ids.append(f"{item['video_id']}::{item['segment_id']}")
                video_embedding_rows.append(embedding)
            persist_faiss_index(
                encoder.embedding_dim,
                video_item_ids,
                video_embedding_rows,
                config.video_index_path,
                config.video_meta_path,
            )
            scheduler.complete_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE)
            print(f"Saved FAISS index: {config.faiss_index_path}", flush=True)
            print(f"Refreshed representative segments: {refreshed_videos}", flush=True)
        except Exception as exc:
            scheduler.fail_task(GLOBAL_FAISS_VIDEO_ID, GLOBAL_FAISS_STAGE, str(exc))
            raise
    else:
        index.save()
        print(f"Saved Milvus collection: {config.vector_store.milvus_collection} @ {config.vector_store.milvus_uri}", flush=True)
    print(f"Saved metadata DB: {config.metadata_db_path}", flush=True)
    print(f"Total indexed frames: {total_indexed_frames}", flush=True)
    print(f"Config: {asdict(config)}", flush=True)


def main():
    args = parse_args()
    config = build_config(args)
    build_index(config, limit=args.limit)


if __name__ == "__main__":
    main()
