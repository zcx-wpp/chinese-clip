from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .config import (
    EmbeddingConfig,
    FrameSamplingConfig,
    MultimodalConfig,
    PipelineConfig,
    RetrievalConfig,
    SegmentConfig,
    VectorStoreConfig,
)
from .embedding import ChineseClipEncoder
from .faiss_store import FaissFrameIndex
from .frame_selector import extract_candidate_frames, select_top_k_frames
from .metadata_store import MetadataStore
from .milvus_store import MilvusFrameIndex
from .multimodal import MultimodalSignals, build_asr_engine, build_ocr_engine
from .segmenter import iter_videos, segment_video


def parse_args():
    parser = argparse.ArgumentParser(description="Offline video vectorization pipeline.")
    parser.add_argument("--video-dir", required=True, help="Directory with source videos.")
    parser.add_argument("--work-dir", required=True, help="Directory for segments, frames and indexes.")
    parser.add_argument("--model-path", required=True, help="Local Chinese-CLIP model directory.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos. 0 means all.")
    parser.add_argument("--segment-seconds", type=int, default=8)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--top-k-per-segment", type=int, default=4)
    parser.add_argument("--dedupe-threshold", type=float, default=0.98)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--ocr-lang", default="ch")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    return parser.parse_args()


def build_config(args) -> PipelineConfig:
    return PipelineConfig(
        work_dir=Path(args.work_dir),
        video_dir=Path(args.video_dir),
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


def probe_video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0:
            return 0.0
        return float(total_frames / fps)
    finally:
        capture.release()


def probe_segment_durations(segment_paths: list[Path]) -> list[float]:
    return [probe_video_duration(segment_path) for segment_path in segment_paths]


def merge_modalities(vision_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    if text_embeddings.size == 0:
        return vision_embedding
    merged = np.vstack([vision_embedding[None, :], text_embeddings]).mean(axis=0)
    norm = np.linalg.norm(merged)
    return (merged / max(norm, 1e-12)).astype(np.float32)


def build_index(config: PipelineConfig):
    config.work_dir.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(config.metadata_db_path)
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
    limit = getattr(config, "limit", 0)
    if limit and limit > 0:
        video_paths = video_paths[:limit]
    total_videos = len(video_paths)
    total_indexed_frames = 0

    for video_idx, video_path in enumerate(video_paths, start=1):
        video_id = video_path.stem
        duration = probe_video_duration(video_path)
        store.upsert_video(
            video_id=video_id,
            title=video_path.stem,
            duration=duration,
            path=str(video_path.resolve()),
        )

        video_segment_dir = config.segments_dir / video_id
        segments = segment_video(
            video_path=video_path,
            output_dir=video_segment_dir,
            segment_seconds=config.segment.segment_seconds,
            ffmpeg_binary=config.segment.ffmpeg_binary,
        )
        segment_durations = probe_segment_durations(segments)
        asr_text = asr_engine.transcribe(video_path)

        cumulative_start_time = 0.0
        for segment_idx, segment_path in enumerate(segments):
            segment_id = segment_path.stem
            segment_duration = (
                segment_durations[segment_idx]
                if segment_idx < len(segment_durations)
                else 0.0
            )
            start_time = cumulative_start_time
            end_time = start_time + (
                segment_duration if segment_duration > 0 else config.segment.segment_seconds
            )
            if duration > 0:
                end_time = min(duration, end_time)

            store.upsert_segment(
                segment_id=segment_id,
                video_id=video_id,
                start_time=start_time,
                end_time=end_time,
                path=str(segment_path.resolve()),
            )

            segment_frame_dir = config.frames_dir / video_id / segment_id
            candidates = extract_candidate_frames(
                segment_path=segment_path,
                output_dir=segment_frame_dir,
                frames_per_second=config.sampling.frames_per_second,
                min_side=config.sampling.min_side,
                image_format=config.sampling.image_format,
                jpg_quality=config.sampling.jpg_quality,
            )
            if not candidates:
                cumulative_start_time = end_time
                continue

            encoded = encoder.encode_images([item.frame_path for item in candidates])
            selected = select_top_k_frames(
                candidates=candidates,
                embeddings=encoded.embeddings,
                norms=encoded.norms,
                top_k=config.sampling.top_k_per_segment,
                dedupe_threshold=config.sampling.dedupe_threshold,
            )
            if not selected:
                cumulative_start_time = end_time
                continue

            frame_ids = []
            merged_embeddings = []
            for item in selected:
                absolute_timestamp = start_time + item.timestamp_seconds
                ocr_text = ocr_engine.extract(Path(item.frame_path))
                signals = MultimodalSignals(ocr_text=ocr_text, asr_text=asr_text)
                merged_text = signals.merged_text()
                text_embeddings = (
                    encoder.encode_texts([merged_text])
                    if merged_text
                    else np.zeros((0, encoder.embedding_dim), dtype=np.float32)
                )
                final_embedding = merge_modalities(item.embedding, text_embeddings)
                frame_id = f"{segment_id}_f{item.frame_index:06d}"
                frame_ids.append(frame_id)
                merged_embeddings.append(final_embedding)
                store.insert_frame_embedding(
                    frame_id=frame_id,
                    video_id=video_id,
                    segment_id=segment_id,
                    timestamp=absolute_timestamp,
                    frame_index=item.frame_index,
                    modality="vision",
                    frame_path=item.frame_path,
                    extra={
                        "embedding_norm": item.embedding_norm,
                        "ocr_text": signals.ocr_text,
                        "asr_text": signals.asr_text,
                        "width": item.width,
                        "height": item.height,
                    },
                )

            index.add(
                frame_ids=frame_ids,
                embeddings=np.vstack(merged_embeddings).astype(np.float32),
            )
            total_indexed_frames += len(frame_ids)
            cumulative_start_time = end_time

        print(f"[progress] processed_videos={video_idx}/{total_videos}")

    if isinstance(index, FaissFrameIndex):
        index.persist(config.faiss_index_path, config.faiss_meta_path)
        print(f"Saved FAISS index: {config.faiss_index_path}")
    else:
        index.save()
        print(
            "Saved Milvus collection: "
            f"{config.vector_store.milvus_collection} @ {config.vector_store.milvus_uri}"
        )
    print(f"Saved metadata DB: {config.metadata_db_path}")
    print(f"Total indexed frames: {total_indexed_frames}")
    print(f"Config: {asdict(config)}")


def main():
    args = parse_args()
    config = build_config(args)
    config.limit = args.limit
    build_index(config)


if __name__ == "__main__":
    main()
