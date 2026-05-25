"""Unified indexing pipeline: CLIP visual vectors and/or hybrid text index."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .clip.minimal_pipeline import (
    build_config as build_clip_config,
)
from .clip.minimal_pipeline import (
    prepare_dirs,
)
from .clip.minimal_pipeline import (
    run_pipeline as run_clip_pipeline,
)
from .config import (
    DEFAULT_CLIP_PROFILE,
    DEFAULT_HYBRID_PROFILE,
    DEFAULT_MODEL_PATH,
    DEFAULT_VIDEO_DIR,
    PIPELINE_DEFAULTS,
)
from .env_loader import load_default_dotenv_files
from .hybrid.build_hybrid_index import build_index, load_records, sync_records_to_store
from .hybrid.doubao_batch_caption import build_config as build_caption_config
from .hybrid.doubao_batch_caption import run_batch
from .hybrid.metadata_store import MetadataStore
from .logging_utils import build_logger
from .profile_paths import ensure_unified_profile_dirs, resolve_profile_layout
from .schemas import PipelineStep


def parse_steps(raw: str) -> list[PipelineStep]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("steps must not be empty")
    allowed = {"clip", "hybrid", "both"}
    for part in parts:
        if part not in allowed:
            raise ValueError(f"invalid step: {part} (allowed: clip, hybrid, both)")
    if "both" in parts:
        return ["clip", "hybrid"]
    return parts  # type: ignore[return-value]


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified video indexing: CLIP visual vectors and/or Doubao hybrid text index."
    )
    p.add_argument(
        "--profile",
        help=(
            "Base profile for both pipelines when clip/hybrid profiles omitted; "
            f"defaults to clip={DEFAULT_CLIP_PROFILE}, hybrid={DEFAULT_HYBRID_PROFILE}."
        ),
    )
    p.add_argument(
        "--clip-profile",
        help=f"CLIP storage profile (default: --profile or {DEFAULT_CLIP_PROFILE}).",
    )
    p.add_argument(
        "--hybrid-profile",
        help=f"Hybrid storage profile (default: --profile or {DEFAULT_HYBRID_PROFILE}).",
    )
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    p.add_argument(
        "--steps",
        default="both",
        help="Comma-separated: clip, hybrid, both (both = clip then hybrid).",
    )
    p.add_argument(
        "--init-unified",
        action="store_true",
        help="Create unified profile directories before indexing.",
    )
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--clip-device", default="cuda")
    p.add_argument("--hybrid-device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Max videos per step. 0 = all.")
    p.add_argument("--segment-seconds", type=int, default=PIPELINE_DEFAULTS.segment_seconds)
    p.add_argument("--frames-per-second", type=float, default=PIPELINE_DEFAULTS.frames_per_second)
    p.add_argument("--top-k-per-segment", type=int, default=PIPELINE_DEFAULTS.top_k_per_segment)
    p.add_argument("--clip-batch-size", type=int, default=PIPELINE_DEFAULTS.clip_batch_size)
    p.add_argument("--hybrid-workers", type=int, default=PIPELINE_DEFAULTS.hybrid_workers)
    p.add_argument(
        "--hybrid-sample-frames", type=int, default=PIPELINE_DEFAULTS.hybrid_sample_frames
    )
    p.add_argument("--overwrite-captions", action="store_true")
    p.add_argument("--retry-failed-captions", action="store_true")
    p.add_argument("--allow-legacy-caption-only", action="store_true")
    p.add_argument("--hybrid-local-files-only", action="store_true")
    p.add_argument("--embedding-model", default=None)
    return p.parse_args()


def run_clip_index(layout, *, video_dir: Path, args) -> None:
    logger = build_logger(layout.clip_logs_dir, name="video_retrieval.clip")
    logger.info("clip index start profile=%s", layout.clip_profile)
    ns = SimpleNamespace(
        video_dir=str(video_dir),
        output_dir=str(layout.clip_output_dir),
        metadata_db=str(layout.clip_metadata_db),
        profile=layout.clip_profile,
        model_path=args.model_path,
        device=args.clip_device,
        limit=args.limit,
        segment_seconds=args.segment_seconds,
        frames_per_second=args.frames_per_second,
        top_k_per_segment=args.top_k_per_segment,
        dedupe_threshold=0.98,
        min_side=128,
        batch_size=args.clip_batch_size,
        ffmpeg_binary="ffmpeg",
        image_format="jpg",
        jpg_quality=95,
        num_workers=2,
        video_workers=1,
    )
    config = build_clip_config(ns)
    prepare_dirs(config)
    run_clip_pipeline(config)
    logger.info("clip index done")


def run_hybrid_steps(layout, *, video_dir: Path, args) -> None:
    from .config import DEFAULT_BGE_MODEL
    from .hybrid.dense_embeddings import DEFAULT_BGE_QUERY_INSTRUCTION

    logger = build_logger(layout.hybrid_logs_dir, name="video_retrieval.hybrid")
    layout.hybrid_captions_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger.info("hybrid caption start profile=%s", layout.hybrid_profile)
    cap_ns = SimpleNamespace(
        video_dir=str(video_dir),
        output_jsonl=str(layout.hybrid_captions_jsonl),
        output_json=None,
        profile=layout.hybrid_profile,
        metadata_db=str(layout.hybrid_metadata_db),
        no_sync_metadata_db=False,
        video_ids=None,
        video_index_start=None,
        video_index_end=None,
        model=None,
        legacy_model=None,
        api_key_env="ARK_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_mode="chat",
        limit=args.limit,
        workers=args.hybrid_workers,
        sample_frames=args.hybrid_sample_frames,
        max_side=640,
        jpg_quality=85,
        temperature=0.2,
        max_tokens=180,
        timeout_seconds=180.0,
        prompt_file=None,
        scene_scan_stride_seconds=2.0,
        scene_change_threshold=0.35,
        progress_interval_seconds=30.0,
        overwrite=args.overwrite_captions,
        retry_failed=args.retry_failed_captions,
    )
    run_batch(build_caption_config(cap_ns))

    logger.info("hybrid index start profile=%s", layout.hybrid_profile)
    store = MetadataStore(layout.hybrid_metadata_db)
    try:
        records = load_records(
            layout.hybrid_captions_jsonl,
            store,
            args.limit,
            args.allow_legacy_caption_only,
        )
        sync_records_to_store(store, records)
        build_index(
            records,
            store,
            layout.hybrid_index_dir,
            embedding_model=args.embedding_model or DEFAULT_BGE_MODEL,
            embedding_device=args.hybrid_device,
            embedding_batch_size=16,
            embedding_max_length=512,
            embedding_query_instruction=DEFAULT_BGE_QUERY_INSTRUCTION,
            embedding_local_files_only=args.hybrid_local_files_only,
        )
    finally:
        store.close()
    logger.info("hybrid index done")


def main():
    load_default_dotenv_files()
    args = parse_args()
    steps = parse_steps(args.steps)

    clip_profile = args.clip_profile or args.profile or DEFAULT_CLIP_PROFILE
    hybrid_profile = args.hybrid_profile or args.profile or DEFAULT_HYBRID_PROFILE
    layout = resolve_profile_layout(
        args.profile,
        clip_profile=clip_profile,
        hybrid_profile=hybrid_profile,
    )
    if args.init_unified:
        ensure_unified_profile_dirs(layout)
        layout = resolve_profile_layout(
            args.profile,
            clip_profile=clip_profile,
            hybrid_profile=hybrid_profile,
        )

    logger = build_logger(layout.clip_logs_dir, name="video_retrieval")
    logger.info(
        "pipeline start steps=%s profiles clip=%s hybrid=%s",
        steps,
        layout.clip_profile,
        layout.hybrid_profile,
    )
    print(f"[run] start_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    if "clip" in steps:
        print("[step] CLIP index", flush=True)
        run_clip_index(layout, video_dir=args.video_dir, args=args)

    if "hybrid" in steps:
        print("[step] Hybrid caption + index", flush=True)
        run_hybrid_steps(layout, video_dir=args.video_dir, args=args)

    print(f"[run] end_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    logger.info("pipeline done")


if __name__ == "__main__":
    main()
