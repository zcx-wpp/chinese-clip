from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DEFAULT_CLIP_PROFILE
from .metadata_store import MetadataStore
from ..profile_paths import default_clip_metadata_db_path
from .scheduler import GLOBAL_FAISS_STAGE, GLOBAL_FAISS_VIDEO_ID, PipelineScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Reset a video processing task back to pending.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_CLIP_PROFILE,
        help=f"CLIP profile when --metadata-db is omitted (default: {DEFAULT_CLIP_PROFILE}).",
    )
    parser.add_argument("--metadata-db")
    parser.add_argument("--video-id", help="Video id to reset.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["segment", "frame_extract", "embedding", "topk", "faiss"],
    )
    parser.add_argument(
        "--no-downstream", action="store_true", help="Only reset the specified stage."
    )
    parser.add_argument(
        "--reset-video-status", action="store_true", help="Also reset the video status to pending."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_db = Path(
        args.metadata_db or default_clip_metadata_db_path(args.profile)
    )
    store = MetadataStore(metadata_db)
    scheduler = PipelineScheduler(store)
    include_downstream = not args.no_downstream

    if args.stage == "faiss":
        scheduler.reset_task(
            video_id=GLOBAL_FAISS_VIDEO_ID,
            stage=GLOBAL_FAISS_STAGE,
            include_downstream=include_downstream,
        )
        print("Reset global faiss task to pending")
        return

    if not args.video_id:
        raise SystemExit("--video-id is required when stage is not faiss")

    scheduler.reset_task(
        video_id=args.video_id,
        stage=args.stage,
        include_downstream=include_downstream,
    )
    if args.reset_video_status:
        scheduler.reset_video_status(args.video_id)
    print(
        f"Reset task: video_id={args.video_id} stage={args.stage} "
        f"include_downstream={include_downstream} reset_video_status={args.reset_video_status}"
    )


if __name__ == "__main__":
    main()
