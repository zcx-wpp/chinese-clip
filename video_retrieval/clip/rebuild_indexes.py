from __future__ import annotations

import argparse

from ..config import DEFAULT_CLIP_PROFILE, DEFAULT_MODEL_PATH
from .index_builder import (
    rebuild_faiss_indexes,
    rebuild_segment_embeddings_from_frames,
)
from .metadata_store import MetadataStore
from .pipeline_utils import refresh_representative_segments
from ..profile_paths import (
    default_clip_metadata_db_path,
    default_clip_output_dir,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild frame/segment/video embeddings and FAISS indexes from cached frame embeddings."
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument(
        "--profile",
        default=DEFAULT_CLIP_PROFILE,
        help=f"CLIP storage profile (default: {DEFAULT_CLIP_PROFILE}).",
    )
    parser.add_argument("--model-name", default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--representative-top-n", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir, default_clip_output_dir(args.profile))
    metadata_db = resolve_path(args.metadata_db, default_clip_metadata_db_path(args.profile))
    store = MetadataStore(metadata_db)
    print(f"[rebuild] profile={args.profile} output_dir={output_dir}", flush=True)
    print(f"[rebuild] metadata_db={metadata_db}", flush=True)
    print(f"[rebuild] representative_top_n={args.representative_top_n}", flush=True)

    def progress(message: str) -> None:
        print(message.replace("[index]", "[rebuild]", 1), flush=True)

    frame_count, segment_count, video_count = rebuild_segment_embeddings_from_frames(
        store=store,
        output_dir=output_dir,
        model_name=args.model_name,
        model_path=args.model_path,
        model_revision=args.model_revision,
        representative_top_n=args.representative_top_n,
        on_progress=progress,
    )
    video_ids = list(dict.fromkeys(item["video_id"] for item in store.get_all_frame_records()))
    refreshed_video_count = refresh_representative_segments(
        store=store,
        video_ids=video_ids,
        top_n=args.representative_top_n,
        genericness_weight=0.35,
        diversity_penalty_weight=0.25,
    )
    print(
        f"[rebuild] refreshed representative segments with global penalties: videos={refreshed_video_count}",
        flush=True,
    )
    frame_index_count, segment_index_count, video_index_count = rebuild_faiss_indexes(
        store=store,
        faiss_dir=output_dir / "faiss",
        on_progress=progress,
    )

    print(
        f"Rebuilt embeddings from cached frames: frames={frame_count} segments={segment_count} videos={video_count}",
        flush=True,
    )
    print(
        f"Refreshed representative segments with global penalties: videos={refreshed_video_count}",
        flush=True,
    )
    print(
        f"Saved FAISS indexes: frame={frame_index_count} segment={segment_index_count} video={video_index_count}",
        flush=True,
    )
    print(f"FAISS dir: {output_dir / 'faiss'}", flush=True)


if __name__ == "__main__":
    main()
