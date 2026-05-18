from __future__ import annotations

import argparse
from pathlib import Path

from .frame_selector import extract_candidate_frames
from .io_utils import write_json
from .segmenter import iter_videos


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos without indexing.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--min-side", type=int, default=128)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    manifest = []
    for video_path in iter_videos(video_dir):
        video_output_dir = output_dir / video_path.stem
        candidates = extract_candidate_frames(
            segment_path=video_path,
            output_dir=video_output_dir,
            frames_per_second=args.frames_per_second,
            min_side=args.min_side,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
        )
        manifest.append(
            {
                "video_id": video_path.stem,
                "video_path": str(video_path.resolve()),
                "saved_frames": len(candidates),
                "frames": [candidate.__dict__ for candidate in candidates],
            }
        )
        print(f"[ok] {video_path.name}: saved {len(candidates)} frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frame_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
