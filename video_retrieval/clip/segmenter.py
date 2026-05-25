from __future__ import annotations

import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def iter_videos(video_dir: Path):
    for path in sorted(video_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _run_ffmpeg(command: list[str], video_path: Path):
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"ffmpeg segment failed for {video_path}: {details}") from exc


def segment_video(
    video_path: Path, output_dir: Path, segment_seconds: int, ffmpeg_binary: str = "ffmpeg"
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video_path.stem}_seg_%04d.mp4"
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(video_path),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    _run_ffmpeg(command, video_path)
    return sorted(output_dir.glob(f"{video_path.stem}_seg_*.mp4"))
