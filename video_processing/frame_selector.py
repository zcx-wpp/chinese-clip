from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameCandidate:
    frame_path: str
    timestamp_seconds: float
    frame_index: int
    width: int
    height: int


@dataclass
class SelectedFrame(FrameCandidate):
    embedding_norm: float
    embedding: np.ndarray


def extract_candidate_frames(
    segment_path: Path,
    output_dir: Path,
    frames_per_second: float,
    min_side: int,
    image_format: str,
    jpg_quality: int,
) -> list[FrameCandidate]:
    capture = cv2.VideoCapture(str(segment_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open segment: {segment_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, int(round(fps / frames_per_second))) if fps > 0 else 1
    candidates = []
    frame_index = -1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % step != 0:
                continue
            height, width = frame.shape[:2]
            if min(width, height) < min_side:
                continue
            frame_name = f"{segment_path.stem}_f{frame_index:06d}.{image_format}"
            frame_path = output_dir / frame_name
            if image_format == "jpg":
                saved = cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                saved = cv2.imwrite(str(frame_path), frame)
            if not saved:
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            timestamp = frame_index / fps if fps > 0 else 0.0
            candidates.append(
                FrameCandidate(
                    frame_path=str(frame_path.resolve()),
                    timestamp_seconds=round(timestamp, 3),
                    frame_index=frame_index,
                    width=width,
                    height=height,
                )
            )
    finally:
        capture.release()

    return candidates


def select_top_k_frames(
    candidates: list[FrameCandidate],
    embeddings: np.ndarray,
    norms: np.ndarray,
    top_k: int,
    dedupe_threshold: float,
) -> list[SelectedFrame]:
    if not candidates:
        return []

    order = np.argsort(-norms)
    selected: list[SelectedFrame] = []
    selected_embeddings: list[np.ndarray] = []

    for idx in order:
        candidate_embedding = embeddings[idx]
        if selected_embeddings:
            sims = [float(candidate_embedding @ other) for other in selected_embeddings]
            if max(sims) >= dedupe_threshold:
                continue
        item = candidates[idx]
        selected.append(
            SelectedFrame(
                frame_path=item.frame_path,
                timestamp_seconds=item.timestamp_seconds,
                frame_index=item.frame_index,
                width=item.width,
                height=item.height,
                embedding_norm=float(norms[idx]),
                embedding=candidate_embedding,
            )
        )
        selected_embeddings.append(candidate_embedding)
        if len(selected) >= top_k:
            break

    return selected
