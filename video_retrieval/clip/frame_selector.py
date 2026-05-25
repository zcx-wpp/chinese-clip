from __future__ import annotations

import re
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


FRAME_INDEX_PATTERN = re.compile(r"_f(\d+)$")


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _pairwise_cosine_similarity(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return vectors @ centers.T


def _initialize_kmeans_centers(
    embeddings: np.ndarray, norms: np.ndarray, cluster_count: int
) -> np.ndarray:
    first_idx = int(np.argmax(norms))
    chosen = [first_idx]
    while len(chosen) < cluster_count:
        chosen_vectors = embeddings[chosen]
        similarities = embeddings @ chosen_vectors.T
        max_similarity = similarities.max(axis=1)
        max_similarity[chosen] = 1.0
        next_idx = int(np.argmin(max_similarity))
        if next_idx in chosen:
            break
        chosen.append(next_idx)
    while len(chosen) < cluster_count:
        chosen.append(chosen[-1])
    return embeddings[chosen].copy()


def _run_kmeans(
    embeddings: np.ndarray, norms: np.ndarray, cluster_count: int, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    centers = _initialize_kmeans_centers(embeddings, norms, cluster_count)
    assignments = np.zeros((embeddings.shape[0],), dtype=np.int32)

    for _ in range(max(1, iterations)):
        similarities = _pairwise_cosine_similarity(embeddings, centers)
        new_assignments = similarities.argmax(axis=1).astype(np.int32)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_idx in range(cluster_count):
            mask = assignments == cluster_idx
            if not np.any(mask):
                fallback_idx = int(np.argmax(norms))
                centers[cluster_idx] = embeddings[fallback_idx]
                continue
            cluster_center = embeddings[mask].mean(axis=0)
            center_norm = np.linalg.norm(cluster_center)
            if center_norm <= 0:
                fallback_idx = int(np.argmax(norms[mask]))
                centers[cluster_idx] = embeddings[np.flatnonzero(mask)[fallback_idx]]
            else:
                centers[cluster_idx] = cluster_center / center_norm

    return assignments, centers


def _cluster_candidate_order(
    cluster_indices: np.ndarray,
    center: np.ndarray,
    embeddings: np.ndarray,
    norms: np.ndarray,
) -> list[int]:
    ranked = []
    for idx in cluster_indices.tolist():
        center_similarity = _cosine_similarity(embeddings[idx], center)
        score = (0.75 * float(norms[idx])) + (0.25 * center_similarity)
        ranked.append((idx, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in ranked]


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
                saved = cv2.imwrite(
                    str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
                )
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


def load_candidate_frames_from_dir(
    segment_path: Path,
    frames_dir: Path,
    image_format: str,
) -> list[FrameCandidate]:
    capture = cv2.VideoCapture(str(segment_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open segment for frame reload: {segment_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    capture.release()

    candidates = []
    for frame_path in sorted(frames_dir.glob(f"*.{image_format}")):
        match = FRAME_INDEX_PATTERN.search(frame_path.stem)
        if not match:
            continue
        frame_index = int(match.group(1))
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        height, width = image.shape[:2]
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
    return candidates


def select_top_k_frames(
    candidates: list[FrameCandidate],
    embeddings: np.ndarray,
    norms: np.ndarray,
    top_k: int,
    dedupe_threshold: float,
    kmeans_iterations: int = 12,
) -> list[SelectedFrame]:
    if not candidates:
        return []
    if top_k <= 0:
        return []

    cluster_count = min(top_k, len(candidates))
    assignments, centers = _run_kmeans(
        embeddings=embeddings,
        norms=norms,
        cluster_count=cluster_count,
        iterations=kmeans_iterations,
    )

    selected_indices: list[int] = []
    selected_embeddings: list[np.ndarray] = []

    for cluster_idx in range(cluster_count):
        cluster_indices = np.flatnonzero(assignments == cluster_idx)
        if cluster_indices.size == 0:
            continue
        for candidate_idx in _cluster_candidate_order(
            cluster_indices, centers[cluster_idx], embeddings, norms
        ):
            candidate_embedding = embeddings[candidate_idx]
            if selected_embeddings:
                sims = [
                    _cosine_similarity(candidate_embedding, other) for other in selected_embeddings
                ]
                if max(sims) >= dedupe_threshold:
                    continue
            selected_indices.append(candidate_idx)
            selected_embeddings.append(candidate_embedding)
            break

    if len(selected_indices) < cluster_count:
        order = np.argsort(-norms)
        for idx in order.tolist():
            if idx in selected_indices:
                continue
            candidate_embedding = embeddings[idx]
            if selected_embeddings:
                sims = [
                    _cosine_similarity(candidate_embedding, other) for other in selected_embeddings
                ]
                if max(sims) >= dedupe_threshold:
                    continue
            selected_indices.append(idx)
            selected_embeddings.append(candidate_embedding)
            if len(selected_indices) >= cluster_count:
                break

    selected_indices.sort(key=lambda idx: (-float(norms[idx]), candidates[idx].timestamp_seconds))

    selected: list[SelectedFrame] = []
    for idx in selected_indices[:cluster_count]:
        item = candidates[idx]
        selected.append(
            SelectedFrame(
                frame_path=item.frame_path,
                timestamp_seconds=item.timestamp_seconds,
                frame_index=item.frame_index,
                width=item.width,
                height=item.height,
                embedding_norm=float(norms[idx]),
                embedding=embeddings[idx],
            )
        )

    return selected
