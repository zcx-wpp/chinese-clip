from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from project.src.video_processing.embedding import ChineseClipEncoder

from .image_io import load_representation_frames
from .vector_utils import average_pool_vectors, l2_normalize

__all__ = ["ChineseClipEncoder", "encode_pil_images", "encode_image_path_pooled"]


def encode_pil_images(encoder: ChineseClipEncoder, images: list[Image.Image]) -> np.ndarray:
    embeddings, _ = encoder._encode([img.convert("RGB") for img in images], input_key="images")
    return l2_normalize(embeddings)


def encode_image_path_pooled(encoder: ChineseClipEncoder, image_path: Path) -> tuple[np.ndarray, float]:
    frames = load_representation_frames(image_path)
    if len(frames) == 1:
        vector = encode_pil_images(encoder, frames)[0]
        return vector, float(np.linalg.norm(vector))
    frame_vectors = encode_pil_images(encoder, frames)
    vector = average_pool_vectors(frame_vectors)
    return vector, float(np.linalg.norm(vector))
