from __future__ import annotations

import numpy as np


def softmax_attention_pooling(frame_embeddings: np.ndarray, frame_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame_embeddings.size == 0:
        raise ValueError("frame_embeddings must not be empty")
    if frame_scores.size == 0:
        raise ValueError("frame_scores must not be empty")
    scores = frame_scores.astype(np.float32)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.clip(np.sum(exp_scores), a_min=1e-12, a_max=None)
    pooled = np.sum(frame_embeddings * weights[:, None], axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32), weights.astype(np.float32)
