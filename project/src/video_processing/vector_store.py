from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, frame_ids: list[str], embeddings: np.ndarray):
        ...

    @abstractmethod
    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        ...

    @abstractmethod
    def save(self):
        ...
