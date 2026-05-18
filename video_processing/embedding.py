from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


def _ensure_feature_tensor(features) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(features)!r}")


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class EncodedBatch:
    embeddings: np.ndarray
    norms: np.ndarray


class ChineseClipEncoder:
    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 32):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = ChineseCLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        projection_dim = getattr(self.model.config, "projection_dim", None)
        if projection_dim is not None:
            return int(projection_dim)
        visual_projection = getattr(self.model, "visual_projection", None)
        out_features = getattr(visual_projection, "out_features", None)
        if out_features is not None:
            return int(out_features)
        return 512

    def encode_images(self, image_paths: Iterable[str]) -> EncodedBatch:
        paths = list(image_paths)
        all_embeddings = []
        all_norms = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = _ensure_feature_tensor(features)
                norms = features.norm(dim=-1)
                features = _normalize(features)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            all_norms.append(norms.cpu().numpy().astype(np.float32))
        return EncodedBatch(
            embeddings=np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 512), dtype=np.float32),
            norms=np.concatenate(all_norms, axis=0) if all_norms else np.zeros((0,), dtype=np.float32),
        )

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        items = [text.strip() for text in texts if text and text.strip()]
        if not items:
            return np.zeros((0, 512), dtype=np.float32)
        outputs = []
        for start in range(0, len(items), self.batch_size):
            batch_texts = items[start:start + self.batch_size]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = _ensure_feature_tensor(features)
                features = _normalize(features)
            outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)
