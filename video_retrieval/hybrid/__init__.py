"""Doubao caption + sparse/dense hybrid video retrieval."""

from __future__ import annotations

from .hybrid_retrieval import HybridSearchConfig, HybridSearchEngine, build_search_engine

__all__ = [
    "HybridSearchConfig",
    "HybridSearchEngine",
    "build_caption_config",
    "build_index_main",
    "build_search_engine",
    "caption_main",
    "run_batch",
]


def __getattr__(name: str):
    if name == "build_caption_config":
        from .doubao_batch_caption import build_config as build_caption_config

        return build_caption_config
    if name in ("caption_main", "run_batch"):
        from . import doubao_batch_caption

        return getattr(doubao_batch_caption, name)
    if name == "build_index_main":
        from .build_hybrid_index import main as build_index_main

        return build_index_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
