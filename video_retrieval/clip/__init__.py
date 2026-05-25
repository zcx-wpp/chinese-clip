"""Chinese-CLIP video indexing and retrieval."""

from .retriever_factory import build_retriever
from .config import PROJECT_ROOT, RetrievalConfig
from .minimal_pipeline import build_config, run_pipeline
from .minimal_pipeline import main as run_index_main
from .retrieval import VideoRetriever

__all__ = [
    "PROJECT_ROOT",
    "RetrievalConfig",
    "VideoRetriever",
    "build_config",
    "build_retriever",
    "run_index_main",
    "run_pipeline",
]
