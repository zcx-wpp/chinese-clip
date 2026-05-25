"""Top-level paths and defaults for the video_retrieval package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VIDEO_RETRIEVAL_ROOT = PACKAGE_ROOT
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT

_DEFAULT_DATA_VIDEOS = WORKSPACE_ROOT / "data" / "videos"
_LOCAL_VIDEOS = PACKAGE_ROOT / "videos"
DEFAULT_VIDEO_DIR = _LOCAL_VIDEOS if _LOCAL_VIDEOS.is_dir() else _DEFAULT_DATA_VIDEOS
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models"
DEFAULT_PROFILES_DIR = PACKAGE_ROOT / "profiles"

# Legacy per-pipeline profile trees (pre-unified layout)
LEGACY_CLIP_PROFILES_ROOT = PACKAGE_ROOT / "legacy" / "clip_profiles"
LEGACY_HYBRID_PROFILES_ROOT = PACKAGE_ROOT / "legacy" / "hybrid_profiles"
# Deprecated on-disk tree; path_compat may still rewrite old metadata paths here.
LEGACY_HYBRID_ARTIFACTS = PACKAGE_ROOT / "artifacts"
ARTIFACT_ROOT = LEGACY_HYBRID_ARTIFACTS

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8023

DEFAULT_CLIP_PROFILE = "apr_media1_project"
DEFAULT_HYBRID_PROFILE = "apr_media1"

DEFAULT_HYBRID_SPARSE_TOP_K = 100
DEFAULT_HYBRID_DENSE_TOP_K = 100
DEFAULT_HYBRID_RRF_K = 60

ARK_API_KEY_ENV = "ARK_API_KEY"
DOUBAO_MODEL_ENV = "DOUBAO_MODEL"
DOUBAO_LEGACY_MODEL_ENV = "DOUBAO_ENDPOINT_ID"
DEFAULT_BGE_MODEL = "BAAI/bge-large-zh-v1.5"


@dataclass(frozen=True)
class PipelineDefaults:
    segment_seconds: int = 4
    frames_per_second: float = 2.0
    top_k_per_segment: int = 4
    clip_batch_size: int = 16
    hybrid_workers: int = 1
    hybrid_sample_frames: int = 8
    caption_limit: int = 0
    index_limit: int = 0


PIPELINE_DEFAULTS = PipelineDefaults()
