from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent
ARTIFACT_ROOT = PIPELINE_ROOT / "artifacts"
DEFAULT_VIDEO_DIR = WORKSPACE_ROOT / "data" / "videos"
DEFAULT_CAPTIONS_JSONL = ARTIFACT_ROOT / "doubao_video_captions.jsonl"
DEFAULT_METADATA_DB = ARTIFACT_ROOT / "metadata.db"
DEFAULT_INDEX_DIR = ARTIFACT_ROOT / "hybrid_index"
