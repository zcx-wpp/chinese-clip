from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SegmentConfig:
    segment_seconds: int = 8
    ffmpeg_binary: str = "ffmpeg"


@dataclass
class FrameSamplingConfig:
    frames_per_second: float = 2.0
    top_k_per_segment: int = 4
    dedupe_threshold: float = 0.98
    min_side: int = 128
    image_format: str = "jpg"
    jpg_quality: int = 95


@dataclass
class EmbeddingConfig:
    model_path: str = "model"
    batch_size: int = 32
    device: str = "cuda"
    query_expansion_templates: tuple[str, ...] = (
        "{query}",
        "{query} 的视频片段",
        "画面中有 {query}",
    )


@dataclass
class MultimodalConfig:
    enable_ocr: bool = False
    enable_asr: bool = False
    ocr_lang: str = "ch"
    whisper_model: str = "base"


@dataclass
class RetrievalConfig:
    recall_top_k: int = 200
    merge_gap_seconds: float = 2.0
    max_segments_per_video: int = 3
    result_videos: int = 5


@dataclass
class VectorStoreConfig:
    backend: str = "faiss"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "video_frame_embeddings"
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "IP"
    milvus_m: int = 16
    milvus_ef_construction: int = 200


@dataclass
class PipelineConfig:
    work_dir: Path
    video_dir: Path
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    @property
    def segments_dir(self) -> Path:
        return self.work_dir / "segments"

    @property
    def frames_dir(self) -> Path:
        return self.work_dir / "frames"

    @property
    def metadata_db_path(self) -> Path:
        return self.work_dir / "metadata.db"

    @property
    def faiss_index_path(self) -> Path:
        return self.work_dir / "frame_index.faiss"

    @property
    def faiss_meta_path(self) -> Path:
        return self.work_dir / "frame_index.meta.json"
