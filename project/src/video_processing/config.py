from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SegmentConfig:
    segment_seconds: int = 4
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
    model_path: str = str(PROJECT_ROOT / "models")
    batch_size: int = 16
    device: str = "cuda"


@dataclass
class MultimodalConfig:
    enable_ocr: bool = False
    enable_asr: bool = False
    ocr_lang: str = "ch"
    whisper_model: str = "base"


@dataclass
class RetrievalConfig:
    video_recall_top_k: int = 64
    segment_recall_top_k: int = 64
    recall_top_k: int = 100
    video_recall_candidate_pool_size: int = 128
    segment_recall_candidate_pool_size: int = 128
    rerank_top_k_average: int = 3
    rerank_segments_per_video: int = 6
    video_genericness_penalty_weight: float = 0.1
    clip_score_weight: float = 0.85
    motion_score_weight: float = 0.1
    merge_gap_seconds: float = 2.0
    temporal_consistency_threshold: float = 0.9
    clip_score_mode: str = "temporal_smoothmax"
    clip_score_top_k: int = 3
    clip_avg_score_weight: float = 0.7
    clip_temporal_consistency_weight: float = 0.3
    clip_smoothmax_beta: float = 12.0
    rerank_score_agg_mode: str = "consensus_smoothmax"
    rerank_smoothmax_beta: float = 10.0
    rerank_support_floor: float = 0.18
    rerank_support_bonus_weight: float = 0.08
    rerank_spike_penalty_weight: float = 0.06
    rerank_segment_support_weight: float = 0.2
    rerank_genericness_penalty_weight: float = 0.05
    rerank_segment_support_top_k: int = 3
    max_segments_per_video: int = 3
    result_videos: int = 5

    @classmethod
    def for_preset(cls, preset: str = "current") -> "RetrievalConfig":
        config = cls()
        if preset == "baseline":
            config.video_recall_top_k = 24
            config.video_recall_candidate_pool_size = 48
        elif preset != "current":
            raise ValueError(f"Unsupported retrieval preset: {preset}")
        return config


@dataclass
class VideoRepresentationConfig:
    representative_segments_top_n: int = 8
    importance_embedding_norm_weight: float = 1.0
    importance_motion_score_weight: float = 1.0
    importance_visual_diversity_weight: float = 1.0
    genericness_weight: float = 0.35
    diversity_penalty_weight: float = 0.25


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
    project_root: Path = PROJECT_ROOT
    video_dir: Path = PROJECT_ROOT / "videos"
    output_dir: Path = PROJECT_ROOT / "output"
    metadata_dir: Path = PROJECT_ROOT / "metadata"
    models_dir: Path = PROJECT_ROOT / "models"
    num_workers: int = 2
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    video_representation: VideoRepresentationConfig = field(default_factory=VideoRepresentationConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    def _output_path(self, *parts: str) -> Path:
        return self.output_dir.joinpath(*parts)

    def _faiss_path(self, name: str) -> Path:
        return self._output_path("faiss", name)

    @property
    def segments_dir(self) -> Path:
        return self._output_path("segments")

    @property
    def frames_dir(self) -> Path:
        return self._output_path("frames")

    @property
    def embeddings_dir(self) -> Path:
        return self._output_path("embeddings")

    @property
    def faiss_dir(self) -> Path:
        return self._output_path("faiss")

    @property
    def logs_dir(self) -> Path:
        return self._output_path("logs")

    @property
    def metadata_db_path(self) -> Path:
        return self.metadata_dir / "metadata.db"

    @property
    def faiss_index_path(self) -> Path:
        return self._faiss_path("frame_index.faiss")

    @property
    def faiss_meta_path(self) -> Path:
        return self._faiss_path("frame_index.meta.json")

    @property
    def segment_index_path(self) -> Path:
        return self._faiss_path("segment_index.faiss")

    @property
    def segment_meta_path(self) -> Path:
        return self._faiss_path("segment_index.meta.json")

    @property
    def video_index_path(self) -> Path:
        return self._faiss_path("video_index.faiss")

    @property
    def video_meta_path(self) -> Path:
        return self._faiss_path("video_index.meta.json")
