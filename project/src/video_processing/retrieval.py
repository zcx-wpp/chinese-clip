from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RetrievalConfig
from .embedding import ChineseClipEncoder
from .faiss_store import FaissFrameIndex
from .metadata_store import MetadataStore


@dataclass
class SearchHit:
    frame_id: str
    video_id: str
    timestamp: float
    score: float
    segment_id: str
    frame_path: str
    video_path: str
    embedding_path: str


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _average_top_k(scores: list[float], top_k: int) -> float:
    ranked = sorted(scores, reverse=True)
    kept = ranked[: max(1, top_k)]
    return float(sum(kept) / len(kept))


def _smoothmax_top_k(scores: list[float], top_k: int, beta: float) -> float:
    ranked = sorted(scores, reverse=True)
    kept = ranked[: max(1, top_k)]
    if not kept:
        return 0.0
    if len(kept) == 1 or beta <= 0:
        return float(kept[0])
    values = np.asarray(kept, dtype=np.float32)
    anchor = float(np.max(values))
    stabilized = np.exp((values - anchor) * float(beta))
    return float(anchor + (np.log(np.sum(stabilized)) - np.log(len(values))) / float(beta))


def _consensus_smoothmax_top_k(
    scores: list[float],
    top_k: int,
    beta: float,
    support_floor: float,
    support_bonus_weight: float,
    spike_penalty_weight: float,
) -> float:
    ranked = sorted(scores, reverse=True)
    kept = ranked[: max(1, top_k)]
    if not kept:
        return 0.0
    if len(kept) == 1:
        return float(kept[0])

    avg_score = _average_top_k(kept, len(kept))
    smooth_score = _smoothmax_top_k(kept, len(kept), beta)
    tail_score = float(kept[-1])
    peak_score = float(kept[0])
    support_ratio = sum(score >= support_floor for score in kept) / len(kept)

    consensus_core = (0.65 * smooth_score) + (0.35 * avg_score)
    support_bonus = support_bonus_weight * support_ratio * tail_score
    spike_penalty = spike_penalty_weight * max(0.0, peak_score - avg_score) * (1.0 - support_ratio)
    return float(consensus_core + support_bonus - spike_penalty)


def _average_adjacent_similarity(similarities: list[float]) -> float:
    if not similarities:
        return 0.0
    return float(sum(similarities) / len(similarities))


def aggregate_hits(
    hits: list[SearchHit],
    video_scores: dict[str, float],
    clip_score_mode: str,
    clip_score_top_k: int,
    merge_gap_seconds: float,
    temporal_consistency_threshold: float,
    clip_avg_score_weight: float,
    clip_temporal_consistency_weight: float,
    clip_smoothmax_beta: float,
    max_segments_per_video: int,
    embedding_loader,
) -> list[dict]:
    grouped = defaultdict(list)
    for hit in hits:
        grouped[hit.video_id].append(hit)

    results = []
    for video_id, video_hits in grouped.items():
        video_hits.sort(key=lambda item: item.timestamp)
        segments = []
        current = None
        previous_embedding = None

        for hit in video_hits:
            current_embedding = embedding_loader(hit.embedding_path)
            can_merge = False
            if current is not None:
                time_gap = hit.timestamp - current["end"]
                semantic_similarity = float(np.dot(previous_embedding, current_embedding)) if previous_embedding is not None else -1.0
                can_merge = (
                    time_gap <= merge_gap_seconds
                    and semantic_similarity >= temporal_consistency_threshold
                )

            if current is None or not can_merge:
                current = {
                    "start": hit.timestamp,
                    "end": hit.timestamp,
                    "frame_scores": [hit.score],
                    "adjacent_similarities": [],
                    "frames": [hit.frame_id],
                }
                segments.append(current)
            else:
                current["end"] = hit.timestamp
                current["frame_scores"].append(hit.score)
                current["adjacent_similarities"].append(float(np.dot(previous_embedding, current_embedding)))
                current["frames"].append(hit.frame_id)

            previous_embedding = current_embedding

        for item in segments:
            avg_score = _average_top_k(item["frame_scores"], clip_score_top_k)
            smooth_score = _smoothmax_top_k(item["frame_scores"], clip_score_top_k, clip_smoothmax_beta)
            temporal_consistency = _average_adjacent_similarity(item["adjacent_similarities"])
            if clip_score_mode == "max":
                clip_score = float(max(item["frame_scores"]))
            elif clip_score_mode == "topk_average":
                clip_score = avg_score
            elif clip_score_mode == "smoothmax":
                clip_score = smooth_score
            elif clip_score_mode == "temporal_smoothmax":
                clip_score = (
                    clip_avg_score_weight * smooth_score
                    + clip_temporal_consistency_weight * temporal_consistency
                )
            else:
                clip_score = (
                    clip_avg_score_weight * avg_score
                    + clip_temporal_consistency_weight * temporal_consistency
                )
            item["avg_score"] = avg_score
            item["smooth_score"] = smooth_score
            item["temporal_consistency"] = temporal_consistency
            item["clip_score"] = float(clip_score)

        segments.sort(key=lambda item: item["clip_score"], reverse=True)
        kept_segments = []
        for item in segments[:max_segments_per_video]:
            kept_segments.append(
                {
                    "start": round(item["start"], 3),
                    "end": round(item["end"], 3),
                    "score": round(item["clip_score"], 4),
                }
            )

        score = video_scores.get(video_id, max(hit.score for hit in video_hits))
        results.append(
            {
                "video_id": video_id,
                "score": round(score, 4),
                "segments": kept_segments,
                "video_path": video_hits[0].video_path,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


class VideoRetriever:
    def __init__(
        self,
        encoder: ChineseClipEncoder,
        index: FaissFrameIndex,
        segment_index: FaissFrameIndex,
        video_index: FaissFrameIndex | None,
        metadata_store: MetadataStore,
        retrieval_config: RetrievalConfig,
    ):
        self.encoder = encoder
        self.index = index
        self.segment_index = segment_index
        self.video_index = video_index
        self.metadata_store = metadata_store
        self.retrieval_config = retrieval_config
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.video_item_genericness_cache: dict[str, float] | None = None

    def _load_video_item_genericness(self) -> dict[str, float]:
        if self.video_item_genericness_cache is not None:
            return self.video_item_genericness_cache
        if self.video_index is None or not getattr(self.video_index, "item_ids", None):
            self.video_item_genericness_cache = {}
            return self.video_item_genericness_cache

        item_ids = list(self.video_index.item_ids)
        segment_ids = [item_id.split("::", 1)[1] for item_id in item_ids if "::" in item_id]
        records = self.metadata_store.get_segment_records(segment_ids)
        genericness_by_segment_id = {
            item["segment_id"]: float(item.get("genericness_score") or 0.0)
            for item in records
        }
        self.video_item_genericness_cache = {
            item_id: genericness_by_segment_id.get(item_id.split("::", 1)[1], 0.0)
            for item_id in item_ids
            if "::" in item_id
        }
        return self.video_item_genericness_cache

    def _recall_video_ids(self, query_embeddings: np.ndarray) -> list[str]:
        if self.video_index is None or not getattr(self.video_index, "item_ids", None):
            return []
        genericness_by_item_id = self._load_video_item_genericness()
        score_by_video: dict[str, float] = {}
        raw_video_results = self.video_index.search(query_embeddings, self.retrieval_config.video_recall_top_k)
        for result_list in raw_video_results:
            for item_id, score in result_list:
                video_id = item_id.split("::", 1)[0]
                adjusted_score = float(score) - self.retrieval_config.video_genericness_penalty_weight * genericness_by_item_id.get(item_id, 0.0)
                score_by_video[video_id] = max(score_by_video.get(video_id, float("-inf")), adjusted_score)
        ranked_videos = sorted(score_by_video.items(), key=lambda item: item[1], reverse=True)
        return [video_id for video_id, _ in ranked_videos[: self.retrieval_config.video_recall_candidate_pool_size]]

    def _recall_segment_ids(
        self,
        query_embeddings: np.ndarray,
        candidate_video_ids: list[str],
    ) -> list[str]:
        score_by_segment: dict[str, float] = {}
        raw_segment_results = self.segment_index.search(query_embeddings, self.retrieval_config.segment_recall_top_k)
        for result_list in raw_segment_results:
            for segment_id, score in result_list:
                score_by_segment[segment_id] = max(score_by_segment.get(segment_id, float("-inf")), float(score))

        if not candidate_video_ids:
            filtered_segments = list(score_by_segment.items())
            filtered_segments.sort(key=lambda item: item[1], reverse=True)
            return [segment_id for segment_id, _ in filtered_segments[: self.retrieval_config.segment_recall_candidate_pool_size]]

        candidate_video_set = set(candidate_video_ids)
        candidate_segment_records = self.metadata_store.get_segment_records_by_video_ids(candidate_video_ids)
        candidate_segment_ids = {
            item["segment_id"]
            for item in candidate_segment_records
            if item["segment_id"] in score_by_segment and item["video_id"] in candidate_video_set
        }
        filtered_segments = [
            (segment_id, score_by_segment[segment_id])
            for segment_id in candidate_segment_ids
            if segment_id in candidate_segment_ids
        ]
        filtered_segments.sort(key=lambda item: item[1], reverse=True)
        if filtered_segments:
            return [segment_id for segment_id, _ in filtered_segments[: self.retrieval_config.segment_recall_candidate_pool_size]]

        fallback_segments = list(score_by_segment.items())
        fallback_segments.sort(key=lambda item: item[1], reverse=True)
        return [segment_id for segment_id, _ in fallback_segments[: self.retrieval_config.segment_recall_candidate_pool_size]]

    def load_frame_embedding(self, embedding_path: str) -> np.ndarray:
        cached = self.embedding_cache.get(embedding_path)
        if cached is not None:
            return cached
        vector = np.load(Path(embedding_path)).astype(np.float32)
        vector = normalize_vector(vector)
        self.embedding_cache[embedding_path] = vector
        return vector

    def rerank_video_scores(
        self,
        full_query_embeddings: np.ndarray,
        records: list[dict],
    ) -> tuple[dict[str, float], dict[str, float]]:
        frame_scores: dict[str, float] = {}
        video_frame_scores: dict[str, list[float]] = defaultdict(list)
        segment_clip_scores: dict[str, list[float]] = defaultdict(list)
        raw_frame_clip_scores: dict[str, float] = {}
        for item in records:
            frame_embedding = self.load_frame_embedding(item["embedding_path"])
            full_query_score = float(np.max(full_query_embeddings @ frame_embedding)) if full_query_embeddings.size else 0.0
            raw_frame_clip_scores[item["frame_id"]] = full_query_score
            segment_clip_scores[item["segment_id"]].append(full_query_score)

        segment_support_scores: dict[str, float] = {}
        for segment_id, scores in segment_clip_scores.items():
            segment_support_scores[segment_id] = _consensus_smoothmax_top_k(
                scores,
                self.retrieval_config.rerank_segment_support_top_k,
                self.retrieval_config.rerank_smoothmax_beta,
                self.retrieval_config.rerank_support_floor,
                self.retrieval_config.rerank_support_bonus_weight,
                self.retrieval_config.rerank_spike_penalty_weight,
            )

        for item in records:
            full_query_score = raw_frame_clip_scores[item["frame_id"]]
            segment_motion_score = float(item.get("motion_score") or 0.0)
            segment_support_score = float(segment_support_scores.get(item["segment_id"], 0.0))
            segment_genericness_score = float(item.get("genericness_score") or 0.0)
            final_frame_score = (
                self.retrieval_config.clip_score_weight * full_query_score
                + self.retrieval_config.motion_score_weight * segment_motion_score
                + self.retrieval_config.rerank_segment_support_weight * segment_support_score
                - self.retrieval_config.rerank_genericness_penalty_weight * segment_genericness_score
            )
            frame_scores[item["frame_id"]] = final_frame_score
            video_frame_scores[item["video_id"]].append(final_frame_score)

        video_scores: dict[str, float] = {}
        for video_id, scores in video_frame_scores.items():
            if self.retrieval_config.rerank_score_agg_mode == "topk_average":
                video_scores[video_id] = _average_top_k(
                    scores,
                    self.retrieval_config.rerank_top_k_average,
                )
            elif self.retrieval_config.rerank_score_agg_mode == "consensus_smoothmax":
                video_scores[video_id] = _consensus_smoothmax_top_k(
                    scores,
                    self.retrieval_config.rerank_top_k_average,
                    self.retrieval_config.rerank_smoothmax_beta,
                    self.retrieval_config.rerank_support_floor,
                    self.retrieval_config.rerank_support_bonus_weight,
                    self.retrieval_config.rerank_spike_penalty_weight,
                )
            else:
                video_scores[video_id] = _smoothmax_top_k(
                    scores,
                    self.retrieval_config.rerank_top_k_average,
                    self.retrieval_config.rerank_smoothmax_beta,
                )
        return video_scores, frame_scores

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        full_query_embeddings = self.encoder.encode_texts([query])
        if full_query_embeddings.size == 0:
            full_query_embeddings = self.encoder.encode_texts([query])
        candidate_video_ids = self._recall_video_ids(full_query_embeddings)
        candidate_segment_ids = self._recall_segment_ids(full_query_embeddings, candidate_video_ids)
        records = self.metadata_store.get_frame_records_by_segment_ids(candidate_segment_ids)
        video_scores, reranked_frame_scores = self.rerank_video_scores(
            full_query_embeddings=full_query_embeddings,
            records=records,
        )

        hits = []
        for item in records:
            hits.append(
                SearchHit(
                    frame_id=item["frame_id"],
                    video_id=item["video_id"],
                    timestamp=float(item["timestamp"]),
                    score=float(reranked_frame_scores[item["frame_id"]]),
                    segment_id=item["segment_id"],
                    frame_path=item["frame_path"],
                    video_path=item["video_path"],
                    embedding_path=item["embedding_path"],
                )
            )

        aggregated = aggregate_hits(
            hits=hits,
            video_scores=video_scores,
            clip_score_mode=self.retrieval_config.clip_score_mode,
            clip_score_top_k=self.retrieval_config.clip_score_top_k,
            merge_gap_seconds=self.retrieval_config.merge_gap_seconds,
            temporal_consistency_threshold=self.retrieval_config.temporal_consistency_threshold,
            clip_avg_score_weight=self.retrieval_config.clip_avg_score_weight,
            clip_temporal_consistency_weight=self.retrieval_config.clip_temporal_consistency_weight,
            clip_smoothmax_beta=self.retrieval_config.clip_smoothmax_beta,
            max_segments_per_video=self.retrieval_config.max_segments_per_video,
            embedding_loader=self.load_frame_embedding,
        )
        return aggregated[: (top_k or self.retrieval_config.result_videos)]
