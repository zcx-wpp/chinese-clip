from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

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


def expand_query(query: str, templates: tuple[str, ...]) -> list[str]:
    outputs = []
    seen = set()
    for template in templates:
        text = template.format(query=query).strip()
        if text and text not in seen:
            outputs.append(text)
            seen.add(text)
    return outputs or [query]


def aggregate_hits(hits: list[SearchHit], merge_gap_seconds: float, max_segments_per_video: int) -> list[dict]:
    grouped = defaultdict(list)
    for hit in hits:
        grouped[hit.video_id].append(hit)

    results = []
    for video_id, video_hits in grouped.items():
        video_hits.sort(key=lambda item: item.timestamp)
        segments = []
        current = None
        for hit in video_hits:
            if current is None or hit.timestamp - current["end"] > merge_gap_seconds:
                current = {
                    "start": hit.timestamp,
                    "end": hit.timestamp,
                    "best_score": hit.score,
                    "frames": [hit.frame_id],
                }
                segments.append(current)
            else:
                current["end"] = hit.timestamp
                current["best_score"] = max(current["best_score"], hit.score)
                current["frames"].append(hit.frame_id)

        segments.sort(key=lambda item: item["best_score"], reverse=True)
        kept_segments = [{"start": round(item["start"], 3), "end": round(item["end"], 3)} for item in segments[:max_segments_per_video]]
        score = max(hit.score for hit in video_hits)
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
        metadata_store: MetadataStore,
        query_templates: tuple[str, ...],
        retrieval_config: RetrievalConfig,
    ):
        self.encoder = encoder
        self.index = index
        self.metadata_store = metadata_store
        self.query_templates = query_templates
        self.retrieval_config = retrieval_config

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        expanded_queries = expand_query(query, self.query_templates)
        query_embeddings = self.encoder.encode_texts(expanded_queries)
        raw_results = self.index.search(query_embeddings, self.retrieval_config.recall_top_k)

        score_by_frame = {}
        for result_list in raw_results:
            for frame_id, score in result_list:
                score_by_frame[frame_id] = max(score_by_frame.get(frame_id, float("-inf")), score)

        top_frames = sorted(score_by_frame.items(), key=lambda item: item[1], reverse=True)
        records = self.metadata_store.get_frame_records([frame_id for frame_id, _ in top_frames[: self.retrieval_config.recall_top_k]])

        hits = []
        for item in records:
            hits.append(
                SearchHit(
                    frame_id=item["frame_id"],
                    video_id=item["video_id"],
                    timestamp=float(item["timestamp"]),
                    score=float(score_by_frame[item["frame_id"]]),
                    segment_id=item["segment_id"],
                    frame_path=item["frame_path"],
                    video_path=item["video_path"],
                )
            )

        aggregated = aggregate_hits(
            hits=hits,
            merge_gap_seconds=self.retrieval_config.merge_gap_seconds,
            max_segments_per_video=self.retrieval_config.max_segments_per_video,
        )
        return aggregated[: (top_k or self.retrieval_config.result_videos)]
