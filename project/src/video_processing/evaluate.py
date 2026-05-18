from __future__ import annotations

import argparse
from pathlib import Path

from .api import build_retriever
from .config import PROJECT_ROOT
from .io_utils import read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video retrieval quality.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--metadata-db", default=str(PROJECT_ROOT / "metadata" / "metadata.db"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--labels", required=True, help="JSON file with query/video/time annotations.")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N labels. 0 means all.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-preset", choices=["current", "baseline"], default="current")
    parser.add_argument("--video-recall-top-k", type=int, help="Optional override for video recall search count.")
    parser.add_argument("--segment-recall-top-k", type=int, help="Optional override for segment recall search count.")
    parser.add_argument(
        "--video-recall-candidate-pool-size",
        type=int,
        help="Optional override for video recall candidate pool size.",
    )
    parser.add_argument(
        "--segment-recall-candidate-pool-size",
        type=int,
        help="Optional override for segment recall candidate pool size.",
    )
    parser.add_argument(
        "--rerank-score-agg-mode",
        choices=["topk_average", "smoothmax", "consensus_smoothmax"],
        help="Optional override for final video score aggregation mode.",
    )
    parser.add_argument("--rerank-top-k-average", type=int, help="Optional override for rerank top-k aggregation count.")
    parser.add_argument("--rerank-smoothmax-beta", type=float, help="Optional override for rerank smoothmax beta.")
    parser.add_argument("--clip-score-weight", type=float, help="Optional override for clip score weight.")
    parser.add_argument("--motion-score-weight", type=float, help="Optional override for motion score weight.")
    parser.add_argument(
        "--rerank-segment-support-weight",
        type=float,
        help="Optional override for segment support weight.",
    )
    parser.add_argument(
        "--rerank-genericness-penalty-weight",
        type=float,
        help="Optional override for genericness penalty weight.",
    )
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_labels(labels_path: Path) -> list[dict]:
    payload = read_json(labels_path)
    if not isinstance(payload, list):
        raise ValueError("Labels file must be a JSON array.")
    return payload


def overlaps(pred_segment: dict, gt_segment: dict) -> bool:
    return not (pred_segment["end"] < gt_segment["start"] or pred_segment["start"] > gt_segment["end"])


def segment_distance(pred_segment: dict, gt_segment: dict) -> float:
    pred_center = (pred_segment["start"] + pred_segment["end"]) / 2.0
    gt_center = (gt_segment["start"] + gt_segment["end"]) / 2.0
    return abs(pred_center - gt_center)


def evaluate_case(results: list[dict], label: dict, top_k: int) -> dict:
    target_video_id = label["video_id"]
    gt_segments = label.get("segments", [])

    video_rank = None
    time_hit = False
    best_time_distance = None

    for rank, item in enumerate(results[:top_k], start=1):
        if item["video_id"] != target_video_id:
            continue
        if video_rank is None:
            video_rank = rank
        if not gt_segments:
            time_hit = True
            best_time_distance = 0.0
            break
        for pred_segment in item.get("segments", []):
            distances = [segment_distance(pred_segment, gt_segment) for gt_segment in gt_segments]
            candidate_distance = min(distances) if distances else None
            if candidate_distance is not None:
                if best_time_distance is None or candidate_distance < best_time_distance:
                    best_time_distance = candidate_distance
            if any(overlaps(pred_segment, gt_segment) for gt_segment in gt_segments):
                time_hit = True
        break

    return {
        "query": label["query"],
        "target_video_id": target_video_id,
        "video_rank": video_rank,
        "video_hit": video_rank is not None and video_rank <= top_k,
        "top1_hit": video_rank == 1,
        "time_hit": time_hit,
        "best_time_distance": best_time_distance,
    }


def summarize(cases: list[dict], top_k: int) -> dict:
    total = len(cases)
    recall_at_k = sum(1 for item in cases if item["video_hit"]) / total if total else 0.0
    top1 = sum(1 for item in cases if item["top1_hit"]) / total if total else 0.0
    time_recall = sum(1 for item in cases if item["time_hit"]) / total if total else 0.0
    mrr = sum(1.0 / item["video_rank"] if item["video_rank"] else 0.0 for item in cases) / total if total else 0.0

    distances = [item["best_time_distance"] for item in cases if item["best_time_distance"] is not None]
    avg_time_distance = sum(distances) / len(distances) if distances else None

    return {
        f"recall@{top_k}": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "top1_accuracy": round(top1, 4),
        "time_hit_rate": round(time_recall, 4),
        "avg_time_center_distance": round(avg_time_distance, 4) if avg_time_distance is not None else None,
        "count": total,
    }


def main():
    args = parse_args()
    labels = load_labels(Path(args.labels))
    if args.limit > 0:
        labels = labels[:args.limit]
    retriever = build_retriever(
        output_dir=Path(args.output_dir),
        metadata_db_path=Path(args.metadata_db),
        model_path=args.model_path,
        device=args.device,
        retrieval_preset=args.retrieval_preset,
        video_recall_top_k=args.video_recall_top_k,
        segment_recall_top_k=args.segment_recall_top_k,
        video_recall_candidate_pool_size=args.video_recall_candidate_pool_size,
        segment_recall_candidate_pool_size=args.segment_recall_candidate_pool_size,
        rerank_score_agg_mode=args.rerank_score_agg_mode,
        rerank_top_k_average=args.rerank_top_k_average,
        rerank_smoothmax_beta=args.rerank_smoothmax_beta,
        clip_score_weight=args.clip_score_weight,
        motion_score_weight=args.motion_score_weight,
        rerank_segment_support_weight=args.rerank_segment_support_weight,
        rerank_genericness_penalty_weight=args.rerank_genericness_penalty_weight,
        vector_backend=args.vector_backend,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        milvus_collection=args.milvus_collection,
    )

    cases = []
    for label in labels:
        results = retriever.search(query=label["query"], top_k=args.top_k)
        case = evaluate_case(results=results, label=label, top_k=args.top_k)
        cases.append(case)
        print(
            f"[eval] query={label['query']} "
            f"rank={case['video_rank']} "
            f"video_hit={case['video_hit']} "
            f"time_hit={case['time_hit']}"
        )

    summary = summarize(cases, top_k=args.top_k)
    print("")
    print("Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
