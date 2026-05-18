from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .analyze_retrieval_gaps import load_eval_labels
from .api import build_retriever
from .config import PROJECT_ROOT
from .io_utils import read_nonempty_lines, write_json
from .profile_paths import default_metadata_db_path, default_output_dir, resolve_path
from .report_paths import timestamped_log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze where GT is lost across retrieval stages.")
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"))
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
    parser.add_argument("--eval-labels", default=str(PROJECT_ROOT / "metadata" / "1799eval_labels.json"))
    parser.add_argument("--queries-file", help="Optional UTF-8 text file, one query per line.")
    parser.add_argument("--batch-results", help="Optional batch_search_results JSON to reuse final ranking output.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Only analyze the first N labeled queries. 0 means all.")
    parser.add_argument("--export-json")
    parser.add_argument("--export-markdown")
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    return resolve_path(args.output_dir, default_output_dir(args.profile))


def _resolve_metadata_db(args) -> Path:
    return resolve_path(args.metadata_db, default_metadata_db_path(args.profile))


def _rank_of(item_id: str, ranked_ids: list[str]) -> int | None:
    for idx, value in enumerate(ranked_ids, start=1):
        if value == item_id:
            return idx
    return None


def _rank_of_any(item_ids: set[str], ranked_ids: list[str]) -> int | None:
    for idx, value in enumerate(ranked_ids, start=1):
        if value in item_ids:
            return idx
    return None


def _expected_segment_ids(retriever, expected_video_id: str, expected_segments: list[dict]) -> set[str]:
    if not expected_video_id:
        return set()
    records = retriever.metadata_store.get_segment_records_by_video_ids([expected_video_id])
    if not expected_segments:
        return {item["segment_id"] for item in records}

    matched = set()
    for record in records:
        start_time = float(record.get("start_time") or 0.0)
        end_time = float(record.get("end_time") or 0.0)
        for seg in expected_segments:
            expected_start = float(seg.get("start") or 0.0)
            expected_end = float(seg.get("end") or 0.0)
            if start_time < expected_end and end_time > expected_start:
                matched.add(record["segment_id"])
                break
    return matched


def _load_batch_results(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_map = {}
    for item in payload.get("results", []):
        query = item.get("query")
        if query:
            result_map[query] = item
    return result_map


def analyze(args) -> dict:
    retriever = build_retriever(
        output_dir=_resolve_output_dir(args),
        metadata_db_path=_resolve_metadata_db(args),
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
    eval_map = load_eval_labels(Path(args.eval_labels))
    if args.queries_file:
        requested_queries = read_nonempty_lines(Path(args.queries_file))
        queries = [query for query in requested_queries if query in eval_map]
    else:
        queries = list(eval_map.keys())
    if args.limit > 0:
        queries = queries[: args.limit]

    batch_result_map: dict[str, dict] = {}
    if args.batch_results:
        batch_result_map = _load_batch_results(Path(args.batch_results))

    summary_counter = Counter()
    results = []
    for idx, query in enumerate(queries, start=1):
        eval_item = eval_map[query]
        expected_video_id = eval_item.get("video_id")
        expected_segments = eval_item.get("segments", [])
        query_embeddings = retriever.encoder.encode_texts([query])
        candidate_video_ids = retriever._recall_video_ids(query_embeddings)
        candidate_segment_ids = retriever._recall_segment_ids(query_embeddings, candidate_video_ids)
        expected_segment_ids = _expected_segment_ids(retriever, expected_video_id, expected_segments)

        if batch_result_map:
            batch_item = batch_result_map.get(query) or {}
            final_results = list(batch_item.get("results") or [])
        else:
            final_results = retriever.search(query=query, top_k=args.top_k)

        merged_video_rank = _rank_of(expected_video_id, candidate_video_ids) if expected_video_id else None
        merged_segment_rank = _rank_of_any(expected_segment_ids, candidate_segment_ids) if expected_segment_ids else None
        final_video_rank = _rank_of(expected_video_id, [item["video_id"] for item in final_results]) if expected_video_id else None

        if merged_video_rank is None:
            terminal_stage = "video_recall_miss"
        elif merged_segment_rank is None:
            terminal_stage = "segment_recall_miss"
        elif final_video_rank is None:
            terminal_stage = "final_rank_beyond_topk"
        elif final_video_rank > 1:
            terminal_stage = "top1_rerank_loss"
        else:
            terminal_stage = "top1_hit"
        summary_counter[terminal_stage] += 1

        results.append(
            {
                "query": query,
                "expected_video_id": expected_video_id,
                "expected_segments": expected_segments,
                "merged_video_rank": merged_video_rank,
                "merged_segment_rank": merged_segment_rank,
                "final_video_rank": final_video_rank,
                "terminal_stage": terminal_stage,
                "candidate_video_count": len(candidate_video_ids),
                "candidate_segment_count": len(candidate_segment_ids),
                "top1_video_id": final_results[0]["video_id"] if final_results else None,
                "top10_video_ids": [item["video_id"] for item in final_results[: args.top_k]],
            }
        )
        print(
            f"[recall-stage] {idx}/{len(queries)} query={query} "
            f"video_rank={merged_video_rank or 'MISS'} seg_rank={merged_segment_rank or 'MISS'} "
            f"final_rank={final_video_rank or 'MISS'} stage={terminal_stage}",
            flush=True,
        )

    query_count = len(results)
    return {
        "query_count": query_count,
        "top_k": args.top_k,
        "retrieval_preset": args.retrieval_preset,
        "queries_file": args.queries_file,
        "stage_counts": dict(summary_counter),
        "stage_rates": {
            key: round(value / query_count, 4) if query_count else 0.0
            for key, value in summary_counter.items()
        },
        "results": results,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Recall Stage Analysis",
        "",
        f"- Query count: {payload['query_count']}",
        f"- Top K: {payload['top_k']}",
        f"- Retrieval preset: {payload.get('retrieval_preset', 'current')}",
        "",
        "## Stage Counts",
        "",
    ]
    for key, value in sorted(payload["stage_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value} ({payload['stage_rates'].get(key, 0.0)})")
    lines.extend(["", "## Sample Failures", ""])
    for item in payload["results"]:
        if item["terminal_stage"] == "top1_hit":
            continue
        lines.append(f"### {item['query']}")
        lines.append("")
        lines.append(f"- expected video: `{item['expected_video_id']}`")
        lines.append(f"- merged video rank: {item['merged_video_rank']}")
        lines.append(f"- merged segment rank: {item['merged_segment_rank']}")
        lines.append(f"- final video rank: {item['final_video_rank']}")
        lines.append(f"- terminal stage: `{item['terminal_stage']}`")
        lines.append(f"- top1 video: `{item['top1_video_id']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    payload = analyze(args)
    export_json = timestamped_log_path(args.profile, args.export_json, "recall_stage_analysis.json")
    export_md = timestamped_log_path(args.profile, args.export_markdown, "recall_stage_analysis.md")
    write_json(export_json, payload)
    export_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved recall stage JSON: {export_json}", flush=True)
    print(f"Saved recall stage markdown: {export_md}", flush=True)


if __name__ == "__main__":
    main()
