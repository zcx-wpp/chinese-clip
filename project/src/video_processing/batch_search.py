from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from .api import build_retriever
from .config import PROJECT_ROOT
from .io_utils import read_nonempty_lines, write_json
from .profile_paths import (
    default_metadata_db_path,
    default_output_dir,
    resolve_path,
)
from .report_paths import timestamped_log_path


_PROCESS_RETRIEVER = None
_PROCESS_RETRIEVER_KEY = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch text-to-video retrieval and export TopK results.")
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-workers", type=int, default=1, help="Number of queries to search concurrently.")
    parser.add_argument("--queries-file", help="UTF-8 text file, one query per line.")
    parser.add_argument("--query", action="append", default=[], help="Single query. Can be passed multiple times.")
    parser.add_argument(
        "--export",
        help="Path to save exported retrieval results.",
    )
    return parser.parse_args()


def resolve_export_path(args) -> Path:
    return timestamped_log_path(args.profile, args.export, "batch_search_results.json")


def resolve_output_dir(args) -> Path:
    return resolve_path(args.output_dir, default_output_dir(args.profile))


def resolve_metadata_db(args) -> Path:
    return resolve_path(args.metadata_db, default_metadata_db_path(args.profile))


def load_queries(queries_file: str | None, inline_queries: list[str]) -> list[str]:
    queries = (read_nonempty_lines(Path(queries_file)) if queries_file else []) + [
        item.strip() for item in inline_queries if item.strip()
    ]
    queries = list(dict.fromkeys(queries))
    if not queries:
        raise ValueError("No queries provided. Use --queries-file or --query.")
    return queries


def _build_batch_retriever(args):
    return build_retriever(
        output_dir=resolve_output_dir(args),
        metadata_db_path=resolve_metadata_db(args),
        model_path=args.model_path,
        device=args.device,
        video_recall_top_k=args.video_recall_top_k,
        segment_recall_top_k=args.segment_recall_top_k,
        video_recall_candidate_pool_size=args.video_recall_candidate_pool_size,
        segment_recall_candidate_pool_size=args.segment_recall_candidate_pool_size,
        rerank_top_k_average=args.rerank_top_k_average,
        rerank_smoothmax_beta=args.rerank_smoothmax_beta,
        clip_score_weight=args.clip_score_weight,
        motion_score_weight=args.motion_score_weight,
        rerank_segment_support_weight=args.rerank_segment_support_weight,
        rerank_genericness_penalty_weight=args.rerank_genericness_penalty_weight,
    )


def _retriever_cache_key(args) -> tuple[str, ...]:
    return (
        str(resolve_output_dir(args)),
        str(resolve_metadata_db(args)),
        args.model_path,
        args.device,
        str(args.video_recall_top_k if args.video_recall_top_k is not None else ""),
        str(args.segment_recall_top_k if args.segment_recall_top_k is not None else ""),
        str(args.video_recall_candidate_pool_size if args.video_recall_candidate_pool_size is not None else ""),
        str(args.segment_recall_candidate_pool_size if args.segment_recall_candidate_pool_size is not None else ""),
        str(args.rerank_top_k_average or ""),
        str(args.rerank_smoothmax_beta or ""),
        str(args.clip_score_weight if args.clip_score_weight is not None else ""),
        str(args.motion_score_weight if args.motion_score_weight is not None else ""),
        str(args.rerank_segment_support_weight if args.rerank_segment_support_weight is not None else ""),
        str(args.rerank_genericness_penalty_weight if args.rerank_genericness_penalty_weight is not None else ""),
    )


def _get_process_retriever(args):
    global _PROCESS_RETRIEVER
    global _PROCESS_RETRIEVER_KEY

    current_key = _retriever_cache_key(args)
    if _PROCESS_RETRIEVER is None or _PROCESS_RETRIEVER_KEY != current_key:
        _PROCESS_RETRIEVER = _build_batch_retriever(args)
        _PROCESS_RETRIEVER_KEY = current_key
    return _PROCESS_RETRIEVER


def _run_single_query(args, query: str, top_k: int) -> dict:
    retriever = _get_process_retriever(args)
    hits = retriever.search(query=query, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": hits,
    }


def _normalize_query_workers(args) -> int:
    return max(1, args.query_workers)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{remainder:.1f}s"


def _log_query_progress(done: int, total: int, query: str, top_video: str, started_at: float) -> None:
    elapsed = perf_counter() - started_at
    avg_per_query = elapsed / done if done else 0.0
    should_log_detail = done <= 5 or done == total or done % 100 == 0
    if should_log_detail:
        print(
            f"[batch-search] {done}/{total} elapsed={_format_elapsed(elapsed)} "
            f"avg={avg_per_query:.3f}s/query query={query} top1={top_video}",
            flush=True,
        )


def run_batch_search(args) -> dict:
    started_at = perf_counter()
    queries = load_queries(args.queries_file, args.query)
    queries_loaded_at = perf_counter()
    query_workers = _normalize_query_workers(args)
    use_process_pool = query_workers > 1 and not args.device.lower().startswith("cuda")
    print(
        f"[batch-search] loaded {len(queries)} unique queries in "
        f"{_format_elapsed(queries_loaded_at - started_at)}",
        flush=True,
    )
    if query_workers > 1:
        if use_process_pool:
            print(
                f"[batch-search] query_workers={query_workers} on device={args.device}; "
                "using ProcessPoolExecutor for real CPU parallelism. "
                "Each worker will load its own retriever/model once.",
                flush=True,
            )
        else:
            print(
                f"[batch-search] query_workers={query_workers} on device={args.device}. "
                "This can help throughput, but very high concurrency may contend for one GPU.",
                flush=True,
            )

    ordered_results: list[dict | None] = [None] * len(queries)
    search_started_at = perf_counter()
    if query_workers == 1:
        for idx, query in enumerate(queries, start=1):
            result = _run_single_query(args, query, args.top_k)
            ordered_results[idx - 1] = result
            hits = result["results"]
            top_video = hits[0]["video_id"] if hits else "NONE"
            _log_query_progress(idx, len(queries), query, top_video, search_started_at)
    else:
        executor_cls = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
        with executor_cls(max_workers=query_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single_query, args, query, args.top_k): idx
                for idx, query in enumerate(queries)
            }
            completed = 0
            total = len(queries)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                ordered_results[idx] = result
                completed += 1
                hits = result["results"]
                top_video = hits[0]["video_id"] if hits else "NONE"
                _log_query_progress(completed, total, result["query"], top_video, search_started_at)

    results = [result for result in ordered_results if result is not None]
    total_elapsed = perf_counter() - started_at
    search_elapsed = perf_counter() - search_started_at
    avg_per_query = search_elapsed / len(queries) if queries else 0.0
    print(
        f"[batch-search] completed {len(queries)} queries in {_format_elapsed(total_elapsed)} "
        f"(search={_format_elapsed(search_elapsed)}, avg={avg_per_query:.3f}s/query)",
        flush=True,
    )

    return {
        "query_count": len(queries),
        "top_k": args.top_k,
        "queries": queries,
        "results": results,
    }


def main():
    args = parse_args()
    payload = run_batch_search(args)
    export_path = resolve_export_path(args)
    write_json(export_path, payload)
    print(f"Saved batch search results: {export_path}", flush=True)


if __name__ == "__main__":
    main()
