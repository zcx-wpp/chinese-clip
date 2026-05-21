from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .hybrid_retrieval import HybridSearchConfig, HybridSearchEngine
from .io_utils import write_json
from .profile_paths import (
    default_index_dir,
    default_metadata_db_path,
    default_output_dir,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Doubao hybrid retrieval with query/label files.")
    parser.add_argument("--profile", help="Optional profile name for metadata.db and hybrid index.")
    parser.add_argument("--metadata-db", help="Optional metadata.db path.")
    parser.add_argument("--index-dir", help="Optional hybrid index directory.")
    parser.add_argument("--labels", required=True, help="JSON file containing query-to-video labels.")
    parser.add_argument("--queries-txt", help="Optional UTF-8 text file, one query per line.")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N labels. 0 means all.")
    parser.add_argument("--top-k-max", type=int, default=10, help="Compute Top1..TopK metrics up to this value.")
    parser.add_argument("--sparse-top-k", type=int, default=100)
    parser.add_argument("--dense-top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        help="Embedding device. Defaults to cuda and falls back to cpu when CUDA is unavailable.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        help="Only load the dense embedding model from local files; do not download from Hugging Face.",
    )
    parser.add_argument(
        "--export",
        help="Optional path to save the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N queries. Defaults to 100.",
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels(path: Path) -> list[dict]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Labels file must be a JSON array.")
    rows: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Label at index {index} is not a JSON object.")
        query = str(item.get("query") or "").strip()
        video_id = str(item.get("video_id") or item.get("videoID") or "").strip()
        if not query:
            raise ValueError(f"Label at index {index} is missing query.")
        if not video_id:
            raise ValueError(f"Label at index {index} is missing video_id.")
        rows.append(item)
    return rows


def resolve_queries(labels: list[dict], queries_txt: Path | None) -> tuple[list[str], list[dict]]:
    if queries_txt is None:
        return [str(item["query"]).strip() for item in labels], []

    queries = read_nonempty_lines(queries_txt)
    if len(queries) < len(labels):
        raise ValueError(
            f"Query count is smaller than label count: queries={len(queries)} labels={len(labels)} "
            f"queries_txt={queries_txt}"
        )
    if len(queries) > len(labels):
        queries = queries[: len(labels)]

    mismatches: list[dict] = []
    for index, (query_text, label) in enumerate(zip(queries, labels, strict=True)):
        label_query = str(label.get("query") or "").strip()
        if query_text != label_query:
            mismatches.append(
                {
                    "query_index": index,
                    "queries_txt": query_text,
                    "labels_json": label_query,
                    "video_id": label.get("video_id"),
                }
            )
    return queries, mismatches


def evaluate_case(results: list[dict], *, target_video_id: str, top_k_max: int) -> tuple[int | None, dict[str, bool]]:
    rank: int | None = None
    for idx, item in enumerate(results[:top_k_max], start=1):
        if str(item.get("video_id") or "").strip() == target_video_id:
            rank = idx
            break

    hits = {f"top{value}": rank is not None and rank <= value for value in range(1, top_k_max + 1)}
    return rank, hits


def build_export_path(args) -> Path:
    if args.export:
        return Path(args.export)
    return default_output_dir(args.profile) / f"retrieval_eval_top{args.top_k_max}.json"


def main():
    args = parse_args()
    if args.top_k_max <= 0:
        raise ValueError("--top-k-max must be positive")

    labels = load_labels(Path(args.labels))
    if args.limit > 0:
        labels = labels[: args.limit]

    queries, mismatches = resolve_queries(labels, Path(args.queries_txt) if args.queries_txt else None)
    search_config = HybridSearchConfig(
        sparse_top_k=max(args.top_k_max, args.sparse_top_k),
        dense_top_k=max(args.top_k_max, args.dense_top_k),
        rrf_k=args.rrf_k,
    )
    metadata_db_path = resolve_path(args.metadata_db, default_metadata_db_path(args.profile))
    index_dir = resolve_path(args.index_dir, default_index_dir(args.profile))

    started_at = perf_counter()
    engine = HybridSearchEngine(
        metadata_db_path=metadata_db_path,
        index_dir=index_dir,
        search_config=search_config,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_local_files_only=args.embedding_local_files_only,
    )

    cases: list[dict] = []
    hit_counts = {f"top{value}": 0 for value in range(1, args.top_k_max + 1)}
    reciprocal_rank_sum = 0.0
    total_search_seconds = 0.0
    progress_every = max(1, args.progress_every)

    try:
        total = len(labels)
        for index, (label, query_text) in enumerate(zip(labels, queries, strict=True), start=1):
            query_started_at = perf_counter()
            results = engine.search(query_text, top_k=args.top_k_max)
            elapsed_seconds = perf_counter() - query_started_at
            total_search_seconds += elapsed_seconds

            target_video_id = str(label["video_id"]).strip()
            rank, hits = evaluate_case(results, target_video_id=target_video_id, top_k_max=args.top_k_max)
            for key, matched in hits.items():
                if matched:
                    hit_counts[key] += 1
            if rank is not None:
                reciprocal_rank_sum += 1.0 / rank

            top_result_ids = [str(item.get("video_id") or "").strip() for item in results]
            case = {
                "query_index": label.get("query_index", index - 1),
                "query": query_text,
                "label_query": str(label.get("query") or "").strip(),
                "target_video_id": target_video_id,
                "rank": rank,
                "search_seconds": round(elapsed_seconds, 4),
                "top_result_video_ids": top_result_ids,
                "hits": hits,
            }
            cases.append(case)

            if index <= 5 or index == total or index % progress_every == 0:
                print(
                    f"[eval] {index}/{total} rank={rank} "
                    f"top1={hits['top1']} top10={hits.get(f'top{args.top_k_max}', False)} "
                    f"query={query_text}",
                    flush=True,
                )
    finally:
        engine.close()

    total = len(cases)
    accuracy_at = {
        key: round((value / total) if total else 0.0, 4)
        for key, value in hit_counts.items()
    }
    summary = {
        "count": total,
        "profile": args.profile,
        "metadata_db": str(metadata_db_path),
        "index_dir": str(index_dir),
        "top_k_max": args.top_k_max,
        "accuracy_at": accuracy_at,
        "hit_counts": hit_counts,
        "mrr": round((reciprocal_rank_sum / total) if total else 0.0, 4),
        "avg_search_seconds": round((total_search_seconds / total) if total else 0.0, 4),
        "total_search_seconds": round(total_search_seconds, 4),
        "total_elapsed_seconds": round(perf_counter() - started_at, 4),
        "query_label_mismatch_count": len(mismatches),
    }
    payload = {
        "summary": summary,
        "query_label_mismatches": mismatches,
        "cases": cases,
    }

    export_path = build_export_path(args)
    write_json(export_path, payload)

    print("")
    print("Summary")
    for key, value in summary.items():
        if key == "accuracy_at":
            continue
        print(f"{key}: {value}")
    for key, value in accuracy_at.items():
        print(f"{key}_accuracy: {value}")
    print(f"export: {export_path}")


if __name__ == "__main__":
    main()
