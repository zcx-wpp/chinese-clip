from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .io_utils import read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two retrieval gap reports.")
    parser.add_argument("--baseline", required=True, help="Older retrieval_gap_report_*.json")
    parser.add_argument("--candidate", required=True, help="Newer retrieval_gap_report_*.json")
    parser.add_argument(
        "--export-markdown",
        default=str(PROJECT_ROOT / "output" / "logs" / "retrieval_report_compare.md"),
    )
    parser.add_argument(
        "--export-json",
        default=str(PROJECT_ROOT / "output" / "logs" / "retrieval_report_compare.json"),
    )
    parser.add_argument("--top-n", type=int, default=50, help="Number of changed queries to keep in the detailed lists.")
    return parser.parse_args()


def _expected_rank(result_item: dict) -> int | None:
    for item in result_item.get("results", []):
        if item.get("is_expected_video"):
            return int(item["rank"])
    return None


def _top1_video_id(result_item: dict) -> str | None:
    results = result_item.get("results", [])
    if not results:
        return None
    return results[0].get("video_id")


def _query_lookup(payload: dict) -> dict[str, dict]:
    return {item["query"]: item for item in payload.get("results", []) if item.get("query")}


def compare(baseline: dict, candidate: dict, top_n: int) -> dict:
    base_lookup = _query_lookup(baseline)
    cand_lookup = _query_lookup(candidate)
    queries = [query for query in base_lookup if query in cand_lookup]

    improved_queries: list[dict] = []
    worsened_queries: list[dict] = []
    changed_queries: list[dict] = []
    unchanged_queries = 0

    for query in queries:
        base_item = base_lookup[query]
        cand_item = cand_lookup[query]
        base_rank = _expected_rank(base_item)
        cand_rank = _expected_rank(cand_item)
        base_top1_hit = bool(base_item.get("top1_is_expected_video"))
        cand_top1_hit = bool(cand_item.get("top1_is_expected_video"))
        base_topk_hit = bool(base_item.get("topk_contains_expected_video"))
        cand_topk_hit = bool(cand_item.get("topk_contains_expected_video"))

        base_sort_rank = base_rank if base_rank is not None else 999999
        cand_sort_rank = cand_rank if cand_rank is not None else 999999
        rank_delta = None
        if base_rank is not None and cand_rank is not None:
            rank_delta = base_rank - cand_rank

        row = {
            "query": query,
            "expected_video_id": cand_item.get("expected_video_id") or base_item.get("expected_video_id"),
            "baseline_expected_rank": base_rank,
            "candidate_expected_rank": cand_rank,
            "baseline_top1_hit": base_top1_hit,
            "candidate_top1_hit": cand_top1_hit,
            "baseline_topk_hit": base_topk_hit,
            "candidate_topk_hit": cand_topk_hit,
            "baseline_top1_video_id": _top1_video_id(base_item),
            "candidate_top1_video_id": _top1_video_id(cand_item),
            "rank_delta": rank_delta,
        }

        if base_rank == cand_rank and base_top1_hit == cand_top1_hit and base_topk_hit == cand_topk_hit:
            unchanged_queries += 1
            continue

        changed_queries.append(row)
        if cand_sort_rank < base_sort_rank:
            improved_queries.append(row)
        elif cand_sort_rank > base_sort_rank:
            worsened_queries.append(row)
        elif (not base_top1_hit) and cand_top1_hit:
            improved_queries.append(row)
        elif base_top1_hit and (not cand_top1_hit):
            worsened_queries.append(row)

    improved_queries.sort(
        key=lambda item: (
            item["candidate_expected_rank"] if item["candidate_expected_rank"] is not None else 999999,
            -(item["baseline_expected_rank"] if item["baseline_expected_rank"] is not None else 999999),
            item["query"],
        )
    )
    worsened_queries.sort(
        key=lambda item: (
            item["baseline_expected_rank"] if item["baseline_expected_rank"] is not None else 999999,
            -(item["candidate_expected_rank"] if item["candidate_expected_rank"] is not None else 999999),
            item["query"],
        )
    )

    top_keys = sorted(
        {
            *baseline.get("cumulative_hit_rates", {}).keys(),
            *candidate.get("cumulative_hit_rates", {}).keys(),
        },
        key=lambda key: int(key.replace("top", "").replace("_hit_rate", "")),
    )
    cumulative_deltas = []
    for key in top_keys:
        cumulative_deltas.append(
            {
                "metric": key,
                "baseline": float(baseline.get("cumulative_hit_rates", {}).get(key, 0.0)),
                "candidate": float(candidate.get("cumulative_hit_rates", {}).get(key, 0.0)),
                "delta": round(
                    float(candidate.get("cumulative_hit_rates", {}).get(key, 0.0))
                    - float(baseline.get("cumulative_hit_rates", {}).get(key, 0.0)),
                    4,
                ),
            }
        )

    return {
        "baseline_query_count": baseline.get("query_count", 0),
        "candidate_query_count": candidate.get("query_count", 0),
        "baseline_top1_hit_rate": float(baseline.get("top1_hit_rate", 0.0)),
        "candidate_top1_hit_rate": float(candidate.get("top1_hit_rate", 0.0)),
        "delta_top1_hit_rate": round(float(candidate.get("top1_hit_rate", 0.0)) - float(baseline.get("top1_hit_rate", 0.0)), 4),
        "baseline_topk_hit_rate": float(baseline.get("topk_hit_rate", 0.0)),
        "candidate_topk_hit_rate": float(candidate.get("topk_hit_rate", 0.0)),
        "delta_topk_hit_rate": round(float(candidate.get("topk_hit_rate", 0.0)) - float(baseline.get("topk_hit_rate", 0.0)), 4),
        "changed_query_count": len(changed_queries),
        "improved_query_count": len(improved_queries),
        "worsened_query_count": len(worsened_queries),
        "unchanged_query_count": unchanged_queries,
        "cumulative_hit_rate_deltas": cumulative_deltas,
        "top_improved_queries": improved_queries[:top_n],
        "top_worsened_queries": worsened_queries[:top_n],
        "changed_queries_sample": changed_queries[:top_n],
    }


def _rank_text(value: int | None) -> str:
    return str(value) if value is not None else "MISS"


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Report Compare")
    lines.append("")
    lines.append(f"- Baseline query count: {payload['baseline_query_count']}")
    lines.append(f"- Candidate query count: {payload['candidate_query_count']}")
    lines.append(
        f"- Top1 hit rate: {payload['baseline_top1_hit_rate']} -> {payload['candidate_top1_hit_rate']} "
        f"(delta={payload['delta_top1_hit_rate']})"
    )
    lines.append(
        f"- TopK hit rate: {payload['baseline_topk_hit_rate']} -> {payload['candidate_topk_hit_rate']} "
        f"(delta={payload['delta_topk_hit_rate']})"
    )
    lines.append(
        f"- Changed queries: {payload['changed_query_count']} "
        f"(improved={payload['improved_query_count']}, worsened={payload['worsened_query_count']}, unchanged={payload['unchanged_query_count']})"
    )
    lines.append("")
    lines.append("## Cumulative Hit Rate Deltas")
    lines.append("")
    for item in payload["cumulative_hit_rate_deltas"]:
        lines.append(f"- {item['metric']}: {item['baseline']} -> {item['candidate']} (delta={item['delta']})")
    lines.append("")
    lines.append("## Top Improved Queries")
    lines.append("")
    for item in payload["top_improved_queries"]:
        lines.append(
            f"- query={item['query']} | expected={item['expected_video_id']} | "
            f"rank { _rank_text(item['baseline_expected_rank']) } -> { _rank_text(item['candidate_expected_rank']) } | "
            f"top1 {item['baseline_top1_video_id']} -> {item['candidate_top1_video_id']}"
        )
    lines.append("")
    lines.append("## Top Worsened Queries")
    lines.append("")
    for item in payload["top_worsened_queries"]:
        lines.append(
            f"- query={item['query']} | expected={item['expected_video_id']} | "
            f"rank { _rank_text(item['baseline_expected_rank']) } -> { _rank_text(item['candidate_expected_rank']) } | "
            f"top1 {item['baseline_top1_video_id']} -> {item['candidate_top1_video_id']}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    baseline = read_json(Path(args.baseline))
    candidate = read_json(Path(args.candidate))
    payload = compare(baseline, candidate, top_n=args.top_n)

    export_json = Path(args.export_json)
    write_json(export_json, payload)
    export_markdown = Path(args.export_markdown)
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Saved retrieval compare JSON: {export_json}", flush=True)
    print(f"Saved retrieval compare Markdown: {export_markdown}", flush=True)


if __name__ == "__main__":
    main()
