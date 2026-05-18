from __future__ import annotations

import argparse
from collections import defaultdict

from .io_utils import read_json, write_json
from .report_paths import resolve_log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate by query bucket.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument(
        "--report-json",
    )
    parser.add_argument(
        "--bucket-summary-json",
    )
    parser.add_argument(
        "--export-json",
    )
    parser.add_argument(
        "--export-markdown",
    )
    return parser.parse_args()


def build_query_lookup(report: dict) -> dict[str, dict]:
    return {item["query"]: item for item in report.get("results", []) if item.get("query")}


def _new_bucket_stats() -> dict:
    return {"count": 0, "top1_hit_count": 0, "topk_hit_count": 0, "queries": []}


def evaluate(report: dict, bucket_summary: dict) -> dict:
    query_lookup = build_query_lookup(report)
    bucket_stats: dict[str, dict] = defaultdict(_new_bucket_stats)

    for record in bucket_summary.get("records", []):
        query = record.get("query")
        tags = record.get("tags", [])
        if not query:
            continue
        result = query_lookup.get(query)
        top1_hit = bool(result and result.get("top1_is_expected_video"))
        topk_hit = bool(result and result.get("topk_contains_expected_video"))

        for tag in tags:
            stats = bucket_stats[tag]
            stats["count"] += 1
            if top1_hit:
                stats["top1_hit_count"] += 1
            if topk_hit:
                stats["topk_hit_count"] += 1
            stats["queries"].append(
                {
                    "query": query,
                    "top1_hit": top1_hit,
                    "topk_hit": topk_hit,
                    "expected_video_id": result.get("expected_video_id") if result else None,
                }
            )

    summary_rows = []
    for tag, stats in sorted(bucket_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        count = stats["count"]
        summary_rows.append(
            {
                "bucket": tag,
                "count": count,
                "top1_hit_count": stats["top1_hit_count"],
                "topk_hit_count": stats["topk_hit_count"],
                "top1_hit_rate": round(stats["top1_hit_count"] / count, 4) if count else 0.0,
                "topk_hit_rate": round(stats["topk_hit_count"] / count, 4) if count else 0.0,
            }
        )

    return {
        "query_count": report.get("query_count", 0),
        "top_k_analyzed": report.get("top_k_analyzed", 0),
        "bucket_results": summary_rows,
        "bucket_queries": bucket_stats,
    }


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Bucket Hit Evaluation")
    lines.append("")
    lines.append(f"- Query count: {payload['query_count']}")
    lines.append(f"- Top K analyzed: {payload['top_k_analyzed']}")
    lines.append("")
    lines.append("| Bucket | Count | Top1 | Top1 Rate | TopK | TopK Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in payload["bucket_results"]:
        lines.append(
            f"| {item['bucket']} | {item['count']} | {item['top1_hit_count']} | {item['top1_hit_rate']} | "
            f"{item['topk_hit_count']} | {item['topk_hit_rate']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    report_json = resolve_log_path(args.profile, args.report_json, "retrieval_gap_report.json")
    bucket_summary_json = resolve_log_path(args.profile, args.bucket_summary_json, "query_bucket_summary.json")
    export_json = resolve_log_path(args.profile, args.export_json, "bucket_hit_eval.json")
    export_markdown = resolve_log_path(args.profile, args.export_markdown, "bucket_hit_eval.md")
    report = read_json(report_json)
    bucket_summary = read_json(bucket_summary_json)
    payload = evaluate(report, bucket_summary)
    write_json(export_json, payload)
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Saved bucket hit evaluation JSON: {export_json}", flush=True)
    print(f"Saved bucket hit evaluation Markdown: {export_markdown}", flush=True)


if __name__ == "__main__":
    main()
