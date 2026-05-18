from __future__ import annotations

import argparse
from collections import Counter

from .io_utils import read_json, write_json
from .report_paths import latest_log_path, timestamped_log_path


FACET_NAMES = ("subject", "action", "scene", "attribute")
FACET_LABELS = {
    "subject": "主体",
    "action": "动作",
    "scene": "场景",
    "attribute": "属性",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize retrieval gap report into high-level statistics.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument(
        "--report-json",
    )
    parser.add_argument(
        "--export-json",
    )
    parser.add_argument(
        "--export-markdown",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def classify_query(result_item: dict) -> list[str]:
    query = result_item.get("query", "")
    top1 = (result_item.get("results") or [None])[0]
    query_facets = {}
    if top1 and top1.get("gap"):
        query_facets = top1["gap"].get("query_facets", {})

    tags: list[str] = []
    subject_values = query_facets.get("subject", [])
    action_values = query_facets.get("action", [])
    scene_values = query_facets.get("scene", [])
    attribute_values = query_facets.get("attribute", [])

    if len(subject_values) >= 2 or any(item in query for item in ("两个人", "多人", "一群人")):
        tags.append("多人")
    elif subject_values:
        tags.append("单人/主体明确")

    if action_values:
        tags.append("动作类")
    if scene_values:
        tags.append("场景类")
    if attribute_values:
        tags.append("属性类")

    if len(query) >= 25:
        tags.append("长描述")
    else:
        tags.append("短描述")

    if not tags:
        tags.append("未分类")
    return tags


def summarize(report: dict, top_n: int) -> dict:
    query_count = int(report.get("query_count", 0))
    top1_hit_count = int(report.get("top1_hit_count", 0))
    topk_hit_count = int(report.get("topk_hit_count", 0))
    cumulative_hit_counts = report.get("cumulative_hit_counts", {})
    cumulative_hit_rates = report.get("cumulative_hit_rates", {})

    facet_missing_counter: Counter[str] = Counter()
    facet_top1_miss_counter: Counter[str] = Counter()
    tag_totals: Counter[str] = Counter()
    tag_top1_hits: Counter[str] = Counter()
    tag_topk_hits: Counter[str] = Counter()
    wrong_video_counter: Counter[str] = Counter()
    expected_miss_examples: list[dict] = []
    strong_hit_examples: list[dict] = []
    error_pattern_counter: Counter[str] = Counter()
    facet_value_missing_counter: dict[str, Counter[str]] = {name: Counter() for name in FACET_NAMES}

    for item in report.get("results", []):
        tags = classify_query(item)
        top1_hit = bool(item.get("top1_is_expected_video"))
        topk_hit = bool(item.get("topk_contains_expected_video"))
        expected_video_id = item.get("expected_video_id")
        top1 = (item.get("results") or [None])[0]

        for tag in tags:
            tag_totals[tag] += 1
            if top1_hit:
                tag_top1_hits[tag] += 1
            if topk_hit:
                tag_topk_hits[tag] += 1

        if top1 and not top1_hit:
            wrong_video_counter[top1["video_id"]] += 1
            missing = top1.get("gap", {}).get("missing", {})
            missing_facets = []
            for facet_name in FACET_NAMES:
                values = missing.get(facet_name, [])
                if values:
                    facet_top1_miss_counter[facet_name] += 1
                    missing_facets.append(FACET_LABELS[facet_name])
                    for value in values:
                        facet_value_missing_counter[facet_name][value] += 1
            error_pattern_counter["+".join(missing_facets) if missing_facets else "无明显缺失"] += 1
            expected_miss_examples.append(
                {
                    "query": item["query"],
                    "expected_video_id": expected_video_id,
                    "top1_video_id": top1["video_id"],
                    "score": top1.get("score"),
                    "missing": missing,
                }
            )

        if top1 and top1_hit:
            strong_hit_examples.append(
                {
                    "query": item["query"],
                    "expected_video_id": expected_video_id,
                    "score": top1.get("score"),
                }
            )

        if top1 and top1.get("gap"):
            missing = top1["gap"].get("missing", {})
            for facet_name in FACET_NAMES:
                if missing.get(facet_name):
                    facet_missing_counter[facet_name] += 1

    tag_breakdown = []
    for tag, total in sorted(tag_totals.items(), key=lambda kv: (-kv[1], kv[0])):
        tag_breakdown.append(
            {
                "tag": tag,
                "count": total,
                "top1_hit_count": tag_top1_hits[tag],
                "topk_hit_count": tag_topk_hits[tag],
                "top1_hit_rate": round(tag_top1_hits[tag] / total, 4) if total else 0.0,
                "topk_hit_rate": round(tag_topk_hits[tag] / total, 4) if total else 0.0,
            }
        )

    return {
        "query_count": query_count,
        "top1_hit_count": top1_hit_count,
        "topk_hit_count": topk_hit_count,
        "top1_hit_rate": round(top1_hit_count / query_count, 4) if query_count else 0.0,
        "topk_hit_rate": round(topk_hit_count / query_count, 4) if query_count else 0.0,
        "cumulative_hit_counts": cumulative_hit_counts,
        "cumulative_hit_rates": cumulative_hit_rates,
        "facet_missing_counts": dict(facet_missing_counter),
        "facet_top1_miss_counts": dict(facet_top1_miss_counter),
        "top_missing_values": {
            facet_name: [{"value": value, "count": count} for value, count in counter.most_common(top_n)]
            for facet_name, counter in facet_value_missing_counter.items()
        },
        "query_type_breakdown": tag_breakdown,
        "most_common_wrong_top1_videos": [
            {"video_id": video_id, "count": count}
            for video_id, count in wrong_video_counter.most_common(top_n)
        ],
        "most_common_error_patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in error_pattern_counter.most_common(top_n)
        ],
        "top_miss_examples": expected_miss_examples[:top_n],
        "top_hit_examples": strong_hit_examples[:top_n],
    }


def render_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Gap Summary")
    lines.append("")
    lines.append(f"- Query count: {summary['query_count']}")
    lines.append(f"- Top1 hit rate: {summary['top1_hit_count']}/{summary['query_count']} = {summary['top1_hit_rate']}")
    lines.append(f"- TopK hit rate: {summary['topk_hit_count']}/{summary['query_count']} = {summary['topk_hit_rate']}")
    cumulative_hit_rates = summary.get("cumulative_hit_rates", {})
    cumulative_hit_counts = summary.get("cumulative_hit_counts", {})
    if cumulative_hit_rates:
        lines.append("- Cumulative hit rates:")
        for idx in range(1, len(cumulative_hit_rates) + 1):
            count_key = f"top{idx}_hit_count"
            rate_key = f"top{idx}_hit_rate"
            lines.append(
                f"  - Top{idx}: {cumulative_hit_counts.get(count_key, 0)}/{summary['query_count']} = {cumulative_hit_rates.get(rate_key, 0.0)}"
            )
    lines.append("")

    lines.append("## Query Type Breakdown")
    lines.append("")
    for item in summary["query_type_breakdown"]:
        lines.append(
            f"- {item['tag']}: count={item['count']}, "
            f"top1={item['top1_hit_count']}/{item['count']} ({item['top1_hit_rate']}), "
            f"topk={item['topk_hit_count']}/{item['count']} ({item['topk_hit_rate']})"
        )
    lines.append("")

    lines.append("## Missing Facets")
    lines.append("")
    for facet_name in FACET_NAMES:
        count = summary["facet_missing_counts"].get(facet_name, 0)
        top_values = summary["top_missing_values"].get(facet_name, [])
        value_text = ", ".join(f"{item['value']}({item['count']})" for item in top_values[:10]) or "无"
        lines.append(f"- {FACET_LABELS[facet_name]}: miss_count={count}; top_missing={value_text}")
    lines.append("")

    lines.append("## Common Error Patterns")
    lines.append("")
    for item in summary["most_common_error_patterns"][:15]:
        lines.append(f"- {item['pattern']}: {item['count']}")
    lines.append("")

    lines.append("## Common Wrong Top1 Videos")
    lines.append("")
    for item in summary["most_common_wrong_top1_videos"][:15]:
        lines.append(f"- {item['video_id']}: {item['count']}")
    lines.append("")

    lines.append("## Miss Examples")
    lines.append("")
    for item in summary["top_miss_examples"][:10]:
        missing_parts = []
        for facet_name in FACET_NAMES:
            values = item["missing"].get(facet_name, [])
            if values:
                missing_parts.append(f"{FACET_LABELS[facet_name]}={','.join(values)}")
        missing_text = "; ".join(missing_parts) if missing_parts else "无明显缺失"
        lines.append(
            f"- query={item['query']} | expected={item['expected_video_id']} | top1={item['top1_video_id']} | missing={missing_text}"
        )
    lines.append("")

    lines.append("## Hit Examples")
    lines.append("")
    for item in summary["top_hit_examples"][:10]:
        lines.append(
            f"- query={item['query']} | expected={item['expected_video_id']} | score={item['score']}"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    report_path = latest_log_path(args.profile, args.report_json, "retrieval_gap_report.json")
    export_json = timestamped_log_path(args.profile, args.export_json, "retrieval_gap_summary.json")
    run_timestamp = export_json.stem.removeprefix("retrieval_gap_summary_")
    export_markdown = timestamped_log_path(
        args.profile,
        args.export_markdown,
        "retrieval_gap_summary.md",
        timestamp=run_timestamp,
    )
    report = read_json(report_path)
    summary = summarize(report, top_n=args.top_n)
    write_json(export_json, summary)
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Saved retrieval gap summary JSON: {export_json}", flush=True)
    print(f"Saved retrieval gap summary Markdown: {export_markdown}", flush=True)


if __name__ == "__main__":
    main()
