from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Analyze top1 rerank failure cases.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--gap-report-json")
    parser.add_argument("--recall-stage-json")
    parser.add_argument("--export-json")
    parser.add_argument("--export-markdown")
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def _gap_lookup(payload: dict) -> dict[str, dict]:
    return {item["query"]: item for item in payload.get("results", []) if item.get("query")}


def _recall_lookup(payload: dict) -> dict[str, dict]:
    return {item["query"]: item for item in payload.get("results", []) if item.get("query")}


def _classify_case(gap_item: dict, recall_item: dict) -> list[str]:
    labels: list[str] = []
    top1 = (gap_item.get("results") or [None])[0]
    if not top1:
        return ["无结果"]

    gap = top1.get("gap") or {}
    missing = gap.get("missing") or {}
    query_facets = gap.get("query_facets") or {}
    description_facets = gap.get("description_facets") or {}

    missing_facets = [facet for facet in FACET_NAMES if missing.get(facet)]
    if missing_facets:
        labels.append("Facet缺失:" + "+".join(FACET_LABELS[facet] for facet in missing_facets))
    else:
        labels.append("Facet表面一致但排序失败")

    if query_facets.get("subject") and description_facets.get("subject"):
        q_subject = set(query_facets["subject"])
        d_subject = set(description_facets["subject"])
        if q_subject != d_subject:
            labels.append("主体不一致")

    if query_facets.get("attribute") and description_facets.get("attribute"):
        q_attr = set(query_facets["attribute"])
        d_attr = set(description_facets["attribute"])
        if q_attr != d_attr:
            labels.append("属性不一致")

    if query_facets.get("action") and description_facets.get("action"):
        q_action = set(query_facets["action"])
        d_action = set(description_facets["action"])
        if q_action != d_action:
            labels.append("动作不一致")

    final_rank = recall_item.get("final_video_rank")
    merged_video_rank = recall_item.get("merged_video_rank")
    merged_segment_rank = recall_item.get("merged_segment_rank")
    if merged_video_rank == 1 and merged_segment_rank == 1 and final_rank and final_rank > 1:
        labels.append("候选第一但终排打掉")
    elif merged_video_rank and merged_video_rank <= 3 and final_rank and final_rank > merged_video_rank:
        labels.append("高位候选被终排压制")
    elif merged_video_rank and merged_video_rank >= 8 and final_rank and final_rank > 1:
        labels.append("候选位置偏后且未拉正")

    if final_rank == 2:
        labels.append("只差一步到Top1")
    elif final_rank and final_rank >= 5:
        labels.append("终排偏移较大")

    return labels


def analyze(gap_report: dict, recall_stage: dict, top_n: int) -> dict:
    gap_lookup = _gap_lookup(gap_report)
    recall_lookup = _recall_lookup(recall_stage)

    label_counter: Counter[str] = Counter()
    facet_counter: Counter[str] = Counter()
    examples_by_label: dict[str, list[dict]] = defaultdict(list)
    rerank_failures: list[dict] = []

    for query, recall_item in recall_lookup.items():
        if recall_item.get("terminal_stage") != "top1_rerank_loss":
            continue
        gap_item = gap_lookup.get(query)
        if not gap_item:
            continue
        top1 = (gap_item.get("results") or [None])[0]
        if not top1:
            continue
        labels = _classify_case(gap_item, recall_item)
        missing = (top1.get("gap") or {}).get("missing") or {}
        missing_facet_names = [facet for facet in FACET_NAMES if missing.get(facet)]
        for facet in missing_facet_names:
            facet_counter[FACET_LABELS[facet]] += 1

        row = {
            "query": query,
            "expected_video_id": gap_item.get("expected_video_id"),
            "top1_video_id": top1.get("video_id"),
            "final_video_rank": recall_item.get("final_video_rank"),
            "merged_video_rank": recall_item.get("merged_video_rank"),
            "merged_segment_rank": recall_item.get("merged_segment_rank"),
            "labels": labels,
            "missing_facets": missing_facet_names,
            "query_facets": (top1.get("gap") or {}).get("query_facets") or {},
            "top1_facets": (top1.get("gap") or {}).get("description_facets") or {},
        }
        rerank_failures.append(row)
        for label in labels:
            label_counter[label] += 1
            if len(examples_by_label[label]) < top_n:
                examples_by_label[label].append(row)

    rerank_failures.sort(
        key=lambda item: (
            item["final_video_rank"] if item["final_video_rank"] is not None else 999999,
            item["merged_video_rank"] if item["merged_video_rank"] is not None else 999999,
            item["query"],
        )
    )
    return {
        "query_count": int(recall_stage.get("query_count", 0)),
        "rerank_failure_count": len(rerank_failures),
        "rerank_failure_rate": round(len(rerank_failures) / int(recall_stage.get("query_count", 1)), 4),
        "label_counts": [{"label": label, "count": count} for label, count in label_counter.most_common()],
        "missing_facet_counts": [{"facet": facet, "count": count} for facet, count in facet_counter.most_common()],
        "examples_by_label": dict(examples_by_label),
        "top_rerank_failures": rerank_failures[:top_n],
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Top1重排失败分析",
        "",
        f"- Query 数：{payload['query_count']}",
        f"- Top1重排失败数：{payload['rerank_failure_count']}",
        f"- Top1重排失败率：{payload['rerank_failure_rate']}",
        "",
        "## 失败类型统计",
        "",
    ]
    for item in payload["label_counts"]:
        lines.append(f"- {item['label']}: {item['count']}")

    lines.extend(["", "## 缺失Facet统计", ""])
    for item in payload["missing_facet_counts"]:
        lines.append(f"- {item['facet']}: {item['count']}")

    lines.extend(["", "## 典型失败样例", ""])
    for item in payload["top_rerank_failures"][:20]:
        lines.append(
            f"- query={item['query']} | expected={item['expected_video_id']} | top1={item['top1_video_id']} | "
            f"video_rank={item['merged_video_rank']} | segment_rank={item['merged_segment_rank']} | "
            f"final_rank={item['final_video_rank']} | labels={','.join(item['labels'])}"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    gap_report_path = latest_log_path(args.profile, args.gap_report_json, "retrieval_gap_report.json")
    recall_stage_path = latest_log_path(args.profile, args.recall_stage_json, "recall_stage_analysis.json")
    payload = analyze(read_json(gap_report_path), read_json(recall_stage_path), top_n=args.top_n)
    export_json = timestamped_log_path(args.profile, args.export_json, "rerank_failure_analysis.json")
    write_json(export_json, payload)
    run_timestamp = export_json.stem.removeprefix("rerank_failure_analysis_")
    export_markdown = timestamped_log_path(
        args.profile,
        args.export_markdown,
        "rerank_failure_analysis.md",
        timestamp=run_timestamp,
    )
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved rerank failure analysis JSON: {export_json}", flush=True)
    print(f"Saved rerank failure analysis Markdown: {export_markdown}", flush=True)


if __name__ == "__main__":
    main()
