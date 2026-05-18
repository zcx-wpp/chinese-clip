from __future__ import annotations

import argparse
from collections import Counter

from .io_utils import read_json, write_json
from .report_paths import resolve_log_path


COUNT_GROUP = {"一个人", "两个人", "多人", "一群人", "有人"}
MALE_GROUP = {"男人", "男生", "男孩"}
FEMALE_GROUP = {"女人", "女生", "女孩"}
CHILD_GROUP = {"小孩", "孩子", "宝宝", "男孩", "女孩"}
ELDER_GROUP = {"老人"}
GENERIC_GROUP = {"人物", "有人"}


def parse_args():
    parser = argparse.ArgumentParser(description="Break down subject-related retrieval misses into concrete error types.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--report-json")
    parser.add_argument("--export-json")
    parser.add_argument("--export-markdown")
    return parser.parse_args()


def _subject_values(result_item: dict) -> tuple[list[str], list[str], list[str]]:
    top1 = (result_item.get("results") or [None])[0]
    if not top1:
        return [], [], []
    gap = top1.get("gap") or {}
    query_subjects = list((gap.get("query_facets") or {}).get("subject") or [])
    desc_subjects = list((gap.get("description_facets") or {}).get("subject") or [])
    missing_subjects = list((gap.get("missing") or {}).get("subject") or [])
    return query_subjects, desc_subjects, missing_subjects


def _has_any(values: set[str], group: set[str]) -> bool:
    return bool(values & group)


def classify_subject_error(query_subjects: list[str], desc_subjects: list[str], missing_subjects: list[str]) -> list[str]:
    query_set = set(query_subjects)
    desc_set = set(desc_subjects)
    missing_set = set(missing_subjects)
    labels: list[str] = []

    if not query_set:
        return labels

    if _has_any(query_set, COUNT_GROUP):
        if not _has_any(desc_set, COUNT_GROUP):
            labels.append("人数缺失")
        elif missing_set & COUNT_GROUP:
            labels.append("人数不匹配")

    if _has_any(query_set, MALE_GROUP) and _has_any(desc_set, FEMALE_GROUP):
        labels.append("性别反转")
    elif _has_any(query_set, FEMALE_GROUP) and _has_any(desc_set, MALE_GROUP):
        labels.append("性别反转")
    elif missing_set & (MALE_GROUP | FEMALE_GROUP):
        labels.append("性别缺失")

    if _has_any(query_set, CHILD_GROUP) and not _has_any(desc_set, CHILD_GROUP):
        labels.append("年龄层不匹配")
    elif _has_any(query_set, ELDER_GROUP) and not _has_any(desc_set, ELDER_GROUP):
        labels.append("年龄层不匹配")

    specific_query = query_set - GENERIC_GROUP
    specific_desc = desc_set - GENERIC_GROUP
    if specific_query and not specific_desc:
        labels.append("主体过泛")

    if missing_set and not labels:
        labels.append("主体缺失")
    if len(query_set) > 1 and len(desc_set) > 1 and missing_set:
        labels.append("复合主体混淆")

    deduped: list[str] = []
    seen = set()
    for label in labels:
        if label not in seen:
            deduped.append(label)
            seen.add(label)
    return deduped or (["主体缺失"] if missing_set else [])


def build_summary(report: dict) -> dict:
    counter: Counter[str] = Counter()
    examples_by_label: dict[str, list[dict]] = {}
    analyzed = []

    for item in report.get("results", []):
        if item.get("top1_is_expected_video"):
            continue
        query_subjects, desc_subjects, missing_subjects = _subject_values(item)
        if not query_subjects:
            continue
        labels = classify_subject_error(query_subjects, desc_subjects, missing_subjects)
        if not labels:
            continue

        top1 = (item.get("results") or [None])[0]
        analyzed.append(
            {
                "query": item.get("query"),
                "expected_video_id": item.get("expected_video_id"),
                "top1_video_id": top1.get("video_id") if top1 else None,
                "query_subjects": query_subjects,
                "top1_subjects": desc_subjects,
                "missing_subjects": missing_subjects,
                "labels": labels,
            }
        )
        for label in labels:
            counter[label] += 1
            examples_by_label.setdefault(label, [])
            if len(examples_by_label[label]) < 8:
                examples_by_label[label].append(analyzed[-1])

    return {
        "query_count": int(report.get("query_count", 0)),
        "top1_hit_rate": report.get("top1_hit_rate"),
        "subject_miss_cases": len(analyzed),
        "label_counts": [{"label": label, "count": count} for label, count in counter.most_common()],
        "examples_by_label": examples_by_label,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Subject Error Breakdown",
        "",
        f"- Query count: {summary['query_count']}",
        f"- Top1 hit rate: {summary['top1_hit_rate']}",
        f"- Subject-related miss cases: {summary['subject_miss_cases']}",
        "",
        "## Label Counts",
        "",
    ]
    for item in summary["label_counts"]:
        lines.append(f"- {item['label']}: {item['count']}")

    for label, examples in summary["examples_by_label"].items():
        lines.extend(["", f"## {label}", ""])
        for example in examples:
            query_subjects = "、".join(example["query_subjects"]) if example["query_subjects"] else "无"
            top1_subjects = "、".join(example["top1_subjects"]) if example["top1_subjects"] else "无"
            missing_subjects = "、".join(example["missing_subjects"]) if example["missing_subjects"] else "无"
            lines.append(
                f"- query={example['query']} | expected={example['expected_video_id']} | top1={example['top1_video_id']} | "
                f"query_subjects={query_subjects} | top1_subjects={top1_subjects} | missing={missing_subjects}"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    report_json = resolve_log_path(args.profile, args.report_json, "retrieval_gap_report.json")
    export_json = resolve_log_path(args.profile, args.export_json, "subject_error_breakdown.json")
    export_markdown = resolve_log_path(args.profile, args.export_markdown, "subject_error_breakdown.md")
    report = read_json(report_json)
    summary = build_summary(report)
    write_json(export_json, summary)
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Saved subject error breakdown JSON: {export_json}")
    print(f"Saved subject error breakdown Markdown: {export_markdown}")


if __name__ == "__main__":
    main()
