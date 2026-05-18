from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .config import PROJECT_ROOT
from .io_utils import read_json, write_json
from .report_paths import latest_log_path, timestamped_log_path


FACET_KEYWORDS = {
    "subject": (
        "女孩", "女生", "女人", "男孩", "男生", "男人", "小孩", "孩子",
        "宝宝", "老人", "一个人", "两个人", "多人", "一群人", "人物", "有人",
    ),
    "action": (
        "跳舞", "舞蹈", "唱歌", "说话", "讲话", "交谈", "走路", "行走", "跑步", "跳跃",
        "挥手", "微笑", "看向镜头", "表演", "展示", "站着", "坐着", "梳头", "吹蜡烛",
        "打球", "比试腕力", "运动", "互动",
    ),
    "scene": (
        "室内", "户外", "房间", "街道", "舞台", "广场", "保龄球室", "健身房",
        "桌子", "镜头前", "近景", "全身", "空地",
    ),
    "attribute": (
        "红衣", "红色", "蓝衣", "蓝色", "白衣", "白色", "黑衣", "黑色", "绿衣", "绿色",
        "黄衣", "黄色", "长发", "短发", "戴着帽子", "拿着麦克风",
    ),
}

FACET_LABELS = {
    "subject": "主体",
    "action": "动作",
    "scene": "场景",
    "attribute": "属性",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare retrieval results with actual Chinese descriptions.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument(
        "--batch-results",
    )
    parser.add_argument(
        "--metadata-jsonl",
        default=str(PROJECT_ROOT.parent / "data" / "metadata_zh_only.jsonl"),
    )
    parser.add_argument(
        "--eval-labels",
        default=str(PROJECT_ROOT / "metadata" / "1799eval_labels.json"),
    )
    parser.add_argument(
        "--export",
    )
    parser.add_argument(
        "--export-markdown",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_id = item.get("videoID") or item.get("video_id")
            if video_id:
                mapping[video_id] = item
    return mapping


def load_eval_labels(path: Path) -> dict[str, dict]:
    items = read_json(path)
    return {item["query"]: item for item in items if item.get("query")}


def extract_facets(text: str) -> dict[str, list[str]]:
    return {facet_name: [keyword for keyword in keywords if keyword in text] for facet_name, keywords in FACET_KEYWORDS.items()}


def summarize_gap(query: str, description: str) -> dict:
    query_facets = extract_facets(query)
    desc_facets = extract_facets(description)

    matched, missing, extra = {}, {}, {}
    for facet_name in FACET_KEYWORDS:
        query_set, desc_set = set(query_facets[facet_name]), set(desc_facets[facet_name])
        matched[facet_name] = sorted(query_set & desc_set)
        missing[facet_name] = sorted(query_set - desc_set)
        extra[facet_name] = sorted(desc_set - query_set)

    return {
        "query_facets": query_facets,
        "description_facets": desc_facets,
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }


def choose_reference_description(item: dict) -> str:
    captions = item.get("chCap") or item.get("captions") or []
    if isinstance(captions, list) and captions:
        return str(captions[0])
    return str(item.get("description") or "")


def _join_values(values: list[str]) -> str:
    return "、".join(values)


def _hit_text(is_hit: bool) -> str:
    return "是" if is_hit else "否"


def build_human_summary(query: str, expected_video_id: str | None, top1: dict | None) -> str:
    if top1 is None:
        if expected_video_id:
            return f"query 想找“{query}”，对应视频应该是“{expected_video_id}”，但当前没有检索到结果。"
        return f"query 想找“{query}”，但当前没有检索到结果。"

    description = top1["reference_description"] or "暂无描述"
    missing = top1["gap"]["missing"]
    matched = top1["gap"]["matched"]
    extra = top1["gap"]["extra"]
    is_expected = top1["is_expected_video"]

    matched_parts = []
    missing_parts = []
    extra_parts = []
    for facet_name in FACET_LABELS:
        if matched[facet_name]:
            matched_parts.append(f"{FACET_LABELS[facet_name]}上命中了{_join_values(matched[facet_name])}")
        if missing[facet_name]:
            missing_parts.append(f"缺少{FACET_LABELS[facet_name]}{_join_values(missing[facet_name])}")
        if extra[facet_name]:
            extra_parts.append(f"实际更偏向{FACET_LABELS[facet_name]}{_join_values(extra[facet_name])}")

    prefix = f"query 想找“{query}”，"
    if expected_video_id:
        prefix += f"对应视频应该是“{expected_video_id}”，"
    if is_expected:
        summary = prefix + f"当前 Top1 已命中对应视频，实际描述是“{description}”。"
    else:
        summary = prefix + f"但当前 Top1 没有命中对应视频，实际命中的是“{top1['video_id']}”，描述是“{description}”。"

    if matched_parts:
        summary += f" 当前结果{'，'.join(matched_parts)}。"
    if missing_parts:
        summary += f" 同时{'，'.join(missing_parts)}。"
    if extra_parts:
        summary += f" 另外，这条结果{'，'.join(extra_parts)}。"
    return summary


def analyze(
    batch_results_path: Path,
    metadata_path: Path,
    eval_labels_path: Path,
    top_k: int,
) -> dict:
    batch_payload = read_json(batch_results_path)
    metadata_map = load_metadata(metadata_path)
    eval_map = load_eval_labels(eval_labels_path)

    analyzed_results = []
    top1_hit_count = 0
    topk_hit_count = 0
    cumulative_hit_counts = [0] * max(top_k, 0)

    for query_item in batch_payload.get("results", []):
        query = query_item["query"]
        eval_item = eval_map.get(query, {})
        expected_video_id = eval_item.get("video_id")
        expected_segments = eval_item.get("segments", [])

        ranked_items = []
        expected_hit_in_top_k = False
        for rank, result in enumerate(query_item.get("results", [])[:top_k], start=1):
            video_id = result["video_id"]
            metadata_item = metadata_map.get(video_id, {})
            description = choose_reference_description(metadata_item)
            gap = summarize_gap(query, description)
            is_expected_video = bool(expected_video_id and video_id == expected_video_id)
            if is_expected_video:
                expected_hit_in_top_k = True
            ranked_items.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "score": result.get("score"),
                    "video_path": result.get("video_path"),
                    "reference_description": description,
                    "segments": result.get("segments", []),
                    "gap": gap,
                    "is_expected_video": is_expected_video,
                }
            )

        first_expected_rank = next(
            (result["rank"] for result in ranked_items if result["is_expected_video"]),
            None,
        )
        if first_expected_rank is not None:
            for idx in range(first_expected_rank - 1, top_k):
                cumulative_hit_counts[idx] += 1

        top1 = ranked_items[0] if ranked_items else None
        top1_is_expected = bool(top1 and top1["is_expected_video"])
        if top1_is_expected:
            top1_hit_count += 1
        if expected_hit_in_top_k:
            topk_hit_count += 1

        analyzed_results.append(
            {
                "query": query,
                "expected_video_id": expected_video_id,
                "expected_segments": expected_segments,
                "top1_is_expected_video": top1_is_expected,
                "topk_contains_expected_video": expected_hit_in_top_k,
                "human_summary": build_human_summary(query, expected_video_id, top1),
                "results": ranked_items,
            }
        )

    query_count = len(analyzed_results)
    cumulative_hit_rates = {
        f"top{idx + 1}_hit_rate": round(cumulative_hit_counts[idx] / query_count, 4) if query_count else 0.0
        for idx in range(top_k)
    }
    cumulative_hit_count_map = {
        f"top{idx + 1}_hit_count": cumulative_hit_counts[idx]
        for idx in range(top_k)
    }
    return {
        "query_count": query_count,
        "top_k_analyzed": top_k,
        "top1_hit_count": top1_hit_count,
        "topk_hit_count": topk_hit_count,
        "top1_hit_rate": round(top1_hit_count / query_count, 4) if query_count else 0.0,
        "topk_hit_rate": round(topk_hit_count / query_count, 4) if query_count else 0.0,
        "cumulative_hit_counts": cumulative_hit_count_map,
        "cumulative_hit_rates": cumulative_hit_rates,
        "results": analyzed_results,
    }


def format_gap_line(title: str, values: dict[str, list[str]]) -> str:
    parts = []
    for facet_name in FACET_LABELS:
        facet_values = values.get(facet_name, [])
        if facet_values:
            parts.append(f"{FACET_LABELS[facet_name]}: {_join_values(facet_values)}")
    return f"- {title}: " + ("; ".join(parts) if parts else "无")


def render_markdown_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Retrieval Gap Report")
    lines.append("")
    lines.append(f"- Query count: {payload['query_count']}")
    lines.append(f"- Top K analyzed: {payload['top_k_analyzed']}")
    lines.append(f"- Top1 hit rate: {payload['top1_hit_count']}/{payload['query_count']} = {payload['top1_hit_rate']}")
    lines.append(f"- TopK hit rate: {payload['topk_hit_count']}/{payload['query_count']} = {payload['topk_hit_rate']}")
    cumulative_hit_rates = payload.get("cumulative_hit_rates", {})
    cumulative_hit_counts = payload.get("cumulative_hit_counts", {})
    if cumulative_hit_rates:
        lines.append("- Cumulative hit rates:")
        for idx in range(1, payload["top_k_analyzed"] + 1):
            count_key = f"top{idx}_hit_count"
            rate_key = f"top{idx}_hit_rate"
            if rate_key in cumulative_hit_rates:
                lines.append(
                    f"  - Top{idx}: {cumulative_hit_counts.get(count_key, 0)}/{payload['query_count']} = {cumulative_hit_rates[rate_key]}"
                )
    lines.append("")

    for idx, item in enumerate(payload["results"], start=1):
        lines.append(f"## {idx}. {item['query']}")
        lines.append("")
        if item["expected_video_id"]:
            lines.append(f"- 对应视频: `{item['expected_video_id']}`")
            lines.append(f"- Top1 是否命中对应视频: {_hit_text(item['top1_is_expected_video'])}")
            lines.append(f"- TopK 是否包含对应视频: {_hit_text(item['topk_contains_expected_video'])}")
            if item["expected_segments"]:
                expected_segment_text = ", ".join(
                    f"{seg['start']}-{seg['end']}" for seg in item["expected_segments"]
                )
                lines.append(f"- 标注时间段: {expected_segment_text}")
        else:
            lines.append("- 对应视频: 未找到")
        lines.append("")
        if item["human_summary"]:
            lines.append(item["human_summary"])
            lines.append("")
        if not item["results"]:
            lines.append("未检索到结果。")
            lines.append("")
            continue

        for result in item["results"]:
            lines.append(
                f"### Top {result['rank']} - {result['video_id']} (score={result['score']}, expected={_hit_text(result['is_expected_video'])})"
            )
            lines.append("")
            lines.append(f"- 真实描述: {result['reference_description'] or '暂无描述'}")
            lines.append(f"- 视频路径: `{result['video_path']}`")
            if result["segments"]:
                segment_text = ", ".join(
                    f"{seg['start']}-{seg['end']} (score={seg['score']})"
                    for seg in result["segments"]
                )
                lines.append(f"- 命中片段: {segment_text}")
            else:
                lines.append("- 命中片段: 无")
            lines.append(format_gap_line("命中点", result["gap"]["matched"]))
            lines.append(format_gap_line("缺失点", result["gap"]["missing"]))
            lines.append(format_gap_line("额外点", result["gap"]["extra"]))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_growth_curve(payload: dict, output_path: Path) -> None:
    top_k = int(payload.get("top_k_analyzed") or 0)
    if top_k <= 0:
        return

    x = list(range(1, top_k + 1))
    cumulative_rates = payload.get("cumulative_hit_rates", {})
    y = [float(cumulative_rates.get(f"top{i}_hit_rate", 0.0)) for i in x]

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(x, y, marker="o", linewidth=2, markersize=4, color="#2563eb")
    for xi, yi in zip(x, y):
        plt.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    plt.title("Retrieval Hit-Rate Growth Curve")
    plt.xlabel("Top-K")
    plt.ylabel("Cumulative Hit Rate")
    plt.xticks(x)
    plt.ylim(0, max(0.1, min(1.0, max(y, default=0.0) + 0.1)))
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    run_timestamp = None
    payload = analyze(
        batch_results_path=latest_log_path(args.profile, args.batch_results, "batch_search_results.json"),
        metadata_path=Path(args.metadata_jsonl),
        eval_labels_path=Path(args.eval_labels),
        top_k=args.top_k,
    )

    export_path = timestamped_log_path(args.profile, args.export, "retrieval_gap_report.json")
    run_timestamp = export_path.stem.removeprefix("retrieval_gap_report_")
    write_json(export_path, payload)

    markdown_path = timestamped_log_path(
        args.profile,
        args.export_markdown,
        "retrieval_gap_report.md",
        timestamp=run_timestamp,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")

    growth_curve_path = timestamped_log_path(
        args.profile,
        None,
        "retrieval_growth_curve.png",
        timestamp=run_timestamp,
    )
    render_growth_curve(payload, growth_curve_path)

    for idx, item in enumerate(payload["results"], start=1):
        if not item["results"]:
            print(f"[gap] {idx}/{payload['query_count']} query={item['query']} top1=NONE expected={item['expected_video_id'] or 'NONE'}", flush=True)
            continue
        top1 = item["results"][0]
        print(
            f"[gap] {idx}/{payload['query_count']} query={item['query']} "
            f"top1={top1['video_id']} expected={item['expected_video_id'] or 'NONE'} "
            f"hit={_hit_text(item['top1_is_expected_video'])}",
            flush=True,
        )

    print(f"Saved retrieval gap report: {export_path}", flush=True)
    print(f"Saved readable retrieval report: {markdown_path}", flush=True)
    print(f"Saved growth curve: {growth_curve_path}", flush=True)


if __name__ == "__main__":
    main()
