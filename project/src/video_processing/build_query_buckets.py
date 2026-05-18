from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .config import PROJECT_ROOT
from .io_utils import read_nonempty_lines, write_json
from .profile_paths import default_logs_dir, default_query_bucket_dir
from .profile_paths import resolve_path


BUCKET_RULES = {
    "subject": ("女孩", "女生", "女人", "男孩", "男生", "男人", "小孩", "孩子", "老人", "人物", "有人"),
    "action": ("跳舞", "舞蹈", "唱歌", "说话", "讲话", "交谈", "走路", "行走", "跑步", "跳跃", "挥手", "微笑", "表演", "展示", "射箭", "滑动", "梳头", "吹"),
    "scene": ("室内", "户外", "房间", "街道", "舞台", "广场", "保龄球室", "教室", "厨房", "草坪", "雪地", "公路", "树林", "镜子"),
    "attribute": ("红色", "红衣", "蓝色", "蓝衣", "白色", "黑色", "绿色", "黄色", "长发", "短发", "戴着帽子", "带帽子"),
    "multi_person": ("两个人", "多人", "一群人", "很多人", "几个人", "台下很多人"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Bucket queries by coarse semantic type.")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument(
        "--queries-file",
        default=str(PROJECT_ROOT / "metadata" / "sample_queries.txt"),
    )
    parser.add_argument(
        "--export-dir",
    )
    parser.add_argument(
        "--export-summary",
    )
    return parser.parse_args()


def classify_query(query: str) -> list[str]:
    tags: list[str] = []
    for bucket_name, keywords in BUCKET_RULES.items():
        if any(keyword in query for keyword in keywords):
            tags.append(bucket_name)
    if len(query) >= 25:
        tags.append("long_description")
    else:
        tags.append("short_description")
    if not tags:
        tags.append("other")
    return tags


def main():
    args = parse_args()
    queries = read_nonempty_lines(Path(args.queries_file))
    export_dir = resolve_path(args.export_dir, default_query_bucket_dir(args.profile))
    export_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[str]] = defaultdict(list)
    records = []
    for query in queries:
        tags = classify_query(query)
        records.append({"query": query, "tags": tags})
        for tag in tags:
            buckets[tag].append(query)

    for tag, bucket_queries in buckets.items():
        path = export_dir / f"{tag}.txt"
        path.write_text("\n".join(bucket_queries) + "\n", encoding="utf-8")

    summary = {
        "query_count": len(queries),
        "bucket_counts": {tag: len(bucket_queries) for tag, bucket_queries in sorted(buckets.items())},
        "bucket_files": {tag: str((export_dir / f"{tag}.txt").resolve()) for tag in sorted(buckets)},
        "records": records,
    }
    export_summary = resolve_path(args.export_summary, default_logs_dir(args.profile) / "query_bucket_summary.json")
    write_json(export_summary, summary)

    print(f"Saved query bucket summary: {export_summary}", flush=True)
    for tag, count in sorted(summary["bucket_counts"].items()):
        print(f"[bucket] {tag}={count}", flush=True)


if __name__ == "__main__":
    main()
