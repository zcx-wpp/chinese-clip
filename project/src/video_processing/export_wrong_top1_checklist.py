from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

from .config import PROJECT_ROOT
from .io_utils import read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Export a checklist for frequent wrong-top1 videos.")
    parser.add_argument(
        "--summary-json",
        default=str(PROJECT_ROOT / "output" / "logs" / "retrieval_gap_summary.json"),
    )
    parser.add_argument(
        "--metadata-db",
        default=str(PROJECT_ROOT / "metadata" / "metadata.db"),
    )
    parser.add_argument(
        "--export-markdown",
        default=str(PROJECT_ROOT / "output" / "logs" / "wrong_top1_checklist.md"),
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def load_representative_segments(db_path: Path, video_id: str) -> list[dict]:
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT segment_id, start_time, end_time, importance_score, motion_score,
                   visual_diversity_score, representative_rank, path
            FROM segments
            WHERE video_id = ? AND COALESCE(is_representative, 0) = 1
            ORDER BY representative_rank, start_time
            """,
            (video_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def render_checklist(summary: dict, db_path: Path, top_n: int) -> str:
    lines: list[str] = []
    lines.append("# Wrong Top1 Checklist")
    lines.append("")
    lines.append("用于人工检查高频误召回视频的 representative segments 是否过于通用。")
    lines.append("")

    for item in summary.get("most_common_wrong_top1_videos", [])[:top_n]:
        video_id = item["video_id"]
        count = item["count"]
        segments = load_representative_segments(db_path, video_id)
        lines.append(f"## {video_id}")
        lines.append("")
        lines.append(f"- Wrong Top1 count: {count}")
        lines.append("- 检查点:")
        lines.append("  - 是否都是单人近景/室内通用画面")
        lines.append("  - 是否 motion_score 偏高但语义不够具体")
        lines.append("  - 是否 Top-N representative segments 过于相似")
        lines.append("")
        if not segments:
            lines.append("- representative segments: 无")
            lines.append("")
            continue
        lines.append("| rank | start | end | importance | motion | diversity | segment_id |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for segment in segments:
            lines.append(
                f"| {segment.get('representative_rank') or ''} | "
                f"{segment.get('start_time')} | {segment.get('end_time')} | "
                f"{round(float(segment.get('importance_score') or 0.0), 4)} | "
                f"{round(float(segment.get('motion_score') or 0.0), 4)} | "
                f"{round(float(segment.get('visual_diversity_score') or 0.0), 4)} | "
                f"{segment.get('segment_id')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    summary = read_json(Path(args.summary_json))
    markdown = render_checklist(summary, Path(args.metadata_db), args.top_n)
    export_path = Path(args.export_markdown)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(markdown, encoding="utf-8")
    print(f"Saved wrong-top1 checklist: {export_path}", flush=True)


if __name__ == "__main__":
    main()
