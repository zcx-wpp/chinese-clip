from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT
from .io_utils import read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two bucket hit evaluation files.")
    parser.add_argument("--baseline", required=True, help="Older bucket_hit_eval.json")
    parser.add_argument("--candidate", required=True, help="Newer bucket_hit_eval.json")
    parser.add_argument(
        "--export-markdown",
        default=str(PROJECT_ROOT / "output" / "logs" / "bucket_hit_compare.md"),
    )
    parser.add_argument(
        "--export-json",
        default=str(PROJECT_ROOT / "output" / "logs" / "bucket_hit_compare.json"),
    )
    return parser.parse_args()


def to_lookup(payload: dict) -> dict[str, dict]:
    return {item["bucket"]: item for item in payload.get("bucket_results", [])}


def compare(baseline: dict, candidate: dict) -> dict:
    baseline_lookup = to_lookup(baseline)
    candidate_lookup = to_lookup(candidate)
    buckets = sorted(set(baseline_lookup) | set(candidate_lookup))

    rows = []
    for bucket in buckets:
        base = baseline_lookup.get(bucket, {})
        cand = candidate_lookup.get(bucket, {})
        base_top1 = float(base.get("top1_hit_rate", 0.0))
        cand_top1 = float(cand.get("top1_hit_rate", 0.0))
        base_topk = float(base.get("topk_hit_rate", 0.0))
        cand_topk = float(cand.get("topk_hit_rate", 0.0))
        rows.append(
            {
                "bucket": bucket,
                "baseline_count": int(base.get("count", 0)),
                "candidate_count": int(cand.get("count", 0)),
                "baseline_top1": base_top1,
                "candidate_top1": cand_top1,
                "delta_top1": round(cand_top1 - base_top1, 4),
                "baseline_topk": base_topk,
                "candidate_topk": cand_topk,
                "delta_topk": round(cand_topk - base_topk, 4),
            }
        )

    rows.sort(key=lambda item: (item["delta_top1"], item["delta_topk"]), reverse=True)
    return {
        "baseline_query_count": baseline.get("query_count", 0),
        "candidate_query_count": candidate.get("query_count", 0),
        "bucket_deltas": rows,
    }


def render_markdown(payload: dict) -> str:
    lines = []
    lines.append("# Bucket Hit Compare")
    lines.append("")
    lines.append(f"- Baseline query count: {payload['baseline_query_count']}")
    lines.append(f"- Candidate query count: {payload['candidate_query_count']}")
    lines.append("")
    lines.append("| Bucket | Base Top1 | New Top1 | Delta Top1 | Base TopK | New TopK | Delta TopK |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in payload["bucket_deltas"]:
        lines.append(
            f"| {item['bucket']} | {item['baseline_top1']} | {item['candidate_top1']} | {item['delta_top1']} | "
            f"{item['baseline_topk']} | {item['candidate_topk']} | {item['delta_topk']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    baseline = read_json(Path(args.baseline))
    candidate = read_json(Path(args.candidate))
    payload = compare(baseline, candidate)

    export_json = Path(args.export_json)
    write_json(export_json, payload)
    export_markdown = Path(args.export_markdown)
    export_markdown.parent.mkdir(parents=True, exist_ok=True)
    export_markdown.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Saved bucket compare JSON: {export_json}", flush=True)
    print(f"Saved bucket compare Markdown: {export_markdown}", flush=True)


if __name__ == "__main__":
    main()
