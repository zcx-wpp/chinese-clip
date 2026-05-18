from __future__ import annotations

import argparse
from pathlib import Path

from .api import build_retriever
from .config import PROJECT_ROOT
from .io_utils import read_json, write_json
from .profile_paths import default_metadata_db_path, default_output_dir, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose cases recovered by a larger recall setting.")
    parser.add_argument("--baseline-stage-json", required=True)
    parser.add_argument("--candidate-stage-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument("--profile", help="Named storage profile, e.g. seg4s.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-preset", choices=["current", "baseline"], default="current")
    parser.add_argument("--video-recall-top-k", type=int)
    parser.add_argument("--segment-recall-top-k", type=int)
    parser.add_argument("--video-recall-candidate-pool-size", type=int)
    parser.add_argument("--segment-recall-candidate-pool-size", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--export-json", required=True)
    parser.add_argument("--export-markdown", required=True)
    return parser.parse_args()


def _resolve_output_dir(args) -> Path:
    return resolve_path(args.output_dir, default_output_dir(args.profile))


def _resolve_metadata_db(args) -> Path:
    return resolve_path(args.metadata_db, default_metadata_db_path(args.profile))


def _lookup(payload: dict) -> dict[str, dict]:
    return {item["query"]: item for item in payload.get("results", []) if item.get("query")}


def _rank_of(video_id: str, ranked_video_ids: list[str]) -> int | None:
    for idx, value in enumerate(ranked_video_ids, start=1):
        if value == video_id:
            return idx
    return None


def _score_gap(score: float | None, reference: float | None) -> float | None:
    if score is None or reference is None:
        return None
    return round(float(score) - float(reference), 6)


def _video_frame_stats(records: list[dict], frame_scores: dict[str, float]) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    segment_counts: dict[str, set[str]] = {}
    for item in records:
        video_id = item["video_id"]
        grouped.setdefault(video_id, []).append(float(frame_scores[item["frame_id"]]))
        segment_counts.setdefault(video_id, set()).add(item["segment_id"])

    stats = {}
    for video_id, scores in grouped.items():
        ranked = sorted(scores, reverse=True)
        stats[video_id] = {
            "frame_count": len(scores),
            "segment_count": len(segment_counts.get(video_id, set())),
            "best_frame_score": round(ranked[0], 6),
            "avg_top3_frame_score": round(sum(ranked[:3]) / min(3, len(ranked)), 6),
        }
    return stats


def analyze(args) -> dict:
    baseline = read_json(Path(args.baseline_stage_json))
    candidate = read_json(Path(args.candidate_stage_json))
    baseline_by_query = _lookup(baseline)
    candidate_by_query = _lookup(candidate)

    recovered = []
    for query, base_item in baseline_by_query.items():
        cand_item = candidate_by_query.get(query)
        if not cand_item:
            continue
        if base_item.get("terminal_stage") == "video_recall_miss" and cand_item.get("merged_video_rank") is not None:
            recovered.append(cand_item)

    retriever = build_retriever(
        output_dir=_resolve_output_dir(args),
        metadata_db_path=_resolve_metadata_db(args),
        model_path=args.model_path,
        device=args.device,
        retrieval_preset=args.retrieval_preset,
        video_recall_top_k=args.video_recall_top_k,
        segment_recall_top_k=args.segment_recall_top_k,
        video_recall_candidate_pool_size=args.video_recall_candidate_pool_size,
        segment_recall_candidate_pool_size=args.segment_recall_candidate_pool_size,
    )

    rows = []
    for idx, item in enumerate(recovered, start=1):
        query = item["query"]
        expected_video_id = item.get("expected_video_id")
        query_embeddings = retriever.encoder.encode_texts([query])
        candidate_video_ids = retriever._recall_video_ids(query_embeddings)
        candidate_segment_ids = retriever._recall_segment_ids(query_embeddings, candidate_video_ids)
        records = retriever.metadata_store.get_frame_records_by_segment_ids(candidate_segment_ids)
        video_scores, frame_scores = retriever.rerank_video_scores(query_embeddings, records)
        ranked_videos = sorted(video_scores.items(), key=lambda value: value[1], reverse=True)
        ranked_video_ids = [video_id for video_id, _ in ranked_videos]
        video_stats = _video_frame_stats(records, frame_scores)

        expected_score = video_scores.get(expected_video_id)
        top1_video_id = ranked_video_ids[0] if ranked_video_ids else None
        top1_score = ranked_videos[0][1] if ranked_videos else None
        topk_boundary_score = ranked_videos[args.top_k - 1][1] if len(ranked_videos) >= args.top_k else None
        full_final_rank = _rank_of(expected_video_id, ranked_video_ids) if expected_video_id else None

        top_competitors = []
        for rank, (video_id, score) in enumerate(ranked_videos[: args.top_k], start=1):
            stat = video_stats.get(video_id, {})
            top_competitors.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "score": round(float(score), 6),
                    "score_minus_expected": _score_gap(score, expected_score),
                    **stat,
                }
            )

        row = {
            "query": query,
            "expected_video_id": expected_video_id,
            "stage_final_video_rank": item.get("final_video_rank"),
            "full_final_video_rank": full_final_rank,
            "merged_video_rank": item.get("merged_video_rank"),
            "merged_segment_rank": item.get("merged_segment_rank"),
            "candidate_video_count": len(candidate_video_ids),
            "candidate_segment_count": len(candidate_segment_ids),
            "expected_score": round(float(expected_score), 6) if expected_score is not None else None,
            "top1_video_id": top1_video_id,
            "top1_score": round(float(top1_score), 6) if top1_score is not None else None,
            "top1_score_gap": _score_gap(top1_score, expected_score),
            "topk_boundary_score": round(float(topk_boundary_score), 6) if topk_boundary_score is not None else None,
            "expected_minus_topk_boundary": _score_gap(expected_score, topk_boundary_score),
            "expected_stats": video_stats.get(expected_video_id, {}),
            "top_competitors": top_competitors,
        }
        rows.append(row)
        print(
            f"[diagnose] {idx}/{len(recovered)} rank={full_final_rank or 'MISS'} "
            f"expected_gap_to_top10={row['expected_minus_topk_boundary']} query={query}",
            flush=True,
        )

    rows.sort(
        key=lambda row: (
            row["full_final_video_rank"] if row["full_final_video_rank"] is not None else 999999,
            row["merged_video_rank"] if row["merged_video_rank"] is not None else 999999,
            row["query"],
        )
    )
    return {
        "baseline_stage_json": args.baseline_stage_json,
        "candidate_stage_json": args.candidate_stage_json,
        "recovered_count": len(rows),
        "top10_recovered_count": sum(1 for row in rows if row["full_final_video_rank"] is not None and row["full_final_video_rank"] <= args.top_k),
        "final_miss_count": sum(1 for row in rows if row["full_final_video_rank"] is None or row["full_final_video_rank"] > args.top_k),
        "results": rows,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Recovered Recall Score Diagnostics",
        "",
        "## Summary",
        "",
        f"- recovered_count: {payload['recovered_count']}",
        f"- top10_recovered_count: {payload['top10_recovered_count']}",
        f"- final_miss_count: {payload['final_miss_count']}",
        "",
        "## Cases",
        "",
    ]
    for item in payload["results"]:
        lines.append(f"### {item['query']}")
        lines.append("")
        lines.append(f"- expected: `{item['expected_video_id']}`")
        lines.append(f"- merged video rank: {item['merged_video_rank']}")
        lines.append(f"- merged segment rank: {item['merged_segment_rank']}")
        lines.append(f"- full final rank: {item['full_final_video_rank']}")
        lines.append(f"- expected score: {item['expected_score']}")
        lines.append(f"- top1: `{item['top1_video_id']}` score={item['top1_score']} gap={item['top1_score_gap']}")
        lines.append(f"- expected minus top10 boundary: {item['expected_minus_topk_boundary']}")
        expected_stats = item.get("expected_stats") or {}
        lines.append(
            "- expected evidence: "
            f"segments={expected_stats.get('segment_count')} "
            f"frames={expected_stats.get('frame_count')} "
            f"best_frame={expected_stats.get('best_frame_score')} "
            f"avg_top3={expected_stats.get('avg_top3_frame_score')}"
        )
        lines.append("- top competitors:")
        for competitor in item["top_competitors"][:5]:
            lines.append(
                f"  - rank={competitor['rank']} video=`{competitor['video_id']}` "
                f"score={competitor['score']} gap={competitor['score_minus_expected']} "
                f"segments={competitor.get('segment_count')} frames={competitor.get('frame_count')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    payload = analyze(args)
    export_json = Path(args.export_json)
    write_json(export_json, payload)
    export_md = Path(args.export_markdown)
    export_md.parent.mkdir(parents=True, exist_ok=True)
    export_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved score diagnostics JSON: {export_json}", flush=True)
    print(f"Saved score diagnostics Markdown: {export_md}", flush=True)


if __name__ == "__main__":
    main()
