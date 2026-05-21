from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ARTIFACT_ROOT, WORKSPACE_ROOT
from .io_utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a batch-retrieval query text file and matching labels JSON from metadata_zh_only.jsonl."
    )
    parser.add_argument("--metadata-jsonl", default=str(WORKSPACE_ROOT / "data" / "metadata_zh_only.jsonl"))
    parser.add_argument("--queries-txt", default=str(ARTIFACT_ROOT / "retrieval_queries_zh_3000.txt"))
    parser.add_argument("--labels-json", default=str(ARTIFACT_ROOT / "retrieval_labels_zh_3000.json"))
    parser.add_argument(
        "--caption-index",
        type=int,
        default=0,
        help="Preferred caption index to use per video. Falls back to the first non-empty caption when unavailable.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def choose_caption(captions: list[str], preferred_index: int) -> tuple[str, int]:
    cleaned = [(idx, str(item).strip()) for idx, item in enumerate(captions or []) if str(item).strip()]
    if not cleaned:
        raise ValueError("No non-empty Chinese captions found for this video.")
    if 0 <= preferred_index < len(captions):
        preferred = str(captions[preferred_index]).strip()
        if preferred:
            return preferred, preferred_index
    return cleaned[0][1], cleaned[0][0]


def build_outputs(rows: list[dict], preferred_index: int) -> tuple[list[str], list[dict]]:
    queries: list[str] = []
    labels: list[dict] = []
    for query_index, row in enumerate(rows):
        video_id = str(row["videoID"]).strip()
        filename = str(row.get("filename") or f"{video_id}.mp4").strip()
        split = str(row.get("split") or "").strip()
        caption, used_index = choose_caption(row.get("chCap") or [], preferred_index)
        queries.append(caption)
        labels.append(
            {
                "query_index": query_index,
                "query": caption,
                "video_id": video_id,
                "filename": filename,
                "split": split,
                "source_caption_index": used_index,
                "segments": [{"start": 0.0, "end": 10.0}],
            }
        )
    return queries, labels


def write_queries_txt(path: Path, queries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(queries) + ("\n" if queries else "")
    path.write_text(payload, encoding="utf-8")


def main():
    args = parse_args()
    metadata_path = Path(args.metadata_jsonl)
    rows = read_jsonl(metadata_path)
    queries, labels = build_outputs(rows, args.caption_index)
    write_queries_txt(Path(args.queries_txt), queries)
    write_json(Path(args.labels_json), labels)
    print(
        f"[done] metadata={metadata_path} queries={len(queries)} "
        f"queries_txt={args.queries_txt} labels_json={args.labels_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
