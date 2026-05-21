from __future__ import annotations

import argparse
import json
from pathlib import Path

from .caption_records import is_structured_record, normalize_caption_record
from .config import DEFAULT_CAPTIONS_JSONL


def parse_args():
    parser = argparse.ArgumentParser(description="Validate whether a captions JSONL file uses the structured Doubao schema.")
    parser.add_argument("--captions-jsonl", default=str(DEFAULT_CAPTIONS_JSONL))
    parser.add_argument(
        "--fail-on-legacy",
        action="store_true",
        help="Exit with a non-zero status when any legacy caption-only rows are found.",
    )
    parser.add_argument("--show-examples", type=int, default=5, help="How many legacy example video_ids to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.captions_jsonl)
    if not path.exists():
        raise SystemExit(f"Captions file not found: {path}")

    total = 0
    structured = 0
    legacy = 0
    failed = 0
    missing_description = 0
    missing_tags = 0
    legacy_examples: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total += 1
        payload = json.loads(line)
        record = normalize_caption_record(payload)
        if record is None:
            failed += 1
            continue
        if record.status != "ok":
            failed += 1
            continue
        if not record.description.strip():
            missing_description += 1
        if not record.tags:
            missing_tags += 1
        if is_structured_record(record):
            structured += 1
            continue
        legacy += 1
        if len(legacy_examples) < max(0, args.show_examples):
            legacy_examples.append(record.video_id)

    print(f"captions_jsonl={path}")
    print(f"total={total}")
    print(f"structured={structured}")
    print(f"legacy_caption_only={legacy}")
    print(f"failed_or_unreadable={failed}")
    print(f"missing_description={missing_description}")
    print(f"missing_tags={missing_tags}")
    if legacy_examples:
        print("legacy_examples=" + ", ".join(legacy_examples))

    if args.fail_on_legacy and legacy > 0:
        raise SystemExit(
            "Legacy caption-only rows detected. Regenerate captions with doubao_batch_caption.py to produce tags + description."
        )


if __name__ == "__main__":
    main()
