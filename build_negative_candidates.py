import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build intra-class and hard negative candidate ids from cleaned JSONL.")
    parser.add_argument(
        "--input-jsonl",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_sample.jsonl",
        help="Input cleaned JSONL path.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_with_negatives.jsonl",
        help="Output JSONL path with negative candidate ids attached.",
    )
    parser.add_argument(
        "--max-intra-class",
        type=int,
        default=20,
        help="Maximum number of intra-class negative ids to keep per sample.",
    )
    parser.add_argument(
        "--max-hard-negatives",
        type=int,
        default=10,
        help="Maximum number of hard negative ids to keep per sample.",
    )
    return parser.parse_args()


def load_records(jsonl_path):
    records = []
    with Path(jsonl_path).open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def lexical_overlap_score(record_a, record_b):
    tokens_a = set(record_a.get("text", "").split())
    tokens_b = set(record_b.get("text", "").split())
    attrs_a = set(record_a.get("attributes", []))
    attrs_b = set(record_b.get("attributes", []))

    token_overlap = len(tokens_a & tokens_b)
    attr_overlap = len(attrs_a & attrs_b)
    same_category = 1 if record_a.get("category") == record_b.get("category") else 0
    return token_overlap * 2 + attr_overlap + same_category


def build_negative_candidates(records, max_intra_class, max_hard_negatives):
    by_category = defaultdict(list)
    by_group = defaultdict(list)

    for record in records:
        by_category[record.get("category", "other")].append(record)
        by_group[record.get("group_id", "")].append(record)

    output_records = []
    for record in records:
        category = record.get("category", "other")
        record_id = record["id"]

        intra_class_candidates = [
            candidate["id"]
            for candidate in by_category[category]
            if candidate["id"] != record_id
        ][:max_intra_class]

        same_group_candidates = [
            candidate
            for candidate in by_group.get(record.get("group_id", ""), [])
            if candidate["id"] != record_id
        ]

        scored_candidates = []
        for candidate in by_category[category]:
            if candidate["id"] == record_id:
                continue
            score = lexical_overlap_score(record, candidate)
            if score > 0:
                scored_candidates.append((score, candidate["id"]))

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        hard_negative_ids = [candidate_id for _, candidate_id in scored_candidates[:max_hard_negatives]]

        # If same-group neighbors exist, force them into hard negatives first.
        prioritized_same_group = [candidate["id"] for candidate in same_group_candidates]
        merged_hard_negatives = []
        for candidate_id in prioritized_same_group + hard_negative_ids:
            if candidate_id not in merged_hard_negatives:
                merged_hard_negatives.append(candidate_id)
            if len(merged_hard_negatives) >= max_hard_negatives:
                break

        enriched = dict(record)
        enriched["intra_class_negative_ids"] = intra_class_candidates
        enriched["hard_negative_ids"] = merged_hard_negatives
        output_records.append(enriched)

    return output_records


def save_records(records, output_jsonl):
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    records = load_records(args.input_jsonl)
    if not records:
        raise SystemExit(f"No records found in {args.input_jsonl}")

    output_records = build_negative_candidates(
        records,
        max_intra_class=args.max_intra_class,
        max_hard_negatives=args.max_hard_negatives,
    )
    save_records(output_records, args.output_jsonl)

    print(f"Saved negative candidates to: {args.output_jsonl}")
    print(f"Record count: {len(output_records)}")
    print(f"Example record id: {output_records[0]['id']}")
    print(f"  intra_class_negative_ids: {output_records[0]['intra_class_negative_ids'][:5]}")
    print(f"  hard_negative_ids: {output_records[0]['hard_negative_ids'][:5]}")


if __name__ == "__main__":
    main()
