import argparse
import base64
import json
import math
import random
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate local MUGE samples with the embedding API.")
    parser.add_argument(
        "--sample-dir",
        default=r"C:\Users\24703\Desktop\chinese_clip\muge_sample",
        help="Directory containing paired .jpg and .txt files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="Base URL of the embedding service.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of paired samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mismatched pair sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\24703\Desktop\chinese_clip\eval_results",
        help="Directory used to save evaluation results.",
    )
    parser.add_argument(
        "--jsonl-path",
        default="",
        help="Optional cleaned JSONL file. If set, evaluation reads pairs from JSONL instead of sample-dir.",
    )
    return parser.parse_args()


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_of_correct_item(scores, correct_index):
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return ranked_indices.index(correct_index) + 1


def recall_at_k(ranks, k):
    hits = sum(1 for rank in ranks if rank <= k)
    return hits / len(ranks) if ranks else 0.0


def load_pairs(sample_dir, limit):
    txt_files = sorted(Path(sample_dir).glob("*.txt"))
    pairs = []
    for txt_path in txt_files:
        stem = txt_path.stem
        image_path = txt_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        pairs.append((stem, image_path, text))
        if len(pairs) >= limit:
            break
    return pairs


def load_pairs_from_jsonl(jsonl_path, limit):
    pairs = []
    with Path(jsonl_path).open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stem = record["id"]
            image_path = Path(record["image"])
            text = (record.get("clean_text") or record.get("raw_text") or "").strip()
            if not image_path.exists() or not text:
                continue
            pairs.append((stem, image_path, text))
            if len(pairs) >= limit:
                break
    return pairs


def build_mismatched_indices(count, seed):
    random.seed(seed)
    if count < 2:
        return list(range(count))

    indices = list(range(count))
    mismatched = indices[:]
    while True:
        random.shuffle(mismatched)
        if all(i != j for i, j in zip(indices, mismatched)):
            return mismatched


def main():
    args = parse_args()
    embed_url = args.base_url.rstrip("/") + "/embed"
    if args.jsonl_path:
        pairs = load_pairs_from_jsonl(args.jsonl_path, args.limit)
        data_source = str(Path(args.jsonl_path).resolve())
    else:
        pairs = load_pairs(args.sample_dir, args.limit)
        data_source = str(Path(args.sample_dir).resolve())

    if not pairs:
        raise SystemExit(f"No valid evaluation pairs found in {data_source}")

    print(f"Evaluating {len(pairs)} MUGE pairs from {data_source}")
    print(f"Embedding API: {embed_url}")

    records = []
    for stem, image_path, text in pairs:
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

        text_resp = post_json(embed_url, {"datatype": "text", "input": text})
        image_resp = post_json(embed_url, {"datatype": "image", "input": image_base64})

        text_embedding = text_resp["embedding"]
        image_embedding = image_resp["embedding"]
        matched_score = cosine_similarity(text_embedding, image_embedding)

        records.append(
            {
                "stem": stem,
                "text": text,
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "text_time_cost": text_resp.get("time_cost"),
                "image_time_cost": image_resp.get("time_cost"),
                "matched_score": matched_score,
            }
        )
        print(
            f"[{stem}] matched={matched_score:.4f} "
            f"text_time={text_resp.get('time_cost')}s image_time={image_resp.get('time_cost')}s"
        )

    mismatched_indices = build_mismatched_indices(len(records), args.seed)
    mismatched_scores = []
    for idx, wrong_idx in enumerate(mismatched_indices):
        wrong_image_embedding = records[wrong_idx]["image_embedding"]
        score = cosine_similarity(records[idx]["text_embedding"], wrong_image_embedding)
        records[idx]["mismatched_stem"] = records[wrong_idx]["stem"]
        records[idx]["mismatched_score"] = score
        mismatched_scores.append(score)

    matched_scores = [record["matched_score"] for record in records]
    text_to_image_ranks = []
    image_to_text_ranks = []

    for idx, record in enumerate(records):
        text_scores = [
            cosine_similarity(record["text_embedding"], candidate["image_embedding"])
            for candidate in records
        ]
        image_scores = [
            cosine_similarity(record["image_embedding"], candidate["text_embedding"])
            for candidate in records
        ]

        text_rank = rank_of_correct_item(text_scores, idx)
        image_rank = rank_of_correct_item(image_scores, idx)

        record["text_to_image_rank"] = text_rank
        record["image_to_text_rank"] = image_rank
        text_to_image_ranks.append(text_rank)
        image_to_text_ranks.append(image_rank)

    summary = {
        "average_matched_similarity": sum(matched_scores) / len(matched_scores),
        "average_mismatched_similarity": sum(mismatched_scores) / len(mismatched_scores),
        "text_to_image_recall_at_1": recall_at_k(text_to_image_ranks, 1),
        "text_to_image_recall_at_5": recall_at_k(text_to_image_ranks, 5),
        "text_to_image_recall_at_10": recall_at_k(text_to_image_ranks, 10),
        "image_to_text_recall_at_1": recall_at_k(image_to_text_ranks, 1),
        "image_to_text_recall_at_5": recall_at_k(image_to_text_ranks, 5),
        "image_to_text_recall_at_10": recall_at_k(image_to_text_ranks, 10),
        "text_to_image_mean_rank": sum(text_to_image_ranks) / len(text_to_image_ranks),
        "image_to_text_mean_rank": sum(image_to_text_ranks) / len(image_to_text_ranks),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"muge_eval_{len(records)}_{timestamp}.json"

    result_payload = {
        "run_info": {
            "timestamp": timestamp,
            "sample_dir": str(Path(args.sample_dir).resolve()),
            "jsonl_path": str(Path(args.jsonl_path).resolve()) if args.jsonl_path else "",
            "data_source": data_source,
            "base_url": args.base_url,
            "limit": args.limit,
            "seed": args.seed,
            "sample_count": len(records),
        },
        "summary": summary,
        "records": records,
    }
    output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("Summary")
    print(f"Average matched similarity:    {summary['average_matched_similarity']:.4f}")
    print(f"Average mismatched similarity: {summary['average_mismatched_similarity']:.4f}")
    print("")
    print("Retrieval Metrics")
    print(
        f"Text->Image Recall@1/5/10: "
        f"{summary['text_to_image_recall_at_1']:.4f} / "
        f"{summary['text_to_image_recall_at_5']:.4f} / "
        f"{summary['text_to_image_recall_at_10']:.4f}"
    )
    print(
        f"Image->Text Recall@1/5/10: "
        f"{summary['image_to_text_recall_at_1']:.4f} / "
        f"{summary['image_to_text_recall_at_5']:.4f} / "
        f"{summary['image_to_text_recall_at_10']:.4f}"
    )
    print(
        f"Mean rank (Text->Image / Image->Text): "
        f"{summary['text_to_image_mean_rank']:.2f} / "
        f"{summary['image_to_text_mean_rank']:.2f}"
    )
    print(f"Saved results to: {output_path}")
    print("")
    print("Per-sample comparison")
    for record in records:
        print(
            f"{record['stem']}: matched={record['matched_score']:.4f}, "
            f"mismatched({record['mismatched_stem']})={record['mismatched_score']:.4f}, "
            f"t2i_rank={record['text_to_image_rank']}, "
            f"i2t_rank={record['image_to_text_rank']}, "
            f"text={record['text']}"
        )


if __name__ == "__main__":
    main()
