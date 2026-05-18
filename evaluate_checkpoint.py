import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Chinese-CLIP checkpoint on local JSONL data.")
    parser.add_argument(
        "--jsonl-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_with_negatives.jsonl",
        help="Evaluation JSONL path.",
    )
    parser.add_argument(
        "--validation-dir",
        default="",
        help="Optional directory containing val_in_domain.jsonl / val_clean.jsonl / val_shifted.jsonl.",
    )
    parser.add_argument(
        "--model-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\model",
        help="Model or checkpoint directory.",
    )
    parser.add_argument(
        "--processor-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\model",
        help="Processor/tokenizer directory. Defaults to the original model directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of samples to evaluate for each dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\24703\Desktop\chinese_clip\eval_results",
        help="Directory to save evaluation JSON.",
    )
    return parser.parse_args()


def load_records(jsonl_path, limit):
    records = []
    with Path(jsonl_path).open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = (record.get("text") or record.get("clean_text") or record.get("raw_text") or "").strip()
            image_path = Path(record["image"])
            if not text or not image_path.exists():
                continue
            records.append(
                {
                    "id": record["id"],
                    "image": str(image_path),
                    "text": text,
                    "category": record.get("category", ""),
                    "group_id": record.get("group_id", ""),
                    "split": record.get("split", ""),
                    "difficulty_tags": record.get("difficulty_tags", []),
                }
            )
            if limit > 0 and len(records) >= limit:
                break
    return records


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
    return sum(1 for rank in ranks if rank <= k) / len(ranks) if ranks else 0.0


def ensure_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(features)!r}")


def embed_texts(model, processor, texts, batch_size, device, dataset_name="dataset"):
    outputs = []
    total_batches = math.ceil(len(texts) / batch_size)
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        print(
            f"[{dataset_name}] Embedding texts batch {batch_idx}/{total_batches} "
            f"({len(batch_texts)} samples)"
        )
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = ensure_feature_tensor(features)
            features = F.normalize(features, dim=-1)
        outputs.extend(features.cpu().tolist())
    return outputs


def embed_images(model, processor, image_paths, batch_size, device, dataset_name="dataset"):
    outputs = []
    total_batches = math.ceil(len(image_paths) / batch_size)
    for batch_idx, start in enumerate(range(0, len(image_paths), batch_size), start=1):
        batch_paths = image_paths[start : start + batch_size]
        print(
            f"[{dataset_name}] Embedding images batch {batch_idx}/{total_batches} "
            f"({len(batch_paths)} samples)"
        )
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = ensure_feature_tensor(features)
            features = F.normalize(features, dim=-1)
        outputs.extend(features.cpu().tolist())
    return outputs


def build_summary(per_sample):
    text_to_image_ranks = [item["text_to_image_rank"] for item in per_sample]
    image_to_text_ranks = [item["image_to_text_rank"] for item in per_sample]
    return {
        "count": len(per_sample),
        "text_to_image_recall_at_1": recall_at_k(text_to_image_ranks, 1),
        "text_to_image_recall_at_5": recall_at_k(text_to_image_ranks, 5),
        "text_to_image_recall_at_10": recall_at_k(text_to_image_ranks, 10),
        "image_to_text_recall_at_1": recall_at_k(image_to_text_ranks, 1),
        "image_to_text_recall_at_5": recall_at_k(image_to_text_ranks, 5),
        "image_to_text_recall_at_10": recall_at_k(image_to_text_ranks, 10),
        "text_to_image_mean_rank": sum(text_to_image_ranks) / len(text_to_image_ranks),
        "image_to_text_mean_rank": sum(image_to_text_ranks) / len(image_to_text_ranks),
        "average_matched_similarity": sum(item["matched_score"] for item in per_sample) / len(per_sample),
    }


def summarize_by_tag(per_sample):
    by_tag = defaultdict(list)
    for item in per_sample:
        tags = item.get("difficulty_tags", [])
        for tag in tags:
            by_tag[tag].append(item)

    return {
        tag: build_summary(items)
        for tag, items in sorted(by_tag.items())
        if items
    }


def evaluate_records(records, model, processor, batch_size, device, dataset_name):
    texts = [record["text"] for record in records]
    image_paths = [record["image"] for record in records]

    print(f"[{dataset_name}] Embedding texts...")
    text_embeddings = embed_texts(model, processor, texts, batch_size, device, dataset_name=dataset_name)
    print(f"[{dataset_name}] Embedding images...")
    image_embeddings = embed_images(model, processor, image_paths, batch_size, device, dataset_name=dataset_name)

    per_sample = []
    for idx, record in enumerate(records):
        text_scores = [
            cosine_similarity(text_embeddings[idx], image_embedding)
            for image_embedding in image_embeddings
        ]
        image_scores = [
            cosine_similarity(image_embeddings[idx], text_embedding)
            for text_embedding in text_embeddings
        ]
        matched_score = text_scores[idx]
        text_rank = rank_of_correct_item(text_scores, idx)
        image_rank = rank_of_correct_item(image_scores, idx)
        per_sample.append(
            {
                "id": record["id"],
                "text": record["text"],
                "category": record["category"],
                "group_id": record["group_id"],
                "split": record.get("split", ""),
                "difficulty_tags": record.get("difficulty_tags", []),
                "matched_score": matched_score,
                "text_to_image_rank": text_rank,
                "image_to_text_rank": image_rank,
            }
        )

    return {
        "summary": build_summary(per_sample),
        "difficulty_tag_summary": summarize_by_tag(per_sample),
        "records": per_sample,
    }


def print_summary_block(name, summary):
    print(name)
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
    print(f"Average matched similarity: {summary['average_matched_similarity']:.4f}")


def collect_datasets(args):
    if args.validation_dir:
        validation_dir = Path(args.validation_dir)
        dataset_map = {}
        for split_name, filename in [
            ("in_domain", "val_in_domain.jsonl"),
            ("clean", "val_clean.jsonl"),
            ("shifted", "val_shifted.jsonl"),
        ]:
            path = validation_dir / filename
            if path.exists():
                dataset_map[split_name] = path
        if not dataset_map:
            raise SystemExit(f"No validation JSONL files found in {validation_dir}")
        return dataset_map

    return {"default": Path(args.jsonl_path)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_map = collect_datasets(args)

    print(f"Loading model from {args.model_path}")
    model = ChineseCLIPModel.from_pretrained(args.model_path, local_files_only=True).to(device)
    resolved_processor_path = args.model_path
    try:
        processor = ChineseCLIPProcessor.from_pretrained(resolved_processor_path, local_files_only=True)
        print(f"Loading processor from {resolved_processor_path}")
    except Exception:
        resolved_processor_path = args.processor_path
        print(f"Loading processor from {resolved_processor_path}")
        processor = ChineseCLIPProcessor.from_pretrained(resolved_processor_path, local_files_only=True)
    model.eval()

    dataset_results = {}
    for dataset_name, dataset_path in dataset_map.items():
        records = load_records(dataset_path, args.limit)
        if not records:
            raise SystemExit(f"No valid records found in {dataset_path}")
        print(f"Loaded {len(records)} records from {dataset_path}")
        dataset_results[dataset_name] = {
            "jsonl_path": str(Path(dataset_path).resolve()),
            **evaluate_records(records, model, processor, args.batch_size, device, dataset_name),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "multi" if len(dataset_results) > 1 else str(next(iter(dataset_results.values()))["summary"]["count"])
    output_path = output_dir / f"checkpoint_eval_{suffix}_{timestamp}.json"
    payload = {
        "run_info": {
            "timestamp": timestamp,
            "model_path": str(Path(args.model_path).resolve()),
            "processor_path": str(Path(resolved_processor_path).resolve()),
            "limit": args.limit,
            "batch_size": args.batch_size,
            "device": str(device),
            "validation_dir": str(Path(args.validation_dir).resolve()) if args.validation_dir else "",
        },
        "datasets": dataset_results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for dataset_name, result in dataset_results.items():
        print_summary_block(f"Summary [{dataset_name}]", result["summary"])
        if result["difficulty_tag_summary"]:
            print(f"Difficulty tags [{dataset_name}]")
            for tag, summary in result["difficulty_tag_summary"].items():
                print(
                    f"  {tag}: count={summary['count']} "
                    f"T2I_R@1={summary['text_to_image_recall_at_1']:.4f} "
                    f"I2T_R@1={summary['image_to_text_recall_at_1']:.4f} "
                    f"MeanRank={summary['text_to_image_mean_rank']:.2f}/{summary['image_to_text_mean_rank']:.2f}"
                )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
