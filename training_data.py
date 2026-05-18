import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import ChineseCLIPProcessor


class MugeJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.jsonl_path = Path(jsonl_path)
        self.records = []
        self.id_to_index = {}
        self.category_to_indices = defaultdict(list)
        self.group_to_indices = defaultdict(list)
        with self.jsonl_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = len(self.records)
                self.records.append(record)
                self.id_to_index[record["id"]] = idx
                self.category_to_indices[record.get("category", "")].append(idx)
                self.group_to_indices[record.get("group_id", "")].append(idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = Path(record["image"])
        image = Image.open(image_path).convert("RGB")
        return {
            "id": record["id"],
            "image": image,
            "text": record["text"],
            "category": record.get("category", ""),
            "group_id": record.get("group_id", ""),
            "attributes": record.get("attributes", []),
            "hard_negative_ids": record.get("hard_negative_ids", []),
            "intra_class_negative_ids": record.get("intra_class_negative_ids", []),
            "image_path": str(image_path),
        }


class HardNegativeBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        used = set()
        for anchor_idx in indices:
            if anchor_idx in used:
                continue

            batch = [anchor_idx]
            used.add(anchor_idx)
            record = self.dataset.records[anchor_idx]

            candidate_indices = []
            for candidate_id in record.get("hard_negative_ids", []):
                candidate_idx = self.dataset.id_to_index.get(candidate_id)
                if candidate_idx is not None and candidate_idx != anchor_idx:
                    candidate_indices.append(candidate_idx)

            for candidate_id in record.get("intra_class_negative_ids", []):
                candidate_idx = self.dataset.id_to_index.get(candidate_id)
                if candidate_idx is not None and candidate_idx != anchor_idx:
                    candidate_indices.append(candidate_idx)

            group_id = record.get("group_id", "")
            if group_id:
                candidate_indices.extend(self.dataset.group_to_indices.get(group_id, []))

            category = record.get("category", "")
            if category:
                candidate_indices.extend(self.dataset.category_to_indices.get(category, []))

            seen_candidates = set()
            deduped_candidates = []
            for candidate_idx in candidate_indices:
                if candidate_idx == anchor_idx or candidate_idx in seen_candidates:
                    continue
                seen_candidates.add(candidate_idx)
                deduped_candidates.append(candidate_idx)

            for candidate_idx in deduped_candidates:
                if len(batch) >= self.batch_size:
                    break
                if candidate_idx in used:
                    continue
                batch.append(candidate_idx)
                used.add(candidate_idx)

            if len(batch) < self.batch_size:
                for candidate_idx in indices:
                    if len(batch) >= self.batch_size:
                        break
                    if candidate_idx in used:
                        continue
                    batch.append(candidate_idx)
                    used.add(candidate_idx)

            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ChineseClipCollator:
    def __init__(self, model_path):
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path, local_files_only=True)

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]

        model_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        model_inputs["ids"] = [item["id"] for item in batch]
        model_inputs["texts"] = texts
        model_inputs["categories"] = [item["category"] for item in batch]
        model_inputs["group_ids"] = [item["group_id"] for item in batch]
        model_inputs["attributes"] = [item["attributes"] for item in batch]
        model_inputs["hard_negative_ids"] = [item["hard_negative_ids"] for item in batch]
        model_inputs["intra_class_negative_ids"] = [
            item["intra_class_negative_ids"] for item in batch
        ]
        model_inputs["image_paths"] = [item["image_path"] for item in batch]
        return model_inputs


def build_dataloader(
    jsonl_path,
    model_path,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    batch_strategy="random",
):
    dataset = MugeJsonlDataset(jsonl_path)
    collator = ChineseClipCollator(model_path=model_path)
    if batch_strategy == "hard_negative":
        batch_sampler = HardNegativeBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collator,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a Chinese-CLIP JSONL dataset batch.")
    parser.add_argument(
        "--jsonl-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_with_negatives.jsonl",
        help="Training JSONL path.",
    )
    parser.add_argument(
        "--model-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\model",
        help="Local Chinese-CLIP model directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Preview batch size.")
    parser.add_argument(
        "--batch-strategy",
        choices=["random", "hard_negative"],
        default="random",
        help="How to organize samples inside each batch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataloader = build_dataloader(
        jsonl_path=args.jsonl_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        batch_strategy=args.batch_strategy,
    )
    batch = next(iter(dataloader))

    print("Batch keys:", sorted(batch.keys()))
    print("Batch size:", len(batch["ids"]))
    print("Input IDs shape:", tuple(batch["input_ids"].shape))
    print("Pixel values shape:", tuple(batch["pixel_values"].shape))
    print("Example ids:", batch["ids"])
    print("Example categories:", batch["categories"])
    print("Example hard negatives:", batch["hard_negative_ids"][0][:5])


if __name__ == "__main__":
    main()
