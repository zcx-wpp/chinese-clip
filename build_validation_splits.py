import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


COLOR_KEYWORDS = [
    "红", "黑", "白", "蓝", "绿", "黄", "紫", "粉", "灰", "棕", "橙", "金", "银",
]

STYLE_KEYWORDS = [
    "复古", "中式", "北欧", "简约", "可爱", "气质", "高跟", "厚底", "长款", "短款",
    "圆领", "v领", "宽松", "修身", "休闲", "时尚", "百搭", "小香风", "洛丽塔",
]

ACCESSORY_KEYWORDS = [
    "项链", "耳环", "耳钉", "戒指", "发簪", "手链", "挂件", "手机壳", "鞋垫", "杯垫",
    "贴纸", "挂钩", "保护套", "帽子", "袜子", "腰带",
]

MAIN_OBJECT_KEYWORDS = [
    "鞋", "裙", "裤", "包", "杯", "壶", "桌", "椅", "衣", "外套", "茶具", "手机",
]

BRANDISH_KEYWORDS = [
    "旗舰店", "官方", "正品", "品牌", "专柜",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build fixed validation splits with heuristic difficulty tags."
    )
    root_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-jsonl",
        default=str(root_dir / "cleaned_muge_with_negatives.jsonl"),
        help="Input cleaned JSONL with negative candidates.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "validation_sets"),
        help="Directory to save validation split JSONL files.",
    )
    parser.add_argument(
        "--in-domain-size",
        type=int,
        default=300,
        help="Target sample count for val_in_domain.jsonl.",
    )
    parser.add_argument(
        "--clean-size",
        type=int,
        default=300,
        help="Target sample count for val_clean.jsonl.",
    )
    parser.add_argument(
        "--shifted-size",
        type=int,
        default=300,
        help="Target sample count for val_shifted.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_records(input_jsonl):
    records = []
    with Path(input_jsonl).open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = Path(record.get("image", ""))
            text = (record.get("text") or record.get("clean_text") or "").strip()
            if not text or not image_path.exists():
                continue
            records.append(record)
    return records


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def ascii_ratio(text):
    if not text:
        return 0.0
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return ascii_count / len(text)


def looks_brand_dominant(record):
    text = record.get("text", "")
    if contains_any(text, BRANDISH_KEYWORDS):
        return True
    if " " not in text and len(text) <= 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9._ -]+", text):
        return True
    return False


def infer_difficulty_tags(record, category_counts):
    text = record.get("text", "")
    tags = []
    category = record.get("category", "")
    hard_negative_ids = record.get("hard_negative_ids", [])

    if category and category_counts.get(category, 0) >= 20 and hard_negative_ids:
        tags.append("same_category_hard")
    if contains_any(text, COLOR_KEYWORDS):
        tags.append("color_sensitive")
    if contains_any(text, STYLE_KEYWORDS):
        tags.append("style_sensitive")
    if contains_any(text, ACCESSORY_KEYWORDS) and contains_any(text, MAIN_OBJECT_KEYWORDS):
        tags.append("accessory_vs_main")
    if looks_brand_dominant(record):
        tags.append("brand_dominant")
    return tags


def clean_score(record):
    text = record.get("text", "")
    score = 0
    if len(text) >= 6:
        score += 2
    if record.get("category") and record.get("category") != "other":
        score += 2
    if record.get("attributes"):
        score += min(2, len(record.get("attributes", [])))
    if ascii_ratio(text) < 0.35:
        score += 1
    if not looks_brand_dominant(record):
        score += 2
    if min(record.get("image_width", 0), record.get("image_height", 0)) >= 224:
        score += 1
    return score


def shifted_score(record):
    text = record.get("text", "")
    score = 0
    if looks_brand_dominant(record):
        score += 3
    if ascii_ratio(text) >= 0.35:
        score += 2
    if len(text) <= 4:
        score += 2
    if record.get("category") == "other":
        score += 1
    if not record.get("attributes"):
        score += 1
    return score


def choose_split(records, used_ids, target_size, score_fn, split_name, rng, category_cap=40):
    candidates = [record for record in records if record["id"] not in used_ids]
    scored = []
    for record in candidates:
        score = score_fn(record)
        scored.append((score, rng.random(), record))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["id"]))

    picked = []
    category_counts = Counter()
    for score, _, record in scored:
        if len(picked) >= target_size:
            break
        category = record.get("category", "other")
        if category_counts[category] >= category_cap:
            continue
        enriched = dict(record)
        enriched["split"] = split_name
        picked.append(enriched)
        used_ids.add(record["id"])
        category_counts[category] += 1

    if len(picked) < target_size:
        for _, _, record in scored:
            if len(picked) >= target_size:
                break
            if record["id"] in {item["id"] for item in picked}:
                continue
            enriched = dict(record)
            enriched["split"] = split_name
            picked.append(enriched)
            used_ids.add(record["id"])

    return picked


def choose_in_domain(records, used_ids, target_size, rng, category_cap=40):
    candidates = [record for record in records if record["id"] not in used_ids]
    rng.shuffle(candidates)
    picked = []
    category_counts = Counter()
    for record in candidates:
        if len(picked) >= target_size:
            break
        category = record.get("category", "other")
        if category_counts[category] >= category_cap:
            continue
        enriched = dict(record)
        enriched["split"] = "in_domain"
        picked.append(enriched)
        used_ids.add(record["id"])
        category_counts[category] += 1

    if len(picked) < target_size:
        for record in candidates:
            if len(picked) >= target_size:
                break
            if record["id"] in {item["id"] for item in picked}:
                continue
            enriched = dict(record)
            enriched["split"] = "in_domain"
            picked.append(enriched)
            used_ids.add(record["id"])
    return picked


def attach_tags(records, category_counts):
    enriched = []
    for record in records:
        item = dict(record)
        item["difficulty_tags"] = infer_difficulty_tags(record, category_counts)
        enriched.append(item)
    return enriched


def save_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records):
    categories = Counter(record.get("category", "other") for record in records)
    tag_counts = Counter(tag for record in records for tag in record.get("difficulty_tags", []))
    return categories, tag_counts


def print_summary(name, records):
    categories, tag_counts = summarize(records)
    print(f"{name}: {len(records)} records")
    print(f"  Top categories: {categories.most_common(5)}")
    print(f"  Difficulty tags: {tag_counts.most_common(5)}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    records = load_records(args.input_jsonl)
    if not records:
        raise SystemExit(f"No valid records found in {args.input_jsonl}")

    all_category_counts = Counter(record.get("category", "other") for record in records)
    used_ids = set()

    clean_records = choose_split(
        records=records,
        used_ids=used_ids,
        target_size=args.clean_size,
        score_fn=clean_score,
        split_name="clean",
        rng=rng,
    )
    shifted_records = choose_split(
        records=records,
        used_ids=used_ids,
        target_size=args.shifted_size,
        score_fn=shifted_score,
        split_name="shifted",
        rng=rng,
    )
    in_domain_records = choose_in_domain(
        records=records,
        used_ids=used_ids,
        target_size=args.in_domain_size,
        rng=rng,
    )

    clean_records = attach_tags(clean_records, all_category_counts)
    shifted_records = attach_tags(shifted_records, all_category_counts)
    in_domain_records = attach_tags(in_domain_records, all_category_counts)

    output_dir = Path(args.output_dir)
    save_jsonl(in_domain_records, output_dir / "val_in_domain.jsonl")
    save_jsonl(clean_records, output_dir / "val_clean.jsonl")
    save_jsonl(shifted_records, output_dir / "val_shifted.jsonl")

    print(f"Saved validation splits to: {output_dir}")
    print_summary("val_in_domain", in_domain_records)
    print_summary("val_clean", clean_records)
    print_summary("val_shifted", shifted_records)


if __name__ == "__main__":
    main()
