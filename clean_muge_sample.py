import argparse
import json
import re
from pathlib import Path

from PIL import Image


NOISE_TERMS = [
    "\u5b98\u65b9\u65d7\u8230\u5e97",
    "\u65d7\u8230\u5e97",
    "\u5b98\u65b9\u5e97",
    "\u6b63\u54c1",
    "\u5305\u90ae",
    "\u7206\u6b3e",
    "\u70ed\u5356",
    "\u7279\u4ef7",
    "\u65b0\u6b3e",
    "\u4e13\u67dc\u540c\u6b3e",
    "\u5382\u5bb6\u76f4\u9500",
]

CATEGORY_RULES = [
    ("shoes", ["鞋", "靴", "凉鞋", "单鞋", "高跟", "乐福鞋", "帆布鞋"]),
    ("clothing", ["裙", "裤", "衫", "卫衣", "外套", "马甲", "t恤", "连衣裙", "毛衣", "背带裤"]),
    ("bag", ["包", "双肩包", "手提包", "斜挎", "背包"]),
    ("hat", ["帽", "贝雷帽", "棒球帽", "遮阳帽"]),
    ("jewelry", ["项链", "耳环", "耳钉", "耳夹", "戒指", "手链", "吊坠", "发簪"]),
    ("cup_bottle", ["杯", "壶", "保温杯", "酒瓶", "茶具", "托盘"]),
    ("furniture", ["椅", "柜", "桌", "沙发", "边几", "床", "花瓶", "落地灯", "壁灯"]),
    ("toy", ["玩具", "抱枕", "盒蛋", "模型", "手办", "毛绒"]),
    ("food_drink", ["饮料", "果汁", "酒", "茶", "咖啡", "大米", "面", "鱼", "鸡精", "糖"]),
    ("beauty", ["粉底", "乳液", "面膜", "精油", "口红", "洁面", "香水", "护肤"]),
    ("pet", ["狗", "猫", "宠物", "犬", "鱼食"]),
    ("home_decor", ["挂画", "窗帘", "墙贴", "地毯", "摆件", "纸巾盒", "线帘"]),
    ("tools", ["刀", "工具", "空压机", "扫描仪", "搅拌桨", "电子秤", "火花塞"]),
    ("digital", ["手机壳", "相机", "轮毂", "电磁炉", "吸奶器", "对讲", "遥控器"]),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Clean paired image-text samples and export JSONL.")
    parser.add_argument(
        "--sample-dir",
        default=r"C:\Users\24703\Desktop\chinese_clip\muge_sample",
        help="Directory containing paired .jpg and .txt files.",
    )
    parser.add_argument(
        "--output-path",
        default=r"C:\Users\24703\Desktop\chinese_clip\cleaned_muge_sample.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=2,
        help="Minimum cleaned text length.",
    )
    parser.add_argument(
        "--min-image-side",
        type=int,
        default=64,
        help="Minimum image width/height.",
    )
    parser.add_argument(
        "--drop-brand-only",
        action="store_true",
        help="Drop samples whose cleaned text looks like a brand/store token only.",
    )
    return parser.parse_args()


def normalize_text(text):
    text = text.strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text):
    cleaned = normalize_text(text)
    for term in NOISE_TERMS:
        cleaned = cleaned.replace(term, " ")
    cleaned = re.sub(r"[|/\\]+", " ", cleaned)
    cleaned = re.sub(r"[\u3001\uff0c,\u3002\uff1b;\uff1a:]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_brand_like(text):
    if " " in text:
        return False
    if len(text) <= 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9._-]+", text):
        return True
    return False


def infer_category(cleaned_text):
    for category, keywords in CATEGORY_RULES:
        if any(keyword in cleaned_text for keyword in keywords):
            return category
    return "other"


def infer_group_id(category, cleaned_text):
    tokens = cleaned_text.split()
    if tokens:
        head = tokens[0][:12]
    else:
        head = cleaned_text[:12]
    return f"{category}:{head}" if head else category


def infer_attributes(cleaned_text):
    attribute_keywords = [
        "红", "黑", "白", "蓝", "粉", "紫", "绿",
        "真皮", "纯棉", "玻璃", "木", "不锈钢",
        "中式", "北欧", "复古", "轻奢", "可爱",
        "高跟", "厚底", "圆形", "长款", "短款",
    ]
    return [keyword for keyword in attribute_keywords if keyword in cleaned_text]


def is_valid_image(image_path, min_image_side):
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            if min(width, height) < min_image_side:
                return False, f"image too small: {width}x{height}"
            return True, {"width": width, "height": height}
    except Exception as exc:
        return False, f"image open failed: {exc}"


def iter_pairs(sample_dir):
    for txt_path in sorted(Path(sample_dir).glob("*.txt")):
        image_path = txt_path.with_suffix(".jpg")
        if image_path.exists():
            yield txt_path, image_path


def main():
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    seen = set()
    drop_reasons = {}

    with output_path.open("w", encoding="utf-8") as writer:
        for txt_path, image_path in iter_pairs(sample_dir):
            try:
                raw_text = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                dropped += 1
                drop_reasons["text decode failed"] = drop_reasons.get("text decode failed", 0) + 1
                continue

            cleaned_text = clean_text(raw_text)
            if len(cleaned_text) < args.min_text_length:
                dropped += 1
                drop_reasons["text too short"] = drop_reasons.get("text too short", 0) + 1
                continue

            if args.drop_brand_only and is_brand_like(cleaned_text):
                dropped += 1
                drop_reasons["brand-like text"] = drop_reasons.get("brand-like text", 0) + 1
                continue

            is_valid, image_info = is_valid_image(image_path, args.min_image_side)
            if not is_valid:
                dropped += 1
                drop_reasons[str(image_info)] = drop_reasons.get(str(image_info), 0) + 1
                continue

            dedupe_key = (str(image_path.resolve()), cleaned_text)
            if dedupe_key in seen:
                dropped += 1
                drop_reasons["duplicate pair"] = drop_reasons.get("duplicate pair", 0) + 1
                continue
            seen.add(dedupe_key)

            category = infer_category(cleaned_text)

            record = {
                "id": txt_path.stem,
                "image": str(image_path.resolve()),
                "raw_text": raw_text,
                "clean_text": cleaned_text,
                "text": cleaned_text,
                "source": "muge",
                "stage": "general",
                "category": category,
                "brand": "",
                "group_id": infer_group_id(category, cleaned_text),
                "attributes": infer_attributes(cleaned_text),
                "hard_negative_ids": [],
                "image_width": image_info["width"],
                "image_height": image_info["height"],
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Saved cleaned data to: {output_path}")
    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")
    if drop_reasons:
        print("Drop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda item: item[1], reverse=True):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
