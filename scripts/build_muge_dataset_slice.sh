#!/usr/bin/env bash
# Extract a slice of images from MUGE train_extracted (for incremental ingest).
# Usage: bash scripts/build_muge_dataset_slice.sh [src] [out_dir] [offset] [limit]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/data/muge/train_extracted}"
OUT="${2:-$ROOT/data/muge/dataset_1k_add}"
OFFSET="${3:-1000}"
LIMIT="${4:-1000}"

exec "$ROOT/.venv/bin/python" - "$SRC" "$OUT" "$OFFSET" "$LIMIT" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
offset = int(sys.argv[3])
limit = int(sys.argv[4])
img_dir = out / "images"
img_dir.mkdir(parents=True, exist_ok=True)

all_jpgs = sorted(src.glob("s*.jpg"), key=lambda p: p.stem)
jpgs = all_jpgs[offset : offset + limit]
if not jpgs:
    raise SystemExit(f"empty slice offset={offset} limit={limit} (total={len(all_jpgs)})")

tsv_lines = []
txt_lines = []
for idx, jpg in enumerate(jpgs, start=offset + 1):
    base = jpg.stem
    dst = img_dir / jpg.name
    dst.write_bytes(jpg.read_bytes())
    txt_path = src / f"{base}.txt"
    caption = (
        txt_path.read_text(encoding="utf-8", errors="replace")
        .replace("\r", "")
        .replace("\n", " ")
        .strip()
        if txt_path.exists()
        else ""
    )
    tsv_lines.append(f"{idx}\t{base}\t{caption}")
    txt_lines.append(caption)

(out / "captions.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
(out / "captions.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
first_id = jpgs[0].stem
last_id = jpgs[-1].stem
(out / "README.md").write_text(
    f"# MUGE 增量子集\n\n"
    f"- 来源排序切片：offset={offset}, limit={limit}\n"
    f"- image_id 范围：{first_id} … {last_id}\n"
    f"- 张数：{len(jpgs)}\n",
    encoding="utf-8",
)
print(f"[done] slice [{offset}:{offset+limit}) -> {out} count={len(jpgs)} ids={first_id}..{last_id}")
PY
