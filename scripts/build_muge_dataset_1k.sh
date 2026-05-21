#!/usr/bin/env bash
# Extract first N images from MUGE train_extracted + merge captions into one file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/data/muge/train_extracted}"
OUT="${2:-$ROOT/data/muge/dataset_1k}"
LIMIT="${3:-1000}"

exec "$ROOT/.venv/bin/python" - "$SRC" "$OUT" "$LIMIT" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
limit = int(sys.argv[3])
img_dir = out / "images"
img_dir.mkdir(parents=True, exist_ok=True)

jpgs = sorted(src.glob("s*.jpg"), key=lambda p: p.stem)[:limit]
if not jpgs:
    raise SystemExit(f"no images under {src}")

tsv_lines = []
txt_lines = []
for idx, jpg in enumerate(jpgs, start=1):
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
(out / "README.md").write_text(
    f"# MUGE 子集（{len(jpgs)} 张）\n\n"
    f"- `images/`：{len(jpgs)} 张 jpg（s0000000 …）\n"
    f"- `captions.tsv`：序号 \\t image_id \\t 文本\n"
    f"- `captions.txt`：仅文本，{len(jpgs)} 行，与图片排序一致\n",
    encoding="utf-8",
)
print(f"[done] {len(jpgs)} -> {out}")
PY
