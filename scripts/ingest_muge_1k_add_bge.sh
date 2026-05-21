#!/usr/bin/env bash
# Add next 1000 images to existing profile muge_1k_bge (new dir + same profile).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-muge_1k_bge}"
ADD_DIR="${ADD_DIR:-$ROOT/data/muge/dataset_1k_add/images}"
WORKERS="${WORKERS:-4}"

echo "[1/2] build slice offset=1000 limit=1000 -> data/muge/dataset_1k_add"
bash "$ROOT/scripts/build_muge_dataset_slice.sh" \
  "$ROOT/data/muge/train_extracted" \
  "$ROOT/data/muge/dataset_1k_add" \
  1000 1000

echo "[2/2] text_ingest profile=$PROFILE image-dir=$ADD_DIR"
export PATH="/opt/conda/bin:${PATH:-}"
exec "$ROOT/.venv/bin/python" -m picture.text_ingest \
  --profile "$PROFILE" \
  --image-dir "$ADD_DIR" \
  --bge-device cuda \
  --workers "$WORKERS"
