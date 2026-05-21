#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLIP_PROFILE="${1:-muge_1k_clip}"
BGE_PROFILE="${2:-muge_1k_bge}"
IMAGE_DIR="${3:-$ROOT/data/muge/train_extracted}"
PORT="${4:-8022}"

export PATH="/opt/conda/bin:${PATH:-}"
exec "$ROOT/.venv/bin/python" -m picture.dual_search_service \
  --clip-profile "$CLIP_PROFILE" \
  --bge-profile "$BGE_PROFILE" \
  --image-dir "$IMAGE_DIR" \
  --model-path project/models \
  --device cuda \
  --bge-device cuda \
  --host 0.0.0.0 \
  --port "$PORT"
