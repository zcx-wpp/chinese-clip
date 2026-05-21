#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE="${1:-default}"
IMAGE_DIR="${2:-}"
if [[ -z "$IMAGE_DIR" ]]; then
  echo "Usage: $0 <profile> <image-dir>" >&2
  exit 1
fi
export PATH="/opt/conda/bin:${PATH:-}"
exec "$ROOT/.venv/bin/python" -m picture.ingest \
  --profile "$PROFILE" \
  --image-dir "$IMAGE_DIR" \
  --model-path project/models \
  --device cuda
