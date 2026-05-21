#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLIP_PROFILE="${1:-apr_media1_project}"
DOUBAO_PROFILE="${2:-apr_media1}"
PORT="${3:-8023}"

export PATH="/opt/conda/bin:${PATH:-}"
exec "$ROOT/.venv/bin/python" -m video_dual.dual_search_service \
  --clip-profile "$CLIP_PROFILE" \
  --doubao-profile "$DOUBAO_PROFILE" \
  --model-path project/models \
  --device cuda \
  --doubao-device cuda \
  --host 0.0.0.0 \
  --port "$PORT"
