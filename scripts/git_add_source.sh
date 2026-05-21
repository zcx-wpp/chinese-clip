#!/usr/bin/env bash
# Stage source code only (respects .gitignore). Review with: git diff --cached --name-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git add .gitignore docs/GITHUB_UPLOAD.md scripts/

# picture: Python package only (profiles/ ignored by .gitignore)
git add picture/*.py

# video dual search
git add video_dual/

# doubao_pipeline: package code only (profiles/ ignored)
git add doubao_pipeline/*.py

# project: source + small metadata text/json (not metadata.db)
git add project/src/
git add project/metadata/*.json project/metadata/*.txt 2>/dev/null || true

# chinese_clip env
git add chinese_clip/environment.yml 2>/dev/null || true

# restore or update top-level docs if present
[[ -f README.md ]] && git add README.md
[[ -f requirements.txt ]] && git add requirements.txt

# track intentional deletions from old layout
git add -u README.md requirements.txt .gitignore project/README.md 2>/dev/null || true
git add -u project/run_ingest_msrvtt500_seg4s_test.cmd project/run_ingest_seg4s_incremental.cmd 2>/dev/null || true

echo "[git_add_source] Staged files:"
git diff --cached --name-only
