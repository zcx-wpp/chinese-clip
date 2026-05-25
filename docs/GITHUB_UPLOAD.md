# 上传到 GitHub（zcx-wpp/chinese-clip）

远程已配置：`git@github.com:zcx-wpp/chinese-clip.git`

## 不会上传的内容（见 `.gitignore`）

- `.env`、`.venv/`
- `data/`、`video_retrieval/models/`、`video_retrieval/videos/`
- `picture/profiles/`、`video_retrieval/profiles/`、`video_retrieval/legacy/`（本地索引与 metadata.db）
- 各类 `*.faiss`、`*.npy`、模型权重

## 推荐步骤

```bash
cd /data/chuangxin.zhang/chinese_clip

# 1. 检查状态（确认没有 .env / data / models）
git status

# 2. 只添加源码与文档
git add .gitignore docs/ README.md requirements.txt pyproject.toml
git add picture/*.py video_retrieval/*.py
git add video_retrieval/clip/ video_retrieval/hybrid/
git add chinese_clip/environment.yml 2>/dev/null || true
git add -u . 2>/dev/null || true

# 3. 再次确认暂存列表
git diff --cached --name-only | head -50

# 4. 提交
git commit -m "$(cat <<'EOF'
Add picture and video dual-search services, doubao pipeline, and project API fixes.

EOF
)"

# 5. 推送
git pull origin main --rebase
git push origin main
```

## SSH 未配置时

```bash
git remote set-url origin https://github.com/zcx-wpp/chinese-clip.git
git push origin main
# 使用 GitHub Personal Access Token 作为密码
```

## 误添加敏感文件时

```bash
git reset HEAD .env
git rm -r --cached data/ video_retrieval/models/ .venv/ 2>/dev/null || true
```
