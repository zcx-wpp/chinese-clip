# video_retrieval — 统一视频检索

视频检索的**源码、模型权重与本地索引**均在此目录。

## 目录布局

```
video_retrieval/
├── clip/                    # Chinese-CLIP 视觉检索
├── hybrid/                  # 豆包字幕 + 稀疏/稠密混合检索
├── api.py、pipeline.py
├── models/                  # Chinese-CLIP 权重
├── videos/                  # 本地视频（可选）
├── profiles/<name>/         # 推荐：clip/ + hybrid/ 统一 profile
├── legacy/
│   ├── clip_profiles/       # 旧 CLIP 索引
│   └── hybrid_profiles/     # 旧 hybrid 索引
```

未传 `--profile` 时默认 profile：
- CLIP → `legacy/clip_profiles/apr_media1_project/`（`DEFAULT_CLIP_PROFILE`）
- Hybrid → `legacy/hybrid_profiles/apr_media1/`（`DEFAULT_HYBRID_PROFILE`）

详见 [`docs/MIGRATION.md`](../docs/MIGRATION.md)。

## 命令

```bash
python -m video_retrieval.pipeline --profile my_media --video-dir data/videos --steps both --init-unified
python -m video_retrieval.api --profile my_media --model-path video_retrieval/models --port 8023
# 默认：CLIP 与 Hybrid 均在启动时后台预加载；可用 --no-preload-clip / --no-preload-hybrid 关闭
```

### POST /search

```json
{"query": "一个人在厨房做饭", "mode": "both", "top_k": 10}
```

## Profile 解析

1. `profiles/<name>/clip|hybrid/`（统一布局）
2. `legacy/clip_profiles/<name>/`、`legacy/hybrid_profiles/<name>/`

## 代码风格

```bash
.venv/bin/ruff format video_retrieval
.venv/bin/ruff check video_retrieval --fix
```

共享实现（包根目录）：`io_utils.py`、`logging_utils.py`、`portable_paths.py`、`env_loader.py`。详见 [`../docs/CODE_STYLE.md`](../docs/CODE_STYLE.md)。
