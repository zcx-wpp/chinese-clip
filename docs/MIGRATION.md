# 目录迁移说明

`project/` 与 `doubao_pipeline/` 已移除，视频检索统一使用 `video_retrieval/`。

## 路径对照

| 旧路径 | 新路径 |
|--------|--------|
| `project/models` | `video_retrieval/models` |
| `project/profiles/<name>` | `video_retrieval/legacy/clip_profiles/<name>` |
| `project/output` | `video_retrieval/legacy/clip_profiles/<clip_profile>/output` |
| `project/metadata` | `video_retrieval/legacy/clip_profiles/<clip_profile>/metadata.db` |
| `video_retrieval/workspace/*`（已废弃） | 同上，由 `path_compat.py` 重写到默认 clip profile |
| `project/videos` | `video_retrieval/videos` |
| `doubao_pipeline/profiles/<name>` | `video_retrieval/legacy/hybrid_profiles/<name>` |
| `doubao_pipeline/artifacts` | `video_retrieval/legacy/hybrid_profiles/<hybrid_profile>/` |
| `video_retrieval/artifacts`（已废弃） | 同上，由 `path_compat.py` 重写到默认 hybrid profile |

索引/metadata 里若仍保存旧绝对路径，运行时会由 `video_retrieval/path_compat.py` 自动改写。

## 命令对照

| 旧命令 | 新命令 |
|--------|--------|
| `python -m project.src.video_processing.minimal_pipeline` | `python -m video_retrieval.clip.minimal_pipeline` |
| `python -m doubao_pipeline.doubao_batch_caption` | `python -m video_retrieval.hybrid.doubao_batch_caption` |
| `python -m doubao_pipeline.build_hybrid_index` | `python -m video_retrieval.hybrid.build_hybrid_index` |

统一建库与检索：

```bash
python -m video_retrieval.pipeline --profile NAME --video-dir data/videos --steps both
python -m video_retrieval.api --profile NAME --model-path video_retrieval/models
```
