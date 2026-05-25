# 代码风格规范

本项目 Python 代码统一由 [Ruff](https://docs.astral.sh/ruff/) 格式化与检查。

## 约定

| 项 | 规范 |
|----|------|
| 缩进 | 4 空格 |
| 行宽 | 100 字符 |
| 引号 | 双引号 `"` |
| 分号 | 不使用（Python 标准） |
| 导入 | `isort` 规则，首方包：`video_retrieval`、`picture`、`chinese_clip` |
| 类型 | 新代码优先 `from __future__ import annotations` |
| 注释 | 模块/复杂逻辑用中文或英文均可；避免无意义重复注释 |

## 常用命令

```bash
# 格式化
.venv/bin/ruff format video_retrieval picture chinese_clip

# 检查并自动修复可安全项
.venv/bin/ruff check video_retrieval picture chinese_clip --fix
```

配置见仓库根目录 `pyproject.toml`。

## 共享工具模块

| 模块 | 说明 |
|------|------|
| `video_retrieval/io_utils.py` | JSON / 文本行读写 |
| `video_retrieval/logging_utils.py` | 统一日志 |
| `video_retrieval/portable_paths.py` | 视频路径解析（`clip/`、`hybrid/` 直接 `from ..portable_paths import ...`） |
| `video_retrieval/path_compat.py` | 旧 `project/`、`doubao_pipeline/` 路径改写 |
| `video_retrieval/metadata_paths.py` | CLIP/hybrid metadata 路径字段规范化 |
| `video_retrieval/clip/index_builder.py` | 从帧向量重建 segment 与 FAISS 索引 |
