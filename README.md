# chinese-clip

文搜图 / 文搜视频多方案检索与入库工具集。

## 模块

| 目录 | 说明 |
|------|------|
| `project/` | Chinese-CLIP **视频**分段、建索引、帧级检索 |
| `doubao_pipeline/` | 豆包字幕 + 稀疏/稠密 **混合视频检索** |
| `picture/` | MUGE 等 **图片**：CLIP 与 MLLM+BGE 双方案 |
| `video_dual/` | **视频双路对比** UI（project CLIP vs doubao 混合） |

## 本地环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 若存在；否则见 chinese_clip/environment.yml
```

复制 `.env.example` 为 `.env`（勿提交），配置 `ARK_API_KEY` 等。

模型权重、数据、FAISS 索引需自行放到 `project/models/`、`data/`、各 `profiles/`（已在 `.gitignore` 中排除）。

## 检索服务

**图片双路（8022）**

```bash
bash scripts/run_dual_picture_search.sh muge_train_clip muge_1k_bge \
  data/muge/train_extracted 8022
```

**视频双路（8023）**

```bash
bash scripts/run_dual_video_search.sh
# 默认：clip=apr_media1_project，doubao=apr_media1
```

上传 GitHub 说明见 [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)。
