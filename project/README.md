# project

视频处理代码已经单独整理到下面这套目录结构中：

```text
project/
|
├── videos/
│
├── output/
│   ├── segments/
│   ├── frames/
│   ├── embeddings/
│   ├── faiss/
│   └── logs/
│
├── metadata/
│   └── metadata.db
│
├── models/
│
└── src/
    └── video_processing/
```

## 各目录职责

- `videos/`: 原始视频输入
- `output/segments/`: ffmpeg 切片结果
- `output/frames/`: 抽帧结果
- `output/embeddings/`: 预留给 embedding 中间产物
- `output/faiss/`: FAISS 索引文件
- `output/logs/`: 日志输出
- `metadata/metadata.db`: SQLite 元数据库
- `models/`: Chinese-CLIP 模型目录
- `src/video_processing/`: 视频处理代码

## 当前代码默认约定

`project/src/video_processing/` 里的代码已经按这套结构设置默认路径：

- 默认视频目录：`project/videos/`
- 默认输出目录：`project/output/`
- 默认元数据目录：`project/metadata/`
- 默认模型目录：`project/models/`

离线建库默认会写：

- `project/output/segments/...`
- `project/output/frames/...`
- `project/output/faiss/frame_index.faiss`
- `project/output/faiss/frame_index.meta.json`
- `project/metadata/metadata.db`

## 运行方式

在仓库根目录执行：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.offline_pipeline --device cpu
```

## Minimal Pipeline

如果你只想跑最小可运行版本，只做：

1. segment
2. frame
3. clip embedding
4. top-k
5. faiss

运行：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --device cpu
```

常用测试命令，例如先跑 1 个视频：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --device cpu --limit 1
```

先跑 10 个视频：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --device cpu --limit 10
```

最小版输出只关注三类结果：

- `project/output/embeddings/frame_embeddings.npy`
- `project/output/embeddings/frame_embeddings.jsonl`
- `project/output/faiss/frame_index.faiss`
- `project/output/faiss/frame_index.meta.json`
- `project/metadata/metadata.db`

评估：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.evaluate --labels D:\path\to\eval_labels.json --device cpu
```

启动 API：

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.api --device cpu --retrieval-preset current
```

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.batch_search --profile seg4s --retrieval-preset current
.venv\Scripts\python.exe -m project.src.video_processing.batch_search --profile seg4s --retrieval-preset baseline
```

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.analyze_recall_stages --profile seg4s --retrieval-preset current
.venv\Scripts\python.exe -m project.src.video_processing.analyze_recall_stages --profile seg4s --retrieval-preset baseline
```

返回结果中的 clip 现在包含 `score`，例如：

```json
[
  {
    "video_id": "demo_video",
    "score": 0.91,
    "segments": [
      {
        "start": 1.0,
        "end": 3.0,
        "score": 0.88
      }
    ],
    "video_path": "D:/path/to/demo_video.mp4"
  }
]
```

## 备注

如果你后面愿意，我还可以继续做两件事：

1. 把 `run_demo.py`、`validate_env.py` 也一起切成这套 `project/` 的默认路径风格  
2. 把现有 `data/` 里的试验数据软链接或拷贝到 `project/videos/` / `project/models/`
