# video_processing

独立的视频向量化处理与检索模块，目标是实现中文文本到视频片段检索。

## 当前能力

- 中文文本 -> 视频检索
- 返回视频 ID
- 返回秒级时间段
- 返回相关片段区间

## 当前版本不处理

- 动作因果推理
- 复杂行为理解

## 模块结构

- `config.py`: 系统配置
- `segmenter.py`: 视频切片
- `frame_selector.py`: 抽帧与 Top-K 代表帧选择
- `embedding.py`: Chinese-CLIP 编码封装
- `multimodal.py`: OCR / ASR 扩展
- `metadata_store.py`: SQLite 元数据存储
- `faiss_store.py`: FAISS 向量索引
- `milvus_store.py`: Milvus 向量索引
- `vector_store.py`: 向量库抽象层
- `offline_pipeline.py`: 离线建库入口
- `retrieval.py`: 在线检索与时间聚合
- `api.py`: FastAPI 搜索服务
- `evaluate.py`: 检索评估脚本
- `extract_video_frames.py`: 独立抽帧工具
- `run_demo.py`: 最小流程脚本
- `validate_env.py`: 环境与依赖自检
- `demo_eval_labels.json`: 标注样例

## 系统流程

1. 原始视频按固定时长切片
2. 每段按固定 FPS 抽样
3. 使用 Chinese-CLIP 编码候选帧
4. 根据向量范数和相似度去重后选 Top-K 代表帧
5. 可选融合 OCR / ASR 文本信号
6. 向量写入 FAISS 或 Milvus
7. 元数据写入 SQLite
8. 在线 Query 编码、召回、聚合，输出视频片段

## 依赖

基础依赖：

```bash
pip install torch transformers pillow opencv-python faiss-cpu fastapi uvicorn numpy
```

可选依赖：

```bash
pip install pymilvus paddleocr openai-whisper
```

系统工具：

- `ffmpeg`

## 环境自检

```bash
python -m video_processing.validate_env ^
  --model-path D:\zcx\chinese_clip\model
```

## 离线建库

```bash
python -m video_processing.offline_pipeline ^
  --video-dir D:\data\videos ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model
```

常用可选参数：

- `--enable-ocr`
- `--enable-asr`
- `--vector-backend faiss|milvus`
- `--segment-seconds 8`
- `--frames-per-second 2.0`
- `--top-k-per-segment 4`

开启 OCR / ASR：

```bash
python -m video_processing.offline_pipeline ^
  --video-dir D:\data\videos ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model ^
  --enable-ocr ^
  --enable-asr ^
  --whisper-model base
```

使用 Milvus：

```bash
python -m video_processing.offline_pipeline ^
  --video-dir D:\data\videos ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model ^
  --vector-backend milvus ^
  --milvus-uri http://127.0.0.1:19530 ^
  --milvus-collection video_frame_embeddings
```

输出目录示例：

- `segments/`
- `frames/`
- `metadata.db`
- `frame_index.faiss`
- `frame_index.meta.json`

## 在线检索服务

```bash
python -m video_processing.api ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model
```

使用 Milvus：

```bash
python -m video_processing.api ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model ^
  --vector-backend milvus ^
  --milvus-uri http://127.0.0.1:19530 ^
  --milvus-collection video_frame_embeddings
```

默认接口：

- `POST /search`
- `GET /health`

请求示例：

```json
{
  "query": "红衣女孩跳舞",
  "top_k": 5
}
```

返回示例：

```json
[
  {
    "video_id": "demo_video",
    "score": 0.92,
    "segments": [
      {
        "start": 83.2,
        "end": 88.1
      }
    ]
  }
]
```

## 评估

标注格式示例：

```json
[
  {
    "query": "红衣女孩跳舞",
    "video_id": "vid_001",
    "segments": [
      {
        "start": 83.0,
        "end": 89.0
      }
    ]
  }
]
```

运行评估：

```bash
python -m video_processing.evaluate ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model ^
  --labels D:\data\video_eval_labels.json
```

当前输出指标：

- `Recall@K`
- `MRR`
- `Top1 accuracy`
- `time_hit_rate`
- `avg_time_center_distance`

## 最小 Demo

样例标注文件：

- `video_processing/demo_eval_labels.json`

一键跑最小流程：

```bash
python -m video_processing.run_demo ^
  --video-dir D:\data\videos ^
  --work-dir D:\data\video_index ^
  --model-path D:\zcx\chinese_clip\model ^
  --labels video_processing\demo_eval_labels.json
```

默认流程：

1. 先做环境检查
2. 再跑离线建库
3. 最后跑评估

可选参数：

- `--skip-validate`
- `--skip-index`
- `--skip-eval`

## 后续建议

- 接入真实 OCR / ASR 生产配置
- 增加重排序模型
- 加入缓存与热门 Query 优化
- 补充真实标注集与回归评估
- 增加前端检索页
