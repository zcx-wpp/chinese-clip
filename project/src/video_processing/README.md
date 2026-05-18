# video_processing

This package now keeps one ingestion path and one retrieval path:

- ingestion: `minimal_pipeline.py`
- retrieval: the current FAISS-based path in `retrieval.py`

Removed branches:

- OCR / ASR multimodal ingestion
- Milvus backend
- alternate offline pipeline
- baseline retrieval preset

## Main modules

- `minimal_pipeline.py`: build segments, frames, embeddings, metadata, and FAISS indexes
- `retrieval.py`: current text-to-video retrieval and temporal aggregation
- `api.py`: FastAPI search service
- `batch_search.py`: run retrieval over many queries
- `evaluate.py`: evaluate retrieval quality
- `rebuild_indexes.py`: rebuild FAISS indexes from stored metadata
- `validate_env.py`: environment check

## Pipeline

1. Split each source video into fixed-length segments.
2. Sample candidate frames from each segment.
3. Encode candidate frames with Chinese-CLIP.
4. Keep diverse top-k frames per segment.
5. Pool selected frame embeddings into segment embeddings.
6. Select representative segments per video.
7. Persist metadata in SQLite and indexes in FAISS.
8. Search with video recall, segment recall, frame rerank, and temporal merge.

## Requirements

```bash
pip install torch transformers pillow opencv-python faiss-cpu fastapi uvicorn numpy
```

Also install `ffmpeg` and make sure it is available in `PATH`.

## Validate the environment

```bash
python -m project.src.video_processing.validate_env --model-path D:\models\chinese-clip
```

## Build an index

```bash
python -m project.src.video_processing.minimal_pipeline ^
  --video-dir D:\data\videos ^
  --output-dir D:\data\video_index\output ^
  --metadata-db D:\data\video_index\metadata.db ^
  --model-path D:\models\chinese-clip ^
  --device cuda
```

Common knobs:

- `--segment-seconds`
- `--frames-per-second`
- `--top-k-per-segment`
- `--limit`

## Run the API

```bash
python -m project.src.video_processing.api ^
  --output-dir D:\data\video_index\output ^
  --metadata-db D:\data\video_index\metadata.db ^
  --model-path D:\models\chinese-clip ^
  --device cuda
```

Default endpoints:

- `POST /search`
- `GET /health`
- `GET /media?path=...`

## Batch search

```bash
python -m project.src.video_processing.batch_search ^
  --output-dir D:\data\video_index\output ^
  --metadata-db D:\data\video_index\metadata.db ^
  --model-path D:\models\chinese-clip ^
  --queries-file D:\data\queries.txt ^
  --top-k 10
```

## Evaluate

```bash
python -m project.src.video_processing.evaluate ^
  --output-dir D:\data\video_index\output ^
  --metadata-db D:\data\video_index\metadata.db ^
  --model-path D:\models\chinese-clip ^
  --labels D:\data\eval_labels.json
```

## Demo flow

```bash
python -m project.src.video_processing.run_demo ^
  --video-dir D:\data\videos ^
  --output-dir D:\data\video_index\output ^
  --metadata-db D:\data\video_index\metadata.db ^
  --model-path D:\models\chinese-clip ^
  --labels D:\data\eval_labels.json
```
