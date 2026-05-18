# video_processing

This package contains the maintained video ingestion and retrieval path for the
repository.

## Main modules

- `minimal_pipeline.py`: build segments, frames, embeddings, metadata, and FAISS indexes
- `retrieval.py`: text-to-video retrieval and temporal aggregation
- `api.py`: FastAPI search service
- `batch_search.py`: run retrieval over many queries
- `evaluate.py`: evaluate retrieval quality
- `rebuild_indexes.py`: rebuild FAISS indexes from stored metadata
- `validate_env.py`: environment check
- `run_demo.py`: validate, index, and evaluate in one flow

## Pipeline

1. Split each source video into fixed-length segments.
2. Sample candidate frames from each segment.
3. Encode candidate frames with Chinese-CLIP.
4. Keep diverse top-k frames per segment.
5. Pool selected frame embeddings into segment embeddings.
6. Select representative segments per video.
7. Persist metadata in SQLite and indexes in FAISS.
8. Search with video recall, segment recall, frame rerank, and temporal merge.

## Runtime requirements

```bash
pip install torch transformers pillow opencv-python faiss-cpu fastapi uvicorn numpy
```

Also install `ffmpeg` and make sure it is available in `PATH`.

## Default paths

Without `--profile`, commands default to:

- videos: `project/videos/`
- output: `project/output/`
- metadata DB: `project/metadata/metadata.db`
- model path: `project/models/`

With `--profile seg4s`, profile-aware commands default to:

- output: `project/profiles/seg4s/output/`
- metadata DB: `project/profiles/seg4s/metadata.db`

Profile-aware commands are:

- `minimal_pipeline.py`
- `api.py`
- `batch_search.py`
- `evaluate.py`
- `rebuild_indexes.py`

## Validate the environment

```bash
python -m project.src.video_processing.validate_env --model-path project/models
```

## Build an index

```bash
python -m project.src.video_processing.minimal_pipeline ^
  --profile seg4s ^
  --video-dir data/videos ^
  --model-path project/models ^
  --device cpu
```

Common knobs:

- `--segment-seconds`
- `--frames-per-second`
- `--top-k-per-segment`
- `--limit`

## Run the API

```bash
python -m project.src.video_processing.api ^
  --profile seg4s ^
  --model-path project/models ^
  --device cpu
```

Default endpoints:

- `POST /search`
- `GET /health`
- `GET /media?path=...`

## Batch search

```bash
python -m project.src.video_processing.batch_search ^
  --profile seg4s ^
  --model-path project/models ^
  --queries-file project/metadata/sample_queries.txt ^
  --device cpu ^
  --top-k 10
```

## Evaluate

```bash
python -m project.src.video_processing.evaluate ^
  --profile seg4s ^
  --model-path project/models ^
  --labels project/metadata/1799eval_labels.json ^
  --device cpu
```

## Rebuild indexes

```bash
python -m project.src.video_processing.rebuild_indexes ^
  --profile seg4s ^
  --model-path project/models
```

## Demo flow

```bash
python -m project.src.video_processing.run_demo ^
  --video-dir data/videos ^
  --output-dir project/output ^
  --metadata-db project/metadata/metadata.db ^
  --model-path project/models ^
  --labels project/metadata/1799eval_labels.json ^
  --device cpu
```
