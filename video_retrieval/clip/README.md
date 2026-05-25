# video_retrieval.clip — Chinese-CLIP 视频检索

## Main modules

- `minimal_pipeline.py`: build segments, frames, embeddings, metadata, and FAISS indexes
- `retrieval.py`: text-to-video retrieval and temporal aggregation
- `retriever_factory.py`: build retriever for programmatic use
- `index_builder.py`: shared segment/FAISS rebuild logic
- `rebuild_indexes.py`: CLI to rebuild indexes from cached frame embeddings
- `reset_task.py`: reset pipeline task status for retry
- `validate_env.py`: environment check

## Default paths

Without `--profile`:

- videos: `video_retrieval/videos/` or `data/videos/`
- 未传 `--profile` 时默认：`legacy/clip_profiles/apr_media1_project/output` 与同级 `metadata.db`
- model path: `video_retrieval/models/`

With `--profile seg4s`:

- output: `video_retrieval/legacy/clip_profiles/seg4s/output/`
- metadata DB: `video_retrieval/legacy/clip_profiles/seg4s/metadata.db`

## Commands

```bash
python -m video_retrieval.clip.validate_env --model-path video_retrieval/models

python -m video_retrieval.clip.minimal_pipeline \
  --profile seg4s --video-dir data/videos \
  --model-path video_retrieval/models --device cuda

python -m video_retrieval.clip.rebuild_indexes --profile seg4s --model-path video_retrieval/models

python -m video_retrieval.clip.reset_task --metadata-db video_retrieval/legacy/clip_profiles/seg4s/metadata.db \
  --video-id SOME_ID --stage embedding
```
