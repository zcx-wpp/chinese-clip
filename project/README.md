# project

This directory contains the maintained video indexing and retrieval workspace.

## Layout

```text
project/
|-- metadata/
|   |-- 1799eval_labels.json
|   |-- metadata.db
|   `-- sample_queries.txt
|-- models/
|-- output/
|-- profiles/
|-- src/
|   `-- video_processing/
|-- videos/
|-- run_ingest_msrvtt500_seg4s_test.cmd
`-- run_ingest_seg4s_incremental.cmd
```

## Path conventions

Code under `project/src/video_processing/` uses these defaults:

- default video directory: `project/videos/`
- default output directory: `project/output/`
- default metadata DB: `project/metadata/metadata.db`
- default model directory: `project/models/`

If you pass `--profile seg4s`, the profile-specific defaults become:

- output directory: `project/profiles/seg4s/output/`
- metadata DB: `project/profiles/seg4s/metadata.db`

The helper scripts in this repo may still point at alternate local video
directories such as `data/videos/` or `data/data1/...`.

## Core commands

Validate the runtime:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.validate_env --model-path project/models
```

Run the maintained ingestion pipeline:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --profile seg4s --video-dir data/videos --model-path project/models --device cpu
```

Useful flags:

- `--limit`
- `--segment-seconds`
- `--frames-per-second`
- `--top-k-per-segment`
- `--video-workers`
- `--num-workers`

Start the search API:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.api --profile seg4s --model-path project/models --device cpu
```

Run batch search:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.batch_search --profile seg4s --model-path project/models --queries-file project/metadata/sample_queries.txt --device cpu --top-k 10
```

Evaluate against the checked-in labels:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.evaluate --profile seg4s --model-path project/models --labels project/metadata/1799eval_labels.json --device cpu
```

Rebuild FAISS indexes from cached embeddings:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.rebuild_indexes --profile seg4s --model-path project/models
```

Run the demo flow with explicit paths:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.run_demo --video-dir data/videos --output-dir project/output --metadata-db project/metadata/metadata.db --model-path project/models --labels project/metadata/1799eval_labels.json --device cpu
```

## Helper scripts

- `run_ingest_seg4s_incremental.cmd`
  Incremental ingest over `data/videos` into the `seg4s` profile.
- `run_ingest_msrvtt500_seg4s_test.cmd`
  Local 500-video ingest helper. Review the hard-coded video path before reuse.

## Notes

- `project/models/` is the local model location kept by the current workflow.
- `project/output/` and `project/profiles/` are generated working areas.
- `project/metadata/sample_queries.txt` and `project/metadata/1799eval_labels.json` are the remaining checked-in examples.
