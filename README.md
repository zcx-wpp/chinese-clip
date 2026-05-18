# chinese_clip

This repository contains several Chinese-CLIP related workstreams in one place:

- training and evaluation scripts at the repository root
- the current video retrieval / indexing pipeline under `project/`
- an API-oriented environment under `chinese_clip/`
- an older standalone `video_processing/` package kept for reference

For current video ingestion, retrieval, and analysis work, prefer
`project/src/video_processing/`.

## Repository layout

```text
.
|-- README.md
|-- requirements.txt
|-- train_chinese_clip.py
|-- evaluate_checkpoint.py
|-- build_validation_splits.py
|-- chinese_clip/
|   |-- environment.yml
|   `-- app/
|-- model/
|-- project/
|   |-- metadata/
|   |-- models/
|   |-- src/video_processing/
|   `-- run_*.cmd
`-- video_processing/
```

## What is tracked

Git is currently set up to track:

- source code
- shell / cmd / powershell scripts
- documentation
- small config files
- selected metadata and query examples

Git is set up to ignore local heavy artifacts such as:

- virtual environments
- checkpoints
- model weight files such as `*.pt` and `*.bin`
- generated embeddings such as `*.npy`
- local outputs under `project/output/` and `project/profiles/`
- large local datasets and logs

If you later need versioned model weights, use Git LFS rather than plain Git.

## Environment

This repo currently has two environment entry points:

1. `requirements.txt`
   General Python environment snapshot used by the current workspace.
2. `chinese_clip/environment.yml`
   Smaller Conda environment focused on the API / embedding service side.

If you are working inside this repo on Windows with the local virtual
environment, common commands can be run with:

```powershell
.venv\Scripts\python.exe --version
```

## Common workflows

### 1. Run the current minimal ingestion pipeline

From the repository root:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --profile seg4s --video-dir data/videos --device cpu
```

There are also ready-made helper scripts:

```powershell
project\run_ingest_seg4s_incremental.cmd
project\run_ingest_msrvtt500_seg4s_test.cmd
project\run_eval_500_seg4s.cmd
```

### 2. Run the offline pipeline directly

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.offline_pipeline --device cpu
```

### 3. Run analysis or retrieval tools

Examples:

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.batch_search --profile seg4s --retrieval-preset current
.venv\Scripts\python.exe -m project.src.video_processing.analyze_recall_stages --profile seg4s --retrieval-preset current
```

## Git workflow

The repository has already been initialized and connected to:

```text
origin = https://github.com/zcx-wpp/chinese-clip.git
```

Daily workflow:

```powershell
git status
git add -A
git commit -m "feat: short summary"
git push
```

Suggested commit prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `refactor:` internal cleanup without behavior change
- `docs:` README or documentation updates
- `chore:` repo maintenance, ignore rules, tooling
- `exp:` temporary experiment or evaluation-only change

## Useful recovery commands

Inspect recent history:

```powershell
git log --oneline --decorate -10
```

See exactly what changed:

```powershell
git diff
git diff --cached
```

Revert a completed commit without rewriting history:

```powershell
git revert <commit>
```

Restore a single file from the latest commit:

```powershell
git restore <path>
```

## Notes

- The repo currently mixes experiment code and pipeline code in one place.
- `project/src/video_processing/` appears to be the main maintained path.
- `video_processing/` is still useful as a smaller standalone reference module.
- Some existing README files in subdirectories may use a different text encoding.
