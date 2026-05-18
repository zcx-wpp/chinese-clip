# chinese_clip

This repository now keeps two active pieces:

- `project/`: the maintained Chinese-CLIP video indexing and retrieval pipeline
- `chinese_clip/`: a smaller API-oriented environment and helper app code

The main maintained code path is `project/src/video_processing/`.

## Repository layout

```text
.
|-- README.md
|-- requirements.txt
|-- chinese_clip/
|   |-- environment.yml
|   `-- app/
|-- data/
`-- project/
    |-- metadata/
    |-- models/
    |-- output/
    |-- profiles/
    |-- src/video_processing/
    |-- videos/
    |-- run_ingest_msrvtt500_seg4s_test.cmd
    `-- run_ingest_seg4s_incremental.cmd
```

## What is tracked

Git is intended to track:

- source code
- helper scripts
- documentation
- small config files
- selected query and label examples

Local heavy artifacts should stay untracked:

- virtual environments
- model weight files
- checkpoints
- generated embeddings
- generated indexes and logs
- large local datasets

If you later need versioned model weights, use Git LFS rather than plain Git.

## Environments

This repo currently has two environment entry points:

1. `requirements.txt`
   General Python environment snapshot used by the current workspace.
2. `chinese_clip/environment.yml`
   Smaller Conda environment for the helper app under `chinese_clip/app/`.

If you are working inside this repo on Windows with the local virtual
environment, common commands can be run with:

```powershell
.venv\Scripts\python.exe --version
```

## Common workflows

### 1. Validate the local runtime

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.validate_env --model-path project/models
```

### 2. Build or update an index

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.minimal_pipeline --profile seg4s --video-dir data/videos --model-path project/models --device cpu
```

That command writes output under `project/profiles/seg4s/`.

There are also ready-made helper scripts:

```powershell
project\run_ingest_seg4s_incremental.cmd
project\run_ingest_msrvtt500_seg4s_test.cmd
```

### 3. Run the search API

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.api --profile seg4s --model-path project/models --device cpu
```

### 4. Run batch retrieval

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.batch_search --profile seg4s --model-path project/models --queries-file project/metadata/sample_queries.txt --device cpu --top-k 10
```

### 5. Evaluate retrieval quality

```powershell
.venv\Scripts\python.exe -m project.src.video_processing.evaluate --profile seg4s --model-path project/models --labels project/metadata/1799eval_labels.json --device cpu
```

## Checked-in examples

The remaining checked-in metadata examples are:

- `project/metadata/sample_queries.txt`
- `project/metadata/1799eval_labels.json`

These are enough for quick local retrieval and evaluation checks.

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

- `project/src/video_processing/` is the main maintained path.
- `project/models/`, `project/output/`, and `project/profiles/` are local working areas.
- Some existing files in the repo may still use older text encodings.
