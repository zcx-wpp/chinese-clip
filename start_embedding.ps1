param(
    [string]$EnvName = "embedding",
    [int]$Port = 8002,
    [string]$BindHost = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Join-Path $repoRoot "chinese_clip"
$modelPath = Join-Path $repoRoot "model"

if (-not (Test-Path $projectRoot)) {
    throw "Project directory not found: $projectRoot"
}

if (-not (Test-Path $modelPath)) {
    throw "Model directory not found: $modelPath"
}

$env:CHINESE_CLIP_MODEL_PATH = $modelPath

Write-Host "Project root: $projectRoot"
Write-Host "Model path: $modelPath"
Write-Host "Starting service on http://$BindHost`:$Port using conda env '$EnvName'..."

conda run -n $EnvName uvicorn app.api_server_embedding:app --host $BindHost --port $Port --app-dir $projectRoot
