@echo off
setlocal

cd /d "%~dp0.."

echo [1/6] Batch retrieval on 500 queries
python -u -m project.src.video_processing.batch_search ^
  --profile seg4s ^
  --device cpu ^
  --top-k 10 ^
  --queries-file project/metadata/sample_queries.txt ^
  --query-workers 2
if errorlevel 1 goto :fail

echo.
echo [2/6] Analyze retrieval gaps against eval labels
python -u -m project.src.video_processing.analyze_retrieval_gaps ^
  --profile seg4s ^
  --top-k 10
if errorlevel 1 goto :fail

echo.
echo [3/6] Summarize retrieval gaps
python -u -m project.src.video_processing.summarize_retrieval_gaps ^
  --profile seg4s
if errorlevel 1 goto :fail

echo.
echo [4/6] Plot TopK hit-rate curve
python -u -m project.src.video_processing.plot_retrieval_hit_curve ^
  --profile seg4s
if errorlevel 1 goto :fail

echo.
echo [5/6] Analyze recall stages
python -u -m project.src.video_processing.analyze_recall_stages ^
  --profile seg4s ^
  --top-k 10 ^
  --device cpu ^
  --queries-file project/metadata/sample_queries.txt
if errorlevel 1 goto :fail

echo.
echo [6/6] Analyze rerank failures
python -u -m project.src.video_processing.analyze_rerank_failures ^
  --profile seg4s
if errorlevel 1 goto :fail

echo.
echo Done.
echo Main outputs are under:
echo   project\profiles\seg4s\output\logs\
goto :end

:fail
echo.
echo Failed with exit code %errorlevel%.

:end
pause
endlocal
