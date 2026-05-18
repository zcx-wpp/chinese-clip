@echo off
setlocal

cd /d "%~dp0.."

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "START_TIME=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"`) do set "START_TS=%%i"

echo [seg4s] Incremental ingest from data/videos
echo [seg4s] Done videos in metadata.db will be skipped automatically.
echo [seg4s] Profile: seg4s
echo [seg4s] Start time: %START_TIME%
echo.

python -u -m project.src.video_processing.minimal_pipeline ^
  --profile seg4s ^
  --video-dir data/videos ^
  --device cpu ^
  --video-workers 2 ^
  --num-workers 4

set "INGEST_EXIT_CODE=%errorlevel%"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "END_TIME=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"`) do set "END_TS=%%i"
set /a ELAPSED_SECONDS=%END_TS%-%START_TS%

echo.
echo [seg4s] End time: %END_TIME%
echo [seg4s] Elapsed seconds: %ELAPSED_SECONDS%
echo Finished with exit code %INGEST_EXIT_CODE%.
pause
endlocal
