@echo off
setlocal

cd /d "%~dp0.."

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "START_TIME=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"`) do set "START_TS=%%i"

echo [msrvtt500_seg4s_test] Ingest first 500 labeled videos
echo [msrvtt500_seg4s_test] Profile: msrvtt500_seg4s_test
echo [msrvtt500_seg4s_test] Video dir: D:\zcx\chinese_clip\data\data1\labeled_0_7009\videos
echo [msrvtt500_seg4s_test] Start time: %START_TIME%
echo.

python -u -m project.src.video_processing.minimal_pipeline ^
  --profile msrvtt500_seg4s_test ^
  --video-dir D:\zcx\chinese_clip\data\data1\labeled_0_7009\videos ^
  --device cpu ^
  --video-workers 2 ^
  --num-workers 8 ^
  --limit 500

set "INGEST_EXIT_CODE=%errorlevel%"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "END_TIME=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"`) do set "END_TS=%%i"
set /a ELAPSED_SECONDS=%END_TS%-%START_TS%

echo.
echo [msrvtt500_seg4s_test] End time: %END_TIME%
echo [msrvtt500_seg4s_test] Elapsed seconds: %ELAPSED_SECONDS%
echo Finished with exit code %INGEST_EXIT_CODE%.
pause
endlocal
