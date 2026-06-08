@echo off
setlocal

set ROOT=D:\field_batch_output_compressed_air
set PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe
set TRAIN_SCRIPT=%ROOT%\scripts\train\train_fno_fullfield_maxwell_pycharm.py
set LOG_DIR=%ROOT%\logs\training

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i
set LOG_FILE=%LOG_DIR%\train_fullfield_%TS%.log

echo Starting training...
echo Log: %LOG_FILE%

"%PYTHON_EXE%" -u "%TRAIN_SCRIPT%" >> "%LOG_FILE%" 2>&1

endlocal
