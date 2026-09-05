@echo off
REM Stop AI Clipping Studio. Only needed if the run.bat window is gone;
REM otherwise just press Ctrl+C in it. Add --ollama to stop Ollama too.
setlocal
cd /d "%~dp0"

set "PY=venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\stop.py %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
    echo.
    pause
)
exit /b %CODE%
