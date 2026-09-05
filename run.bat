@echo off
REM Start AI Clipping Studio: Ollama, the interface build, the server, the browser.
REM Double-click this file, or run it from a terminal. Ctrl+C stops everything.
setlocal
cd /d "%~dp0"

set "PY=venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\launch.py %*
set "CODE=%ERRORLEVEL%"

REM Exit code 3 means the launcher printed something you need to read, so hold
REM the window open. Any other code means it was simply stopped: close quietly.
if "%CODE%"=="3" (
    echo.
    pause
)
exit /b %CODE%
