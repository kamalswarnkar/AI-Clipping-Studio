@echo off
REM Start AI Clipping Studio. Builds the UI on first run.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Run:
    echo   python -m venv venv
    echo   venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    exit /b 1
)

if not exist "frontend\dist\index.html" (
    echo Building the interface...
    pushd frontend
    call npm install
    call npm run build
    popd
)

echo.
echo   AI Clipping Studio  ->  http://127.0.0.1:8000
echo.
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
