#!/usr/bin/env bash
# Start AI Clipping Studio. Builds the UI on first run.
set -e
cd "$(dirname "$0")"

PY="venv/Scripts/python.exe"
[ -f "$PY" ] || PY="venv/bin/python"

if [ ! -f "$PY" ]; then
    echo "Virtual environment not found. Run:"
    echo "  python -m venv venv"
    echo "  $PY -m pip install -r backend/requirements.txt"
    exit 1
fi

if [ ! -f "frontend/dist/index.html" ]; then
    echo "Building the interface..."
    (cd frontend && npm install && npm run build)
fi

echo
echo "  AI Clipping Studio  ->  http://127.0.0.1:8000"
echo
exec "$PY" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
