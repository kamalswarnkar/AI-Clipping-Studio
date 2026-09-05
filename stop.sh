#!/usr/bin/env bash
# Stop AI Clipping Studio. Only needed if the run.sh terminal is gone;
# otherwise just press Ctrl+C in it. Add --ollama to stop Ollama too.
cd "$(dirname "$0")"

PY="venv/Scripts/python.exe"
[ -f "$PY" ] || PY="venv/bin/python"
[ -f "$PY" ] || PY="$(command -v python3 || command -v python)"

exec "$PY" scripts/stop.py "$@"
