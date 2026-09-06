#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${PORT:-8000}
HOST=${HOST:-"127.0.0.1"}

# Locate Python binary
if [ -f "./venv/bin/python" ]; then
    PY_BIN="./venv/bin/python"
elif [ -f "/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python" ]; then
    PY_BIN="/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python"
else
    PY_BIN="python3"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=================================================="
echo "🚀 Starting TodoPro Fullstack Application"
echo "🌐 URL: http://${HOST}:${PORT}"
echo "📖 API Docs: http://${HOST}:${PORT}/docs"
echo "=================================================="

exec $PY_BIN -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --reload
