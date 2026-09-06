#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="${PROJECT_DIR}/../../proj_3/backend/.venv/bin/python"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Warning: Virtualenv python at $PYTHON_EXEC not found, falling back to system python3"
    PYTHON_EXEC="python3"
fi

cd "$PROJECT_DIR"
echo "=================================================="
echo "    Starting NYC Taxi ML Server on http://localhost:${PORT}"
echo "    Interactive Map: http://localhost:${PORT}/"
echo "    API Documentation: http://localhost:${PORT}/docs"
echo "=================================================="

exec "$PYTHON_EXEC" -m uvicorn src.api.app:app --host "$HOST" --port "$PORT" --reload
