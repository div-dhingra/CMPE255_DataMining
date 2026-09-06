#!/usr/bin/env bash
set -e

echo "=================================================="
echo "    Running NYC Taxi DS Test Suite (CRISP-DM)     "
echo "=================================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="${PROJECT_DIR}/../../proj_3/backend/.venv/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Warning: Virtualenv python at $PYTHON_EXEC not found, falling back to system python3"
    PYTHON_EXEC="python3"
fi

cd "$PROJECT_DIR"
"$PYTHON_EXEC" -m pytest -v tests/

echo "=================================================="
echo "          All tests passed successfully!          "
echo "=================================================="
