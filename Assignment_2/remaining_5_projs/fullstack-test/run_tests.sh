#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "🧪 Running TodoPro Automated Test Suite"
echo "=================================================="

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

echo "Using Python: $PY_BIN"
$PY_BIN -m pytest backend/tests/ -v --disable-warnings

echo ""
echo "✅ All tests completed successfully!"
