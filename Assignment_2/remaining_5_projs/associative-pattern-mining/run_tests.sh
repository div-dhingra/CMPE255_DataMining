#!/usr/bin/env bash
# ==============================================================================
# Associative Pattern Mining & Autoresearch Engine - Comprehensive Test Runner
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "  CRISP-DM Associative Pattern Mining & Autoresearch: Test Runner"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"

# Determine Python interpreter
PYTHON_BIN=""
if [ -x "${SCRIPT_DIR}/../proj_3/backend/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/../proj_3/backend/.venv/bin/python"
elif [ -x "/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python" ]; then
    PYTHON_BIN="/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python interpreter not found!"
    exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Execute pytest
echo "----------------------------------------------------------------------"
echo "Running pytest test suite across algorithms, metrics, autoresearch & API..."
echo "----------------------------------------------------------------------"

${PYTHON_BIN} -m pytest tests/ -v --tb=short

TEST_EXIT_CODE=$?

echo "----------------------------------------------------------------------"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✅ [SUCCESS] All test suites passed cleanly!"
else
    echo "❌ [FAILURE] Test suite encountered failures (Exit code: ${TEST_EXIT_CODE})."
fi
echo "======================================================================"

exit ${TEST_EXIT_CODE}
