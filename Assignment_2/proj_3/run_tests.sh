#!/usr/bin/env bash
# ==============================================================================
# CRISP-DM Autonomous Clustering & Autoresearch Engine - Test Runner
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"
TEST_DIR="${BACKEND_DIR}/tests"

echo "======================================================================"
echo "  CRISP-DM Clustering & Autoresearch Engine: E2E Test Suite Runner"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"
# Locate suitable Python binary
PYTHON_BIN="python3"
if [ -x "/usr/local/bin/python3" ]; then
    PYTHON_BIN="/usr/local/bin/python3"
elif [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [ -x "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" ]; then
    PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
fi

echo "Python interpreter: ${PYTHON_BIN}"
export PYTHONPATH="${BACKEND_DIR}:${BACKEND_DIR}/src:${PYTHONPATH}"

# Parse optional arguments
TARGET_ARGS=""
if [[ "$1" == "--tier1" ]]; then
    echo "Running Tier 1: Feature Coverage Tests (34 Features)..."
    TARGET_ARGS="${TEST_DIR}/e2e/test_tier1_features.py"
elif [[ "$1" == "--tier2" ]]; then
    echo "Running Tier 2: Boundary & Corner Cases..."
    TARGET_ARGS="${TEST_DIR}/e2e/test_tier2_boundaries.py"
elif [[ "$1" == "--tier3" ]]; then
    echo "Running Tier 3: Cross-Feature Pairwise Interactions..."
    TARGET_ARGS="${TEST_DIR}/e2e/test_tier3_interactions.py"
elif [[ "$1" == "--tier4" ]]; then
    echo "Running Tier 4: Real-World Application Scenarios..."
    TARGET_ARGS="${TEST_DIR}/e2e/test_tier4_applications.py"
else
    echo "Running Full Test Suite (Tiers 1, 2, 3, and 4)..."
    TARGET_ARGS="${TEST_DIR}/e2e"
fi

echo "----------------------------------------------------------------------"
${PYTHON_BIN} -m pytest ${TARGET_ARGS} -v --tb=short

TEST_EXIT_CODE=$?

echo "----------------------------------------------------------------------"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✅ [SUCCESS] All E2E test suites passed cleanly!"
else
    echo "❌ [FAILURE] Test suite encountered failures (Exit code: ${TEST_EXIT_CODE})."
fi
echo "======================================================================"

exit ${TEST_EXIT_CODE}
