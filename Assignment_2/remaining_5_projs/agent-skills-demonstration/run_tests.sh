#!/usr/bin/env bash
# ==============================================================================
# Agent Skills Demonstration: Full Test Suite Runner
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "  Project 5: Agent Skills Demonstration - Automated Test Suite"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"

# Locate Python binary
PYTHON_BIN="python3"
if [ -x "/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python" ]; then
    PYTHON_BIN="/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -x "/usr/local/bin/python3" ]; then
    PYTHON_BIN="/usr/local/bin/python3"
fi

echo "Python interpreter: ${PYTHON_BIN}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export MPLCONFIGDIR="${SCRIPT_DIR}/.mplcache"
mkdir -p "${MPLCONFIGDIR}"

TARGET_TESTS="tests/"
if [[ "$1" == "--agent-ml" ]]; then
    echo "Running 15 Agent-ML Skill Tests..."
    TARGET_TESTS="tests/test_agent_ml_skills.py"
elif [[ "$1" == "--data-analytics" ]]; then
    echo "Running 31 Data-Analytics Skill Tests..."
    TARGET_TESTS="tests/test_data_analytics_skills.py"
elif [[ "$1" == "--crisp" ]]; then
    echo "Running CRISP-DM 6-Phase Pipeline Tests..."
    TARGET_TESTS="tests/test_crisp_dm_pipeline.py"
elif [[ "$1" == "--api" ]]; then
    echo "Running FastAPI REST Endpoints Tests..."
    TARGET_TESTS="tests/test_api_endpoints.py"
else
    echo "Running Full Test Suite (Agent-ML, Data-Analytics, CRISP-DM, and FastAPI REST Endpoints)..."
fi

echo "----------------------------------------------------------------------"
${PYTHON_BIN} -m pytest ${TARGET_TESTS} -v --tb=short

TEST_EXIT_CODE=$?

echo "----------------------------------------------------------------------"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✅ [SUCCESS] All automated test suites passed cleanly!"
else
    echo "❌ [FAILURE] Test suite encountered failures (Exit code: ${TEST_EXIT_CODE})."
fi
echo "======================================================================"

exit ${TEST_EXIT_CODE}
