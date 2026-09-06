#!/usr/bin/env bash
# ==============================================================================
# SOTA LLM Chatbot & Autoresearch Engine - Test Runner Script
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/tests"

echo "======================================================================"
echo "  SOTA LLM & Autoresearch Engine: Comprehensive Test Suite Runner"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"

# Locate suitable Python binary
PYTHON_BIN="python3"
PROJ3_VENV="/Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/proj_3/backend/.venv/bin/python"
RELATIVE_VENV="${SCRIPT_DIR}/../../proj_3/backend/.venv/bin/python"

if [ -x "${PROJ3_VENV}" ]; then
    PYTHON_BIN="${PROJ3_VENV}"
elif [ -x "${RELATIVE_VENV}" ]; then
    PYTHON_BIN="${RELATIVE_VENV}"
elif [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [ -x "/usr/local/bin/python3" ]; then
    PYTHON_BIN="/usr/local/bin/python3"
fi

echo "Python interpreter: ${PYTHON_BIN}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Parse tier selection argument
TARGET_ARGS=""
if [[ "$1" == "--tier1" ]]; then
    echo "Running Tier 1: SOTA Transformer Primitives (RoPE, SwiGLU, RMSNorm, GQA, KV-Cache)..."
    TARGET_ARGS="${TEST_DIR}/test_tier1_primitives.py"
elif [[ "$1" == "--tier2" ]]; then
    echo "Running Tier 2: Tokenizer, Sampling Logic & Chat Generation Engine..."
    TARGET_ARGS="${TEST_DIR}/test_tier2_generation.py"
elif [[ "$1" == "--tier3" ]]; then
    echo "Running Tier 3: Autoresearch Hill-Climbing Optimization & Ledger..."
    TARGET_ARGS="${TEST_DIR}/test_tier3_autoresearch.py"
elif [[ "$1" == "--tier4" ]]; then
    echo "Running Tier 4: FastAPI E2E Endpoints & Dashboard Serving..."
    TARGET_ARGS="${TEST_DIR}/test_tier4_api_e2e.py"
else
    echo "Running Full Multi-Tier Test Suite (Tiers 1, 2, 3, and 4)..."
    TARGET_ARGS="${TEST_DIR}"
fi

echo "----------------------------------------------------------------------"
${PYTHON_BIN} -m pytest ${TARGET_ARGS} -v --tb=short

TEST_EXIT_CODE=$?

echo "----------------------------------------------------------------------"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "✅ [SUCCESS] All SOTA LLM & Autoresearch test suites passed cleanly!"
else
    echo "❌ [FAILURE] Test suite encountered failures (Exit code: ${TEST_EXIT_CODE})."
fi
echo "======================================================================"

exit ${TEST_EXIT_CODE}
