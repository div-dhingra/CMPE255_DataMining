#!/usr/bin/env bash
# ==============================================================================
# CMPE 255 - Assignment 2: Master Live Server Launcher
# Runs all 5 projects concurrently on distinct ports (8001 - 8005)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_DS_ENV="$SCRIPT_DIR/../proj_3/backend/.venv/bin/python"
PY_FULLSTACK="$SCRIPT_DIR/fullstack-test/venv/bin/python"

if [ ! -f "$PY_DS_ENV" ]; then
    echo "Warning: DS venv not found at $PY_DS_ENV, falling back to python3"
    PY_DS_ENV="python3"
fi

if [ ! -f "$PY_FULLSTACK" ]; then
    PY_FULLSTACK="$PY_DS_ENV"
fi

export PYTHONNOUSERSITE=1

PIDS=()

cleanup() {
    echo ""
    echo "=================================================="
    echo "🛑 Shutting down all 5 running servers..."
    echo "=================================================="
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "✅ All servers stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "======================================================================"
echo "  🚀 CMPE 255 - ASSIGNMENT 2: MULTI-PROJECT LIVE SERVER LAUNCHER"
echo "======================================================================"
echo ""

# 1. Project 1: Fullstack Todo App on Port 8001
echo "Starting [Project 1] Fullstack Dynamic Todo App on Port 8001..."
(
    cd "$SCRIPT_DIR/fullstack-test"
    PORT=8001 "$PY_FULLSTACK" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
) > "$SCRIPT_DIR/project_1.log" 2>&1 &
PIDS+=($!)

# 2. Project 2: NYC Taxi DS on Port 8002
echo "Starting [Project 2] NYC Taxi DS & Interactive Map on Port 8002..."
(
    cd "$SCRIPT_DIR/nyc-taxi-ds"
    PORT=8002 "$PY_DS_ENV" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8002
) > "$SCRIPT_DIR/project_2.log" 2>&1 &
PIDS+=($!)

# 3. Project 3: LLM Chatbot & Autoresearch on Port 8003
echo "Starting [Project 3] SOTA LLM Chatbot & Autoresearch Studio on Port 8003..."
(
    cd "$SCRIPT_DIR/llm-chatbot-autoresearch"
    PORT=8003 "$PY_DS_ENV" -m uvicorn src.api.app:app --host 127.0.0.1 --port 8003
) > "$SCRIPT_DIR/project_3.log" 2>&1 &
PIDS+=($!)

# 4. Project 4: Associative Pattern Mining on Port 8004
echo "Starting [Project 4] Associative Pattern Mining & Rule Graph on Port 8004..."
(
    cd "$SCRIPT_DIR/associative-pattern-mining"
    PORT=8004 "$PY_DS_ENV" -m uvicorn backend.src.main:app --host 127.0.0.1 --port 8004
) > "$SCRIPT_DIR/project_4.log" 2>&1 &
PIDS+=($!)

# 5. Project 5: Agent ML & Analytics Skills Demonstration on Port 8005
echo "Starting [Project 5] Agent ML & Analytics Skills Demonstration on Port 8005..."
(
    cd "$SCRIPT_DIR/agent-skills-demonstration"
    PORT=8005 "$PY_DS_ENV" -m uvicorn src.server.app:app --host 127.0.0.1 --port 8005
) > "$SCRIPT_DIR/project_5.log" 2>&1 &
PIDS+=($!)

sleep 2

echo ""
echo "======================================================================"
echo "  🎉 ALL 5 SERVERS ARE NOW RUNNING LIVE!"
echo "======================================================================"
echo ""
echo "  👉 Project 1 (Fullstack Todo App):"
echo "     • Live App:  http://127.0.0.1:8001"
echo "     • Swagger:   http://127.0.0.1:8001/docs"
echo ""
echo "  👉 Project 2 (NYC Taxi DS & Leaflet Map):"
echo "     • Live Map:  http://127.0.0.1:8002"
echo "     • Swagger:   http://127.0.0.1:8002/docs"
echo ""
echo "  👉 Project 3 (SOTA LLM Chatbot & Autoresearch):"
echo "     • Dashboard: http://127.0.0.1:8003"
echo "     • Swagger:   http://127.0.0.1:8003/docs"
echo ""
echo "  👉 Project 4 (Associative Pattern Mining & Rule Graph):"
echo "     • Dashboard: http://127.0.0.1:8004"
echo "     • Swagger:   http://127.0.0.1:8004/docs"
echo ""
echo "  👉 Project 5 (Agent ML & Analytics Skills Engine):"
echo "     • Dashboard: http://127.0.0.1:8005"
echo "     • Swagger:   http://127.0.0.1:8005/docs"
echo ""
echo "----------------------------------------------------------------------"
echo "  Press Ctrl+C at any time to gracefully stop all 5 servers."
echo "======================================================================"
echo ""

wait
