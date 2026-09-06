#!/usr/bin/env bash
# ==============================================================================
# Helper script to terminate any servers running on ports 8001 through 8005
# ==============================================================================

echo "Checking and freeing ports 8001 - 8005..."
for PORT in 8001 8002 8003 8004 8005; do
    PID=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Stopping process $PID on port $PORT..."
        kill -9 $PID 2>/dev/null || true
    else
        echo "Port $PORT is clean."
    fi
done
echo "All ports 8001-8005 are now free."
