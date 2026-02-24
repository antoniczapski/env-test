#!/usr/bin/env bash
# run_tests.sh – Install test deps (if needed) and run the environment test suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Environment Compatibility Test Runner"
echo "============================================"
echo ""

# ---- Check Python ---------------------------------------------------------
PYTHON="${PYTHON:-python3}"
echo "[1/2] Python: $($PYTHON --version 2>&1)"
echo ""

# ---- Run tests -------------------------------------------------------------
echo "[2/2] Running tests …"
echo ""
$PYTHON -m pytest tests/ -v --tb=short "$@"
