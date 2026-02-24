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
echo "[1/3] Python: $($PYTHON --version 2>&1)"
echo ""

# ---- Install test dependencies (optional) ---------------------------------
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    echo "[2/3] Installing test dependencies …"
    $PYTHON -m pip install --quiet -r requirements-test.txt
    echo "      Done."
else
    echo "[2/3] Skipping dependency install (SKIP_INSTALL=1)"
fi
echo ""

# ---- Run tests -------------------------------------------------------------
echo "[3/3] Running tests …"
echo ""
$PYTHON -m pytest tests/ -v --tb=short "$@"
