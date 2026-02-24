#!/usr/bin/env bash
# run_tests.sh – Read-only environment test runner.
# This script does NOT install, upgrade, or remove any packages.
# It only verifies that the environment already has the expected setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Environment Compatibility Test Runner"
echo "============================================"
echo ""
echo "NOTE: This script is read-only – it will NOT"
echo "install or modify any packages or system state."
echo ""

# ---- Check Python ---------------------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found on PATH." >&2
    exit 1
fi
echo "[1/2] Python: $($PYTHON --version 2>&1)"
echo ""

# ---- Verify pytest is available -------------------------------------------
if ! "$PYTHON" -c "import pytest" &>/dev/null; then
    echo "ERROR: pytest is not installed in the current environment." >&2
    echo "       Install it before running this script (pip install pytest)." >&2
    exit 1
fi

# ---- Run tests -------------------------------------------------------------
echo "[2/2] Running tests …"
echo ""
$PYTHON -m pytest tests/ -v --tb=short "$@"
