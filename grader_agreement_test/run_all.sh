#!/usr/bin/env bash
# Run synthetic data collection, grading, and agreement analysis in order.
# Optional: pass extra args only to collect_test_data.py (e.g. -o /path/out.json).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "==> collect_test_data.py" >&2
python3 collect_test_data.py "$@"
echo "==> run_grading.py" >&2
python3 run_grading.py
echo "==> analyze_agreement.py" >&2
python3 analyze_agreement.py
