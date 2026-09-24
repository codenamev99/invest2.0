#!/usr/bin/env bash
# NOTE: Keep LF line endings so bash runs correctly.
#
# GitLab-only, ~9:35am ET step. Checks yesterday's (and a rolling window of
# recent) Alpaca paper entries from alpaca_pm_entries.py: once an entry has
# actually filled, submits a real +2%/-1% OCO exit mirroring PM Simulation's
# own target/stop, and force-closes after the same 5-trading-day timeout the
# simulation uses. Runs after 9:30am ET because Alpaca doesn't accept OCO/stop
# orders during extended hours. No price refresh needed -- trading-day
# counting only reads the daily bars data 2/ already has cached.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ] && [ -f "requirements.txt" ]; then
  $PYTHON_CMD -m pip install -r "requirements.txt"
fi

$PYTHON_CMD alpaca_pm_exits.py --root "$PROJECT_DIR/data 2/daily/us"

echo "Done."
