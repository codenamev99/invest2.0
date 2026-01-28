#!/usr/bin/env bash
# NOTE: Keep LF line endings so bash runs correctly.
set -euo pipefail

# Run from this script's folder
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Pick Python command: prefer python3
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

# Optional: activate venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# -------- Configuration --------
# Set this to your Stooq download path if you want auto refresh
STOOQ_SRC="/Users/v/Downloads/data"
STOOQ_MODE="move"
DATA_DEST="$PROJECT_DIR/data 2"
TICKERS_FILE="$PROJECT_DIR/nyse_tickers.csv"
RESULTS_FILE="$PROJECT_DIR/results.xlsx"
ROOT_DATA="$PROJECT_DIR/data 2/daily/us"
BENCHMARK="SPY.US"

# -------- Daily Steps --------
if [ -n "$STOOQ_SRC" ]; then
  $PYTHON_CMD refresh_stooq_dump.py --src "$STOOQ_SRC" --dest "$DATA_DEST" --mode "$STOOQ_MODE"
else
  echo "Skipping data refresh. Set STOOQ_SRC in run_daily.sh to enable."
fi

if [ -f "requirements.txt" ]; then
  $PYTHON_CMD -m pip install -r "requirements.txt"
fi

$PYTHON_CMD generate_tickers.py --dir "$ROOT_DATA/nyse stocks" --out "$TICKERS_FILE"

$PYTHON_CMD screen_stooq.py --tickers "$TICKERS_FILE" --root "$ROOT_DATA" --benchmark "$BENCHMARK" --out "$RESULTS_FILE"

echo "Done."
