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
# Preferred: set POLYGON_API_KEY in your shell or scheduler to auto-refresh from Polygon.
# Optional fallback: set this to your Stooq download path if you still want manual Stooq refresh.
STOOQ_SRC=""
STOOQ_MODE="move"
DATA_DEST="$PROJECT_DIR/data 2"
TICKERS_FILE="$PROJECT_DIR/nyse_tickers.csv"
RESULTS_FILE="$PROJECT_DIR/results.xlsx"
ROOT_DATA="$PROJECT_DIR/data 2/daily/us"
BENCHMARK="SPY.US"
POLYGON_BOOTSTRAP_YEARS="${POLYGON_BOOTSTRAP_YEARS:-2}"
POLYGON_BACKFILL_DAYS="${POLYGON_BACKFILL_DAYS:-30}"

# Existing terminals may not inherit keys set later through macOS launchctl.
# Pull the local GUI/user environment key when the shell variable is empty.
if [ -z "${POLYGON_API_KEY:-}" ] && command -v launchctl >/dev/null 2>&1; then
  POLYGON_API_KEY="$(launchctl getenv POLYGON_API_KEY || true)"
  export POLYGON_API_KEY
fi

# -------- Daily Steps --------
if [ -n "${POLYGON_API_KEY:-}" ]; then
  if [ ! -d "$ROOT_DATA/nyse stocks" ]; then
    echo "NYSE data folder missing; bootstrapping ${POLYGON_BOOTSTRAP_YEARS} years from Polygon."
    $PYTHON_CMD refresh_polygon_daily.py --bootstrap --root "$ROOT_DATA" --bootstrap-years "$POLYGON_BOOTSTRAP_YEARS"
  fi
  $PYTHON_CMD refresh_polygon_daily.py --root "$ROOT_DATA" --backfill-days "$POLYGON_BACKFILL_DAYS"
elif [ -n "$STOOQ_SRC" ]; then
  $PYTHON_CMD refresh_stooq_dump.py --src "$STOOQ_SRC" --dest "$DATA_DEST" --mode "$STOOQ_MODE"
else
  echo "Skipping data refresh. Set POLYGON_API_KEY or STOOQ_SRC in run_daily.sh to enable."
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ] && [ -f "requirements.txt" ]; then
  $PYTHON_CMD -m pip install -r "requirements.txt"
fi

$PYTHON_CMD generate_tickers.py --dir "$ROOT_DATA/nyse stocks" --out "$TICKERS_FILE"

$PYTHON_CMD screen_stooq.py --run_mode all --tickers "$TICKERS_FILE" --root "$ROOT_DATA" --benchmark "$BENCHMARK" --out "$RESULTS_FILE"

echo "Done."
