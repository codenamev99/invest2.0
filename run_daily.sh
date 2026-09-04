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
TICKERS_FILE="$PROJECT_DIR/us_tickers.csv"
RESULTS_FILE="$PROJECT_DIR/results.xlsx"
ROOT_DATA="$PROJECT_DIR/data 2/daily/us"
BENCHMARK="SPY.US"
# Which listing venues to screen: "us" (NYSE + NASDAQ), "nyse", "nasdaq" or
# "all". The CI config for each host pins its own value, so the same code
# screens a different universe on each remote without the source diverging.
SCREEN_UNIVERSE="${SCREEN_UNIVERSE:-us}"
case "$SCREEN_UNIVERSE" in
  nyse)   TICKER_DIRS=("$ROOT_DATA/nyse stocks") ;;
  nasdaq) TICKER_DIRS=("$ROOT_DATA/nasdaq stocks") ;;
  *)      TICKER_DIRS=("$ROOT_DATA/nyse stocks" "$ROOT_DATA/nasdaq stocks") ;;
esac
POLYGON_BOOTSTRAP_YEARS="${POLYGON_BOOTSTRAP_YEARS:-2}"
POLYGON_BACKFILL_DAYS="${POLYGON_BACKFILL_DAYS:-60}"

# Existing terminals may not inherit keys set later through macOS launchctl.
# Pull the local GUI/user environment key when the shell variable is empty.
if [ -z "${POLYGON_API_KEY:-}" ] && command -v launchctl >/dev/null 2>&1; then
  POLYGON_API_KEY="$(launchctl getenv POLYGON_API_KEY || true)"
  export POLYGON_API_KEY
fi

# -------- Daily Steps --------
if [ -n "${POLYGON_API_KEY:-}" ]; then
  if [ ! -d "$ROOT_DATA/nyse stocks" ]; then
    echo "Stock data folders missing; bootstrapping ${POLYGON_BOOTSTRAP_YEARS} years from Polygon."
    $PYTHON_CMD refresh_polygon_daily.py --bootstrap --include-today --root "$ROOT_DATA" \
      --bootstrap-years "$POLYGON_BOOTSTRAP_YEARS" --bootstrap-universe "$SCREEN_UNIVERSE"
  elif [ "$SCREEN_UNIVERSE" != "nyse" ] && [ ! -d "$ROOT_DATA/nasdaq stocks" ]; then
    # The daily backfill only touches symbols that already have a file, so it
    # cannot introduce NASDAQ on its own. Keep screening NYSE rather than firing
    # a bootstrap that would abort on the existing data, and say what is needed.
    echo "NOTE: no '$ROOT_DATA/nasdaq stocks' folder; screening NYSE only."
    echo "      To add NASDAQ history, run once:"
    echo "        POLYGON_RATE_LIMIT_SLEEP=0 $PYTHON_CMD refresh_polygon_daily.py \\"
    echo "          --bootstrap --replace-existing --include-today --root \"$ROOT_DATA\""
  fi
  $PYTHON_CMD refresh_polygon_daily.py --include-today --ensure-benchmark-history-days 400 --root "$ROOT_DATA" --backfill-days "$POLYGON_BACKFILL_DAYS"
elif [ -n "$STOOQ_SRC" ]; then
  $PYTHON_CMD refresh_stooq_dump.py --src "$STOOQ_SRC" --dest "$DATA_DEST" --mode "$STOOQ_MODE"
else
  echo "Skipping data refresh. Set POLYGON_API_KEY or STOOQ_SRC in run_daily.sh to enable."
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ] && [ -f "requirements.txt" ]; then
  $PYTHON_CMD -m pip install -r "requirements.txt"
fi

echo "Screening universe: $SCREEN_UNIVERSE"
$PYTHON_CMD generate_tickers.py --dir "${TICKER_DIRS[@]}" --out "$TICKERS_FILE"

$PYTHON_CMD screen_stooq.py --run_mode all --tickers "$TICKERS_FILE" --root "$ROOT_DATA" --benchmark "$BENCHMARK" --out "$RESULTS_FILE"

echo "Done."
