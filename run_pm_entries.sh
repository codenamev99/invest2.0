#!/usr/bin/env bash
# NOTE: Keep LF line endings so bash runs correctly.
#
# GitLab-only, same-day PM entry step. Runs ~4:15pm ET, while the 4:00-8:00pm
# after-hours session is still open, so real Alpaca limit orders have a
# session left to execute in -- unlike the 8:15pm full pipeline (run_daily.sh),
# which deliberately runs after that window closes (see CLAUDE.md's PM
# Simulation section for why). This does its own incremental price refresh
# and a rank-only screen, then exits without touching results.xlsx or Daily
# Runs -- see screen_stooq.py's --rank_only and alpaca_pm_entries.py.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python"
fi

ROOT_DATA="$PROJECT_DIR/data 2/daily/us"
BENCHMARK="SPY.US"
SCREEN_UNIVERSE="${SCREEN_UNIVERSE:-us}"
case "$SCREEN_UNIVERSE" in
  nyse)   TICKER_DIRS=("$ROOT_DATA/nyse stocks") ;;
  nasdaq) TICKER_DIRS=("$ROOT_DATA/nasdaq stocks") ;;
  *)      TICKER_DIRS=("$ROOT_DATA/nyse stocks" "$ROOT_DATA/nasdaq stocks") ;;
esac

if [ -z "${POLYGON_API_KEY:-}" ]; then
  echo "POLYGON_API_KEY is required for the entry step (no Stooq fallback here)." >&2
  exit 1
fi

# Only today's bar is needed -- data 2 already carries history through
# yesterday via the same cache the evening job restores.
$PYTHON_CMD refresh_polygon_daily.py --include-today --backfill-days 1 \
  --ensure-benchmark-history-days 400 --root "$ROOT_DATA"

# Fail loudly instead of silently ranking on yesterday's close if Polygon
# hasn't published today's grouped-daily bar yet by trigger time -- there is
# no wait/poll for this upstream, so this is the safety net until 4:15pm ET
# is confirmed to be late enough in practice.
$PYTHON_CMD -c "
import sys
from datetime import date
with open('$ROOT_DATA/etfs/spy.us.txt') as f:
    last = f.readlines()[-1]
today = date.today().strftime('%Y%m%d')
if not last.startswith(f'SPY.US,D,{today}'):
    sys.exit(f\"SPY's latest daily bar is not today's ({today}) close yet; aborting rather than rank on stale data.\")
"

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ] && [ -f "requirements.txt" ]; then
  $PYTHON_CMD -m pip install -r "requirements.txt"
fi

$PYTHON_CMD generate_tickers.py --dir "${TICKER_DIRS[@]}" --out "$PROJECT_DIR/us_tickers.csv"

$PYTHON_CMD screen_stooq.py --run_mode all --tickers "$PROJECT_DIR/us_tickers.csv" \
  --root "$ROOT_DATA" --benchmark "$BENCHMARK" \
  --rank_only --rank_only_out "$PROJECT_DIR/top10.json"

$PYTHON_CMD alpaca_pm_entries.py --top10 "$PROJECT_DIR/top10.json" --root "$ROOT_DATA"

echo "Done."
