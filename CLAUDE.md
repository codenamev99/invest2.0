# invest2.0

Daily NYSE stock screener and paper-trading simulator. Python script suite (no
package manifest, no web framework) that runs on a GitHub Actions cron,
refreshes price data, screens tickers, simulates trades, and emails a summary.

## Pipeline

`run_daily.sh` (macOS/Linux) / `run_daily.bat` (Windows) chain these steps:

1. **Refresh price data** into `data 2/daily/us/{nyse stocks,etfs}/` (Stooq-format
   per-ticker daily OHLCV `.txt` files):
   - Preferred: `refresh_polygon_daily.py` pulls from the Polygon.io API
     (`POLYGON_API_KEY` env var). Bootstraps full history if `data 2` is
     missing, otherwise does an incremental backfill.
   - Fallback: `refresh_stooq_dump.py` imports a manually downloaded Stooq
     bulk dump (`STOOQ_SRC` env var / `--src` flag) — used only if
     `POLYGON_API_KEY` is unset.
2. **Rebuild ticker universe**: `generate_tickers.py` scans
   `data 2/daily/us/nyse stocks/*.txt` → writes `nyse_tickers.csv`.
3. **Screen + simulate**: `screen_stooq.py` (~3600 lines, the core engine) —
   computes RSI/MACD/ATR/beta-vs-SPY, screens for setups, runs a simulated
   trade tracker (+2% target / -1% stop-loss, using Polygon 1-minute bars
   intraday when available else daily OHLC), applies a SPY-based market
   regime gate before allowing new simulated entries, and tracks upcoming
   IPOs (60d) / earnings (14d, via the public Nasdaq calendar API). Writes
   `results.xlsx` (sheets: Single Tickers, Simulation, AM Simulation,
   Investment Dashboard, Upcoming IPOs/Earnings, Top 10 OHLC Tracking, Daily
   Runs, plus one dated sheet per run day).
4. **Email report**: `send_daily_email.py` reads `results.xlsx` and emails a
   formatted summary via SMTP.

## Running

```bash
./run_daily.sh                 # full pipeline, uses env vars below
python screen_stooq.py --tickers nyse_tickers.csv --root "data 2/daily/us" \
    --benchmark SPY.US --run_mode all --out results.xlsx
python screen_stooq.py --run_mode single --single_symbol AAPL --root "data 2/daily/us" ...
```

`screen_stooq.py` prompts interactively for `--run_mode` if it's omitted.
Key flags: `--rsi_low/high`, `--macd_fast/slow/signal`, `--atr_period`,
`--beta_min`, `--beta_lookback`/`--beta_months`, `--market_regime_mode
aggressive|standard` (default `aggressive`: SPY above 50d & 200d MA, 20d MA
above 50d MA, positive 5d return; `standard`: SPY above 50d MA and 5d return
better than -2%).

## Environment variables

- `POLYGON_API_KEY` — required for Polygon refresh + intraday target/stop
  simulation; never commit this, set it in the shell/scheduler.
- `POLYGON_BOOTSTRAP_YEARS` (default 2), `POLYGON_BACKFILL_DAYS` (default 60),
  `POLYGON_RATE_LIMIT_SLEEP`
- `STOOQ_SRC`, `STOOQ_MODE` (`copy`|`move`) — Stooq fallback, only used if
  `POLYGON_API_KEY` is unset
- `SMTP_HOST/PORT/USERNAME/PASSWORD`, `EMAIL_FROM`, `EMAIL_TO`,
  `EMAIL_SUBJECT_PREFIX`, `EMAIL_ATTACH_RESULTS` — for `send_daily_email.py`

## Automation

`.github/workflows/daily-screener.yml` runs on cron `0 18 * * 1-5` (6pm
America/New_York, weekdays): checkout → Python 3.12 setup → install deps →
restore `data 2` cache → run `run_daily.sh` → email results → commit the
updated `results.xlsx` back to the repo as `github-actions[bot]`.

## Notes

- `data 2/` and `results.xlsx` are treated as generated/cached state, not
  hand-edited source — they're rewritten by every daily run and committed by
  CI. `nyse_tickers.csv` and `results.csv` are gitignored (regenerated
  locally) despite `nyse_tickers.csv` currently being tracked — don't add
  logic that depends on either being fresh in a clean checkout.
- No automated test suite exists. Sanity-check changes to `screen_stooq.py`
  by running `--run_mode single --single_symbol <TICKER>` against existing
  `data 2` before running the full universe.
- No live broker integration — the "Simulation"/"AM Simulation" sheets are
  paper-trading only.
