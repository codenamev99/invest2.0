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
   `results.xlsx` (sheets: How It Works, Single Tickers, Simulation, AM
   Simulation, PM Simulation, Commit Summary, Investment Dashboard, Upcoming
   IPOs/Earnings, Recent Splits (90D), Top 10 OHLC Tracking, Daily Runs, plus
   one dated sheet per run day). The three simulation sheets share one engine
   (`build_investment_simulation_rows`, switched by `entry_session`) and differ
   only in entry fill: `regular` takes the next session's 9:30am open, `am` the
   next morning's 4:00-9:29am pre-market open, `pm` the rank date's own
   4:00-8:00pm after-hours session. A `pm` entry lands on the rank date itself,
   so its exit scan starts the following trading day — that day's own high/low
   preceded the fill and would otherwise fabricate exits.

   Daily Runs' last column, `Run Finished`, is stamped by `stamp_run_finished`
   immediately before `wb.save` so it records the end of the run. `pm` entries
   price off it: the fill is the first minute bar at or after that time plus
   `PM_ENTRY_DELAY_MINUTES` (10). The bar is ten minutes in the future when the
   stamp is written, so **a day's PM row is only priced on a later run** — it
   reports `Pending` until then, which is fine because the simulation sheets are
   rebuilt from Daily Runs on every run. Rows predating the column fall back to
   the after-hours session open, then to the rank date's close.

   PM Simulation additionally carries a `4M Daily Variance` column (mean of
   `(high - low) / low` over `VARIANCE_LOOKBACK_MONTHS`, as of each row's rank
   date) and a movement test: an entry must sit at least `PM_MIN_MOVE_FROM_CLOSE`
   (2%) away from the rank date's close, **in either direction, or it is excluded
   from the totals** — listed with a reason, shares and investment zeroed, the
   same treatment a blocked market regime gets. Note the direction: rows that
   moved *less* than the threshold are the excluded ones, so any row that fell
   back to the rank date's close scores a 0% move and is always excluded.
   Totals are a `SUMIF(..., "Good*", ...)` over the condition column, so any
   reason string that does not start with `Good` drops the row from the totals.
   "How It Works" is a non-technical guide to the pipeline, rebuilt each run
   from the live thresholds; adding a sheet means registering its name in
   `PROTECTED_SHEET_NAMES` here *and* in `send_daily_email.py`, which
   otherwise treats an unrecognized sheet as the ranked stock list.
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

`.github/workflows/daily-screener.yml` runs on cron `30 16 * * 1-5` (4:30pm
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
