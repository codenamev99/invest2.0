# invest2.0

Daily NYSE + NASDAQ stock screener and paper-trading simulator. Python script suite (no
package manifest, no web framework) that runs on a GitHub Actions cron,
refreshes price data, screens tickers, simulates trades, and emails a summary.

## Pipeline

`run_daily.sh` (macOS/Linux) / `run_daily.bat` (Windows) chain these steps:

1. **Refresh price data** into `data 2/daily/us/{nyse stocks,nasdaq stocks,etfs}/`
   (Stooq-format per-ticker daily OHLCV `.txt` files):
   - Preferred: `refresh_polygon_daily.py` pulls from the Polygon.io API
     (`POLYGON_API_KEY` env var). Bootstraps full history if `data 2` is
     missing, otherwise does an incremental backfill.
t.
2. **Rebuild ticker universe**: `generate_tickers.py` pools
   `data 2/daily/us/{nyse stocks,nasdaq stocks}/*.txt` → writes `us_tickers.csv`.
   `--dir` takes one or more folders; a folder that does not exist is skipped
   with a note, so a NYSE-only checkout still runs.

   The venue split comes from the bootstrap: `fetch_reference_symbols` maps each
   `primary_exchange` (`XNYS`/`XNAS`) to its folder via `EXCHANGE_DIRS`, and
   `--bootstrap-universe` (`us` default, or `nyse`/`nasdaq`/`all`) picks which
   venues to include. Only `CS`/`ADRC`/`ADRP` are kept — warrants, units, rights
   and preferred shares are not traded here. ETFs are restricted to
   `BENCHMARK_ETFS` (SPY, QQQ), the only two anything reads; fetching the full
   US ETF list instead costs ~5,400 files nothing ever opens.

   **The daily backfill cannot add a new venue.** `upsert_grouped_bars` skips
   symbols with no existing file, so switching `--bootstrap-universe` only takes
   effect on a `--bootstrap --replace-existing` rebuild. The grouped-daily
   endpoint already returns the whole US market in one call per trading day, so
   widening the universe costs disk and screening CPU but no extra API calls.
3. **Screen + simulate**: `screen_stooq.py` (~3600 lines, the core engine) —
   computes RSI/MACD/ATR/beta-vs-SPY, screens for setups, runs a simulated
   trade tracker (+2% target / -1% stop-loss, using Polygon 1-minute bars
   intraday when available else daily OHLC), applies a SPY-based market
   regime gate before allowing new simulated entries, and tracks upcoming
   IPOs (60d) / earnings (14d, via the public Nasdaq calendar API). Writes
   `results.xlsx` (sheets: How It Works, Single Tickers, Simulation, AM
   Simulation, PM Simulation, Commit Summary, Investment Dashboard, Upcoming
   IPOs/Earnings, Top 10 OHLC Tracking, Daily Runs, plus one dated sheet per
   run day). The three simulation sheets share one engine
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
   date) and a **headroom test** built on it. The strategy is a +2%/-1% scalp,
   and a PM entry fills after the close, by which point the stock may already
   have run up in extended hours. The test asks whether enough of the stock's
   normal daily range is left to still reach the target:

       run_up   = max(0, entry / rank_date_close - 1)   # downward moves cost nothing
       headroom = variance_4m - run_up
       excluded = headroom < gain_pct                   # gain_pct is the +2% target

   Excluded rows are listed with the reason, shares and investment zeroed — the
   same treatment a blocked market regime gets. Totals are a
   `SUMIF(..., "Good*", ...)` over the condition column, so any reason string
   that does not start with `Good` drops the row from the totals. The test is
   skipped when there is too little history to compute a variance.
   "How It Works" is a non-technical guide to the pipeline, rebuilt each run
   from the live thresholds; adding a sheet means registering its name in
   `PROTECTED_SHEET_NAMES` here *and* in `send_daily_email.py`, which
   otherwise treats an unrecognized sheet as the ranked stock list.
4. **Email report**: `send_daily_email.py` reads `results.xlsx` and emails a
   formatted summary via SMTP.

## Running

```bash
./run_daily.sh                 # full pipeline, uses env vars below
python screen_stooq.py --tickers us_tickers.csv --root "data 2/daily/us" \
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
  CI. `us_tickers.csv` and `results.csv` are gitignored and regenerated
  locally — don't add logic that depends on either being fresh in a clean
  checkout.
- No automated test suite exists. Sanity-check changes to `screen_stooq.py`
  by running `--run_mode single --single_symbol <TICKER>` against existing
  `data 2` before running the full universe.
- The daily backfill collects every day's grouped bars in memory first and
  writes each symbol's file **once** (`group_bars_by_path` → `flush_pending_rows`).
  The old per-day `upsert_daily_row` path rewrote a whole file to change one
  row, so 60 days x ~5,500 files meant ~330,000 read-modify-write cycles per
  run; batching made it ~85x faster. `upsert_daily_row` is still used for the
  single-date refresh, where there is nothing to batch.
- Simulation rows that already resolved are reused rather than recomputed.
  `load_settled_simulation_rows` reads back rows whose exit date precedes the
  run, and `build_investment_simulation_rows` emits them before touching a file
  or Polygon. Cohorts are never retired, so without this the per-run cost grows
  every trading day; it takes the simulation from ~1,600 Polygon requests to
  ~10. The three sessions also share one `ohlc_cache`/`intraday_cache`.
  A reused row keeps the answer it was built with, so **after changing the
  entry/exit thresholds or `--market_regime_mode`, rerun with
  `--rebuild_simulation`** or the sheets will mix old and new assumptions.
  The stored market-condition text round-trips verbatim as `cached_condition`,
  which is what keeps the `SUMIF(..., "Good*", ...)` totals unchanged.
- Splits are deliberately not tracked or repaired. Polygon's adjusted bars are
  adjusted as of the request and the backfill only rewrites the last
  `POLYGON_BACKFILL_DAYS`, so a split leaves a price seam at the window edge in
  that symbol's file. Anything level-based across the seam (52-week and
  multi-year highs, RSI/MACD, average dollar volume, beta, the daily OHLC exit
  scan) is wrong for that symbol until a `--bootstrap --replace-existing`
  rebuild. `validate_price_data.py` still flags the resulting price jumps.
- `--as_of_date` is a real cutoff: `load_series_from_file`,
  `scan_all_time_high` and `load_ohlc_from_file` all drop rows past it, so a
  historical replay sees only the data that existed then. Any new file reader
  must honour `AS_OF_DATE_INT` too, or the simulation gets to see the bars it
  is supposed to be predicting.
- Simulation totals are `SUMIF(..., "Good*", ...)` over the market-condition
  column. openpyxl writes formulas but never evaluates them, so a workbook this
  pipeline just wrote has no cached total — `send_daily_email.py` recomputes
  both totals in Python and must keep applying the same `Good*` filter, or the
  email reports trades the workbook does not count.
- No live broker integration — the "Simulation"/"AM Simulation" sheets are
  paper-trading only.
