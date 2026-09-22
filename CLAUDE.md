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
   - Fallback: `refresh_stooq_dump.py` copies/moves a manually downloaded
     Stooq bundle into place (`STOOQ_SRC`, `STOOQ_MODE`); only used when
     `POLYGON_API_KEY` is unset.
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
   `PM_ENTRY_DELAY_MINUTES` (10). Rows predating the column fall back to the
   after-hours session open, then to the rank date's close.

   PM extended hours run 4:00-8:00pm ET, and both CI schedules now kick off at
   8:15pm ET — after that window has already closed for the day. So once
   `Run Finished` is stamped, `Run Finished + 10 min` always lands past the
   session's last bar; rather than search forever for a bar that will never
   arrive, this case falls back to the session's **last published bar** (data
   source "PM extended hours (session close)") — see the
   `session_closed_fallback` branch in `build_investment_simulation_rows`.

   A cohort's own run hasn't stamped `Run Finished` yet when that cohort is
   built (the stamp happens later, right before save), so there's no
   `Run Finished + 10 min` target to search for on day one. But the run is
   already executing after 8:15pm ET, i.e. after the session closed, so the
   **same** last-published-bar fallback applies immediately, gated on wall
   clock (`datetime.now(EASTERN_TZ)`) rather than the not-yet-stamped
   timestamp — a same-day cohort is priced same-day, not deferred to
   tomorrow's run. It only falls through to `Pending` if the run executes
   before 8pm ET (e.g. a manual `workflow_dispatch`/`web` trigger) or Polygon
   hasn't published any PM bars for the symbol yet, in which case the
   simulation sheets rebuild from Daily Runs on the next run and pick it up.

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
- `SCREEN_UNIVERSE` (`us` default, or `nyse`/`nasdaq`/`all`) — which listing
  venues to screen. Read by both runners; it picks the `--bootstrap-universe`
  and the folder list handed to `generate_tickers.py`.
- `STOOQ_SRC`, `STOOQ_MODE` (`copy`|`move`) — Stooq fallback, only used if
  `POLYGON_API_KEY` is unset
- `SMTP_HOST/PORT/USERNAME/PASSWORD`, `EMAIL_FROM`, `EMAIL_TO`,
  `EMAIL_SUBJECT_PREFIX`, `EMAIL_ATTACH_RESULTS` — for `send_daily_email.py`
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — Alpaca **paper** trading keys for
  `alpaca_pm_entries.py`, used only by GitLab's `pm-entries` job (see
  Automation below); never commit these

## Automation

This project is pushed to two remotes running **identical source**, each
screening a different universe:

| remote | CI config | `SCREEN_UNIVERSE` |
|---|---|---|
| GitLab | `.gitlab-ci.yml` | `us` — NYSE + NASDAQ |
| GitHub | `.github/workflows/daily-screener.yml` | `nyse` — NYSE only |

GitHub Actions reads only `.github/workflows/`, GitLab only `.gitlab-ci.yml`, so
both files live in the same commit and each host ignores the other's. **Keep the
code, the schedule, and every CI behavior (timeouts, credential checks,
artifact retention, etc.) identical between the two configs for the daily
screening job** — `SCREEN_UNIVERSE` is the only intended difference in that
job between the two hosts, never a branch that only one remote carries, or
the two will drift. GitLab's second job, `pm-entries` (below), is a
deliberate, documented, GitLab-only exception to this rule, not a drift to
fix.

`.github/workflows/daily-screener.yml` runs on cron `15 20 * * 1-5` (8:15pm
America/New_York, weekdays): checkout → Python 3.12 setup → install deps →
restore `data 2` cache → run `run_daily.sh` → email results → commit the
updated `results.xlsx` back to the repo as `github-actions[bot]`. The GitLab
`daily-screener` job runs the same steps at the same 8:15pm America/New_York
time, on a schedule defined in the GitLab UI (Build > Pipeline schedules —
its cron timezone lives on the schedule, not in the YAML, so **that side of a
schedule change has to be made by hand in the GitLab UI**), and keeps
`results.xlsx` as a build artifact rather than committing it.

### GitLab-only: `pm-entries`

Alpaca's PM extended-hours window (4:00-8:00pm ET) is the same window the PM
Simulation fallback described above waits for to *close* before pricing off
it. That means a real order can't be submitted at 8:15pm — there's no live
session left. `pm-entries` (`.gitlab-ci.yml`) runs earlier, on its own GitLab
Pipeline Schedule at 4:15pm America/New_York (15 minutes after the 4:00pm
regular close, mirroring `daily-screener`'s own post-close buffer), while the
PM session is still open, and submits real Alpaca **paper** limit orders for
that day's top-10 cohort. GitLab only — GitLab screens the broader `us`
universe; GitHub gets no equivalent job.

It does its own lightweight ranking pass rather than reusing
`daily-screener`'s: `run_pm_entries.sh` does a same-day-only incremental
Polygon refresh (`--backfill-days 1 --include-today`), then
`screen_stooq.py --rank_only` screens and ranks but exits before touching
`results.xlsx` or `Daily Runs` (writing the ranked cohort to `top10.json`
instead), then `alpaca_pm_entries.py` applies the same regime gate
(`evaluate_market_regime`) and headroom formula the PM Simulation itself uses
(using the live quote fetched at submission time as the entry-price proxy,
since no fill exists yet) and submits a limit order — live quote + $0.05,
whole shares only, sized off the same $10,000 position size — for every row
that passes both gates. It writes every row's outcome (submitted, blocked,
excluded, or skipped, plus the Alpaca order id if submitted) to
`alpaca_state/<rank_date>.json` and commits that file back to git.

This file is the handoff to the evening job: GitLab's native artifact
passing (`needs:`) only works within one pipeline, and these are two
separately-scheduled pipelines, so a git commit is the only durable bridge
between them. `daily-screener` reads `alpaca_state/<rank_date>.json` back
(best-effort — the file may be absent) purely to attach reporting columns to
PM Simulation; it does not feed into `counts_toward_totals` or any other
gating logic there.

Both jobs live in one `.gitlab-ci.yml`, kept mutually exclusive by a
`SCHEDULE_KIND` variable set on each job's own GitLab Pipeline Schedule
(`SCHEDULE_KIND=entries` on the 4:15pm schedule only) — see the file's rules
for both jobs. A manual "New pipeline" run with no `SCHEDULE_KIND` set still
runs `daily-screener`, same as before this job existed.

The next-day stop/target exit-order step (mirroring `intraday_exit`'s
target/stop-first logic against the real Alpaca position) is not yet built.

## Notes

- `data 2/` and `results.xlsx` are treated as generated/cached state, not
  hand-edited source — they're rewritten by every daily run. `data 2/` is
  gitignored (`data [0-9]*/`) and is never committed; it only persists between
  runs via each CI host's own cache (see Automation above). `results.xlsx` is
  committed back to the repo by the GitHub workflow but kept only as a build
  artifact by GitLab. `us_tickers.csv`, `results.csv`, and `top10.json` are
  gitignored and regenerated locally — don't add logic that depends on any of
  them being fresh in a clean checkout. `alpaca_state/*.json` is the one
  exception: it *is* committed (by GitLab's `pm-entries` job only — see
  Automation), since it's the durable handoff between two separately
  scheduled pipelines, not throwaway intermediate output.
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
