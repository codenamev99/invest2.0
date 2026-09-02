from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings
from collections import Counter, OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)
import requests


GROUPED_DAILY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
TICKER_DAILY_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
REFERENCE_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"
SPLITS_URL = "https://api.polygon.io/stocks/v1/splits"
STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"

# One folder per listing venue, matching Stooq's own layout so the
# refresh_stooq_dump.py fallback stays coherent with what Polygon writes.
EXCHANGE_DIRS = {"XNYS": "nyse stocks", "XNAS": "nasdaq stocks"}
MIXED_EXCHANGE_DIR = "us stocks"
# Which venues each --bootstrap-universe covers. "all" has no reference lookup
# at all, so it is not listed here.
UNIVERSE_EXCHANGES = {
    "us": ("XNYS", "XNAS"),
    "nyse": ("XNYS",),
    "nasdaq": ("XNAS",),
}
# Common stock plus the two ADR classes; everything else on these venues is a
# warrant, unit, right or preferred share, which this screener does not trade.
EQUITY_TYPES = {"CS", "ADRC", "ADRP"}
# The screener reads exactly two ETFs -- SPY for the beta benchmark, QQQ for the
# market regime gate -- so those are the only ones worth storing. Fetching every
# US ETF instead costs thousands of files and hundreds of megabytes that nothing
# ever opens.
BENCHMARK_ETFS = {"SPY.US", "QQQ.US"}
EASTERN_TZ = ZoneInfo("America/New_York")

# Which splits have already been dealt with, so a repair runs once per split
# rather than on every run for as long as it stays inside the lookback window.
SPLIT_STATE_FILENAME = ".split_repairs.json"
# A row still on the old factor sits at split_from/split_to of its adjusted
# value, so an unrepaired seam reads as a close-to-close ratio of
# split_to/split_from. Genuine overnight gaps do not land near a split ratio, so
# a loose band still tells the two apart.
SPLIT_SEAM_TOLERANCE = 0.15
# Below this the ratio is too close to 1 for the seam test to mean anything.
SPLIT_MIN_RATIO_DISTANCE = 0.05


def resolve_path(p: str) -> Path:
    """
    Supports:
      - ${workspaceFolder} -> current working directory (Cursor/VS Code style)
      - ~ home expansion
      - environment variables like $HOME
      - relative paths resolved from current working directory
    """
    p = (p or "").strip()
    if "${workspaceFolder}" in p:
        p = p.replace("${workspaceFolder}", str(Path.cwd()))
    p = os.path.expandvars(os.path.expanduser(p))
    return Path(p).resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch Polygon grouped daily bars and upsert them into existing "
            "Stooq-style *.us.txt files."
        )
    )
    ap.add_argument(
        "--root",
        default="${workspaceFolder}/data 2/daily/us",
        help='Root folder containing Stooq-style US folders, e.g. "${workspaceFolder}/data 2/daily/us"',
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("POLYGON_API_KEY", ""),
        help="Polygon API key. Prefer setting POLYGON_API_KEY instead of passing this on the command line.",
    )
    ap.add_argument(
        "--date",
        default="",
        help="Trading date to refresh, YYYY-MM-DD. Defaults to the most recent date with grouped data.",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="When --date is omitted, try this many prior calendar days until Polygon returns data.",
    )
    ap.add_argument(
        "--backfill-days",
        type=int,
        default=0,
        help="Fetch and upsert every weekday in the last N calendar days, ending yesterday by default.",
    )
    ap.add_argument(
        "--include-today",
        action="store_true",
        help=(
            "Include the current calendar day in latest-date, backfill, and bootstrap requests. "
            "Use this only when the job runs after the market has closed."
        ),
    )
    ap.add_argument(
        "--ensure-benchmark-history-days",
        type=int,
        default=0,
        help=(
            "Ensure SPY.US and QQQ.US contain this many recent calendar days of daily history "
            "using Polygon's per-ticker aggregate endpoint. Disabled by default."
        ),
    )
    ap.add_argument(
        "--new-symbols-dir",
        default="",
        help=(
            "Optional folder for Polygon symbols that do not already exist under --root. "
            "If omitted, new symbols are skipped."
        ),
    )
    ap.add_argument(
        "--include-otc",
        action="store_true",
        help="Include OTC symbols in the grouped daily request.",
    )
    ap.add_argument(
        "--unadjusted",
        action="store_true",
        help="Use unadjusted prices. Default is adjusted=true, matching Polygon's split-adjusted bars.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report changes without writing files.",
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="Build a fresh Stooq-style history folder from Polygon grouped daily data.",
    )
    ap.add_argument(
        "--bootstrap-years",
        type=float,
        default=float(os.environ.get("POLYGON_BOOTSTRAP_YEARS", "2")),
        help="Years of history to build when --bootstrap is used. Free tier supports about 2 years.",
    )
    ap.add_argument(
        "--bootstrap-start",
        default="",
        help="Bootstrap start date, YYYY-MM-DD. Overrides --bootstrap-years.",
    )
    ap.add_argument(
        "--bootstrap-end",
        default="",
        help="Bootstrap end date, YYYY-MM-DD. Defaults to yesterday, or today with --include-today.",
    )
    ap.add_argument(
        "--bootstrap-universe",
        choices=["us", "nyse", "nasdaq", "all"],
        default="us",
        help=(
            "Universe to write during bootstrap. Default 'us' is NYSE plus NASDAQ common "
            "stock and ADRs, along with the benchmark ETFs. 'nyse'/'nasdaq' restrict to one "
            "venue; 'all' writes every symbol Polygon returns into a single folder."
        ),
    )
    ap.add_argument(
        "--replace-existing",
        action="store_true",
        help="Allow --bootstrap to replace existing *.txt data under --root.",
    )
    ap.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=float(os.environ.get("POLYGON_RATE_LIMIT_SLEEP", "13")),
        help="Seconds to sleep between bootstrap API calls. 13 seconds is safe for free-tier limits.",
    )
    ap.add_argument(
        "--split-repair-days",
        type=int,
        default=int(os.environ.get("POLYGON_SPLIT_REPAIR_DAYS", "120")),
        help=(
            "Look this many calendar days back for splits and restate the affected "
            "symbols' stored history. Keep it comfortably above --backfill-days so a "
            "missed run cannot let a split slip past. The first run repairs every "
            "split in the window, so it is slower than later ones. Default: 120."
        ),
    )
    ap.add_argument(
        "--no-split-repair",
        action="store_true",
        help="Skip the post-refresh split repair pass.",
    )
    return ap.parse_args()


def normalize_polygon_symbol(symbol: str) -> str:
    # Case is preserved deliberately: Polygon uses it to disambiguate distinct
    # tickers that share the same letters (e.g. "TPC" common stock vs "TpC" a
    # preferred share). Uppercasing here would silently merge unrelated
    # instruments into the same output file.
    symbol = symbol.strip()
    if not symbol.upper().endswith(".US"):
        symbol = f"{symbol}.US"
    return symbol


def symbol_to_file_name(symbol: str) -> str:
    return f"{symbol.lower()}.txt"


def parse_yyyy_mm_dd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_file_map(root: Path) -> dict[str, Path]:
    mp: dict[str, Path] = {}
    if not root.exists():
        return mp
    for p in root.rglob("*.us.txt"):
        sym = p.name[:-4].upper()
        if not sym.endswith(".US"):
            sym = f"{sym}.US"
        mp[sym] = p
    return mp


def candidate_dates(lookback_days: int, include_today: bool = False) -> list[date]:
    # Morning/default runs avoid today's incomplete bar; post-close runs may opt in.
    start = date.today() if include_today else date.today() - timedelta(days=1)
    return [start - timedelta(days=i) for i in range(max(1, lookback_days))]


def backfill_candidate_dates(backfill_days: int, include_today: bool = False) -> list[date]:
    end = date.today() if include_today else date.today() - timedelta(days=1)
    start = end - timedelta(days=max(1, backfill_days) - 1)
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


def fetch_grouped_daily(
    api_key: str,
    trading_date: date,
    adjusted: bool,
    include_otc: bool,
) -> list[dict[str, Any]]:
    resp = requests.get(
        GROUPED_DAILY_URL.format(date=trading_date.isoformat()),
        params={
            "adjusted": str(adjusted).lower(),
            "include_otc": str(include_otc).lower(),
            "apiKey": api_key,
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise RuntimeError("Polygon rate limit hit. Try again later or use a paid plan.")
    if resp.status_code in {401, 403}:
        raise RuntimeError("Polygon rejected the API key or this endpoint is not enabled for the plan.")
    resp.raise_for_status()

    payload = resp.json()
    status = str(payload.get("status", "")).upper()
    if status not in {"OK", "DELAYED"}:
        message = payload.get("message") or payload.get("error") or payload
        raise RuntimeError(f"Polygon returned status {status!r}: {message}")

    return list(payload.get("results") or [])


def ensure_benchmark_history(
    root: Path,
    api_key: str,
    calendar_days: int,
    adjusted: bool,
    include_today: bool,
    dry_run: bool,
) -> None:
    """Repair SPY/QQQ history with two targeted Polygon aggregate requests."""
    if calendar_days <= 0:
        return
    end = date.today() if include_today else date.today() - timedelta(days=1)
    start = end - timedelta(days=calendar_days - 1)
    symbol_paths = build_file_map(root)

    for symbol in ("SPY.US", "QQQ.US"):
        path = symbol_paths.get(symbol) or root / "etfs" / symbol_to_file_name(symbol)
        polygon_symbol = symbol.removesuffix(".US")
        resp = requests.get(
            TICKER_DAILY_URL.format(
                ticker=polygon_symbol,
                start=start.isoformat(),
                end=end.isoformat(),
            ),
            params={
                "adjusted": str(adjusted).lower(),
                "sort": "asc",
                "limit": 50000,
                "apiKey": api_key,
            },
            timeout=60,
        )
        if resp.status_code in {401, 403}:
            raise RuntimeError(f"Polygon rejected the API key while repairing {symbol} history.")
        resp.raise_for_status()
        rows = resp.json().get("results") or []
        counts = {"added": 0, "updated": 0, "unchanged": 0}
        for bar in rows:
            timestamp_ms = bar.get("t")
            if timestamp_ms is None:
                continue
            trading_date = datetime.utcfromtimestamp(float(timestamp_ms) / 1000.0).date()
            row = stooq_row(symbol, trading_date, bar)
            action = upsert_daily_row(
                path,
                row,
                int(trading_date.strftime("%Y%m%d")),
                dry_run=dry_run,
            )
            counts[action] += 1
        print(
            f"Ensured {symbol} history {start.isoformat()} to {end.isoformat()}: "
            f"added={counts['added']}, updated={counts['updated']}, unchanged={counts['unchanged']}"
        )


def fetch_grouped_daily_with_retries(
    api_key: str,
    trading_date: date,
    adjusted: bool,
    include_otc: bool,
    rate_limit_sleep: float,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_grouped_daily(api_key, trading_date, adjusted, include_otc)
        except RuntimeError as e:
            if "rate limit" not in str(e).lower() or attempt >= max_attempts:
                raise
            print(f"Rate limited on {trading_date.isoformat()}; sleeping {rate_limit_sleep:.1f}s...")
            time.sleep(rate_limit_sleep)
        except requests.RequestException:
            if attempt >= max_attempts:
                raise
            time.sleep(min(rate_limit_sleep, 5.0))
    return []


def fetch_reference_symbols(
    api_key: str,
    rate_limit_sleep: float,
    exchanges: tuple[str, ...],
) -> dict[str, str]:
    """
    Map each tradable symbol on `exchanges` to the folder it belongs in.

    Only common stock and ADRs are returned; ETFs are not looked up here because
    the screener needs just the two benchmarks in BENCHMARK_ETFS.
    """
    stock_dirs: dict[str, str] = {}
    next_url: str | None = REFERENCE_TICKERS_URL
    params: dict[str, Any] | None = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "apiKey": api_key,
    }

    while next_url:
        resp = requests.get(next_url, params=params, timeout=60)
        if resp.status_code == 429:
            print(f"Rate limited while fetching ticker reference; sleeping {rate_limit_sleep:.1f}s...")
            time.sleep(rate_limit_sleep)
            continue
        if resp.status_code in {401, 403}:
            raise RuntimeError("Polygon rejected the API key or reference tickers endpoint is not enabled.")
        resp.raise_for_status()
        payload = resp.json()
        status = str(payload.get("status", "")).upper()
        if status not in {"OK", "DELAYED"}:
            message = payload.get("message") or payload.get("error") or payload
            raise RuntimeError(f"Polygon reference tickers returned status {status!r}: {message}")

        for item in payload.get("results") or []:
            ticker = str(item.get("ticker") or "").strip()
            if not ticker:
                continue
            symbol = normalize_polygon_symbol(ticker)
            security_type = str(item.get("type") or "").upper()
            primary_exchange = str(item.get("primary_exchange") or "").upper()
            if primary_exchange in exchanges and security_type in EQUITY_TYPES:
                stock_dirs[symbol] = EXCHANGE_DIRS[primary_exchange]

        next_url = payload.get("next_url")
        params = {"apiKey": api_key} if next_url else None
        if next_url:
            time.sleep(rate_limit_sleep)

    return stock_dirs


def bootstrap_date_range(args: argparse.Namespace) -> list[date]:
    default_end = date.today() if args.include_today else date.today() - timedelta(days=1)
    end = parse_yyyy_mm_dd(args.bootstrap_end) if args.bootstrap_end else default_end
    if args.bootstrap_start:
        start = parse_yyyy_mm_dd(args.bootstrap_start)
    else:
        start = end - timedelta(days=int(args.bootstrap_years * 365))
    if start > end:
        raise SystemExit("--bootstrap-start must be on or before --bootstrap-end.")
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


class AppendFileCache:
    def __init__(self, dry_run: bool, max_open: int = 128) -> None:
        self.dry_run = dry_run
        self.max_open = max_open
        self._handles: OrderedDict[Path, Any] = OrderedDict()
        self._initialized: set[Path] = set()

    def append_line(self, path: Path, line: str) -> None:
        if self.dry_run:
            return
        if path not in self._initialized:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size == 0:
                path.write_text(STOOQ_HEADER + "\n", encoding="utf-8")
            self._initialized.add(path)

        handle = self._handles.get(path)
        if handle is None:
            if len(self._handles) >= self.max_open:
                _, old_handle = self._handles.popitem(last=False)
                old_handle.close()
            handle = path.open("a", encoding="utf-8")
            self._handles[path] = handle
        else:
            self._handles.move_to_end(path)
        handle.write(line + "\n")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "AppendFileCache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def bootstrap_target_path(
    root: Path,
    symbol: str,
    stock_dirs: dict[str, str],
    etf_symbols: set[str],
    universe: str,
) -> Path | None:
    if symbol in etf_symbols:
        return root / "etfs" / symbol_to_file_name(symbol)
    if universe == "all" or not stock_dirs:
        # No per-symbol venue is known, so everything shares one folder.
        return root / MIXED_EXCHANGE_DIR / symbol_to_file_name(symbol)
    subdir = stock_dirs.get(symbol)
    if subdir is None:
        return None
    return root / subdir / symbol_to_file_name(symbol)


def bootstrap_history(args: argparse.Namespace, root: Path, api_key: str) -> None:
    existing_files = list(root.rglob("*.txt")) if root.exists() else []
    if existing_files and not args.replace_existing:
        raise SystemExit(
            f"{root} already contains data. Use --replace-existing with --bootstrap to rebuild it."
        )

    if existing_files and args.replace_existing and not args.dry_run:
        shutil.rmtree(root)

    stock_dirs: dict[str, str] = {}
    etf_symbols: set[str] = set(BENCHMARK_ETFS)
    exchanges = UNIVERSE_EXCHANGES.get(args.bootstrap_universe)
    if exchanges:
        venues = ", ".join(EXCHANGE_DIRS[code].removesuffix(" stocks").upper() for code in exchanges)
        print(f"Fetching Polygon ticker reference for {venues} common stocks...")
        stock_dirs = fetch_reference_symbols(api_key, args.rate_limit_sleep, exchanges)
        per_venue = Counter(stock_dirs.values())
        breakdown = ", ".join(f"{count} in {name}" for name, count in sorted(per_venue.items()))
        print(f"Reference universe: {len(stock_dirs)} stocks ({breakdown}), {len(etf_symbols)} benchmark ETFs.")

    dates = bootstrap_date_range(args)
    if not dates:
        raise SystemExit("No weekdays found in requested bootstrap date range.")

    print(
        f"Bootstrapping Polygon history from {dates[0].isoformat()} to {dates[-1].isoformat()} "
        f"({len(dates)} weekdays)."
    )
    if args.rate_limit_sleep > 0:
        print(
            f"Sleeping {args.rate_limit_sleep:g}s between calls for free-tier rate limits; "
            "set POLYGON_RATE_LIMIT_SLEEP=0 on a paid plan to skip the wait."
        )

    counts = {"days_with_data": 0, "rows_written": 0, "skipped": 0}
    with AppendFileCache(dry_run=args.dry_run) as files:
        for idx, trading_date in enumerate(dates, start=1):
            bars = fetch_grouped_daily_with_retries(
                api_key,
                trading_date,
                adjusted=not args.unadjusted,
                include_otc=args.include_otc,
                rate_limit_sleep=args.rate_limit_sleep,
            )
            if bars:
                counts["days_with_data"] += 1
            for bar in bars:
                raw_symbol = str(bar.get("T") or "").strip()
                if not raw_symbol:
                    counts["skipped"] += 1
                    continue
                symbol = normalize_polygon_symbol(raw_symbol)
                path = bootstrap_target_path(root, symbol, stock_dirs, etf_symbols, args.bootstrap_universe)
                if path is None:
                    counts["skipped"] += 1
                    continue
                try:
                    files.append_line(path, stooq_row(symbol, trading_date, bar))
                except KeyError:
                    counts["skipped"] += 1
                    continue
                counts["rows_written"] += 1

            print(
                f"[{idx}/{len(dates)}] {trading_date.isoformat()}: "
                f"bars={len(bars)}, written={counts['rows_written']}, skipped={counts['skipped']}"
            )
            if idx < len(dates):
                time.sleep(args.rate_limit_sleep)

    mode = "DRY RUN: " if args.dry_run else ""
    print(
        f"{mode}Bootstrap complete -> days_with_data={counts['days_with_data']}, "
        f"rows_written={counts['rows_written']}, skipped={counts['skipped']}, root={root}"
    )


def find_latest_grouped_data(
    api_key: str,
    lookback_days: int,
    adjusted: bool,
    include_otc: bool,
    include_today: bool = False,
) -> tuple[date, list[dict[str, Any]]]:
    errors: list[str] = []
    for d in candidate_dates(lookback_days, include_today=include_today):
        try:
            rows = fetch_grouped_daily(api_key, d, adjusted, include_otc)
        except RuntimeError:
            raise
        except requests.RequestException as e:
            errors.append(f"{d.isoformat()}: {e}")
            time.sleep(1)
            continue
        if rows:
            return d, rows
    detail = f" Recent request errors: {'; '.join(errors)}" if errors else ""
    raise RuntimeError(f"No Polygon grouped daily rows found in the last {lookback_days} days.{detail}")


def stooq_row(symbol: str, trading_date: date, bar: dict[str, Any]) -> str:
    date_i = trading_date.strftime("%Y%m%d")
    return ",".join(
        [
            symbol,
            "D",
            date_i,
            "000000",
            str(bar["o"]),
            str(bar["h"]),
            str(bar["l"]),
            str(bar["c"]),
            str(int(bar.get("v") or 0)),
            "0",
        ]
    )


def row_date(line: str) -> int | None:
    parts = line.strip().split(",")
    if len(parts) < 9 or parts[1] != "D":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def upsert_daily_row(path: Path, new_row: str, date_int: int, dry_run: bool) -> str:
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        lines = [STOOQ_HEADER]

    changed = False
    replaced = False
    out: list[str] = []
    for line in lines:
        existing_date = row_date(line)
        if existing_date == date_int:
            if line != new_row:
                changed = True
                out.append(new_row)
            else:
                out.append(line)
            replaced = True
        else:
            out.append(line)

    if replaced:
        action = "updated" if changed else "unchanged"
    else:
        out.append(new_row)
        changed = True
        action = "added"

    if changed:
        header_lines = [line for line in out if row_date(line) is None]
        data_lines = [line for line in out if row_date(line) is not None]
        data_lines.sort(key=lambda line: row_date(line) or 0)
        out = header_lines + data_lines
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(out) + "\n", encoding="utf-8")

    return action


def upsert_grouped_bars(
    bars: list[dict[str, Any]],
    trading_date: date,
    symbol_paths: dict[str, Path],
    new_symbols_dir: Path | None,
    dry_run: bool,
) -> dict[str, int]:
    date_int = int(trading_date.strftime("%Y%m%d"))
    counts = {"added": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for bar in bars:
        raw_symbol = str(bar.get("T") or "").strip()
        if not raw_symbol:
            counts["skipped"] += 1
            continue

        symbol = normalize_polygon_symbol(raw_symbol)
        path = symbol_paths.get(symbol)
        if path is None:
            if new_symbols_dir is None:
                counts["skipped"] += 1
                continue
            path = new_symbols_dir / symbol_to_file_name(symbol)

        try:
            row = stooq_row(symbol, trading_date, bar)
        except KeyError:
            counts["skipped"] += 1
            continue

        action = upsert_daily_row(path, row, date_int, dry_run=dry_run)
        counts[action] += 1

    return counts


def merge_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


# -----------------------------
# Split repair
#
# Polygon's adjusted bars are adjusted as of the moment they are requested, and
# the daily backfill only rewrites its own recent window. A split therefore
# leaves every row older than that window sitting on the pre-split factor, and
# the file keeps a permanent price seam at the window edge. That corrupts every
# level-based figure computed across it -- 52-week and multi-year highs, the
# RSI/MACD averages, average dollar volume, beta, and the simulation's daily
# OHLC exit scan.
# -----------------------------
def bar_trading_date(timestamp_ms: Any) -> date | None:
    """Polygon stamps a daily bar at midnight Eastern on its trading day."""
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=EASTERN_TZ).date()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def date_int_to_date(date_int: int) -> date:
    return datetime.strptime(str(date_int), "%Y%m%d").date()


def ratio_leg_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def price_text(value: float) -> str:
    """
    Format a price for a Stooq-style row.

    Rescaling multiplies by a ratio, which leaves float noise like
    49.999999999999996. These files carry plain decimals, so trim to a precision
    no US equity quote exceeds and drop the trailing zeros.
    """
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_stooq_row(line: str) -> dict[str, Any] | None:
    parts = line.strip().split(",")
    if len(parts) < 9 or parts[1] != "D":
        return None
    try:
        return {
            "symbol": parts[0],
            "date_int": int(parts[2]),
            "open": float(parts[4]),
            "high": float(parts[5]),
            "low": float(parts[6]),
            "close": float(parts[7]),
            "volume": float(parts[8]),
            "openint": parts[9] if len(parts) > 9 else "0",
        }
    except ValueError:
        return None


def format_stooq_row(row: dict[str, Any]) -> str:
    return ",".join(
        [
            str(row["symbol"]),
            "D",
            str(row["date_int"]),
            "000000",
            price_text(row["open"]),
            price_text(row["high"]),
            price_text(row["low"]),
            price_text(row["close"]),
            str(int(round(row["volume"]))),
            str(row.get("openint", "0")),
        ]
    )


def split_state_path(root: Path) -> Path:
    return root / SPLIT_STATE_FILENAME


def load_split_repair_state(root: Path) -> dict[str, Any]:
    path = split_state_path(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A damaged state file only costs a redundant repair pass, which is a
        # no-op on already-correct history, so it is not worth failing the run.
        return {}
    return data if isinstance(data, dict) else {}


def save_split_repair_state(root: Path, state: dict[str, Any], keep_since: date) -> None:
    pruned = {
        key: value
        for key, value in state.items()
        if str((value or {}).get("execution_date", "")) >= keep_since.isoformat()
    }
    path = split_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pruned, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_splits_since(api_key: str, since: date) -> list[dict[str, Any]]:
    """Every split Polygon reports as executing on or after `since`."""
    rows: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "execution_date.gte": since.isoformat(),
        "sort": "execution_date.desc",
        "limit": 1000,
        "apiKey": api_key,
    }
    url: str | None = SPLITS_URL
    while url:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code in {401, 403}:
            raise RuntimeError("Polygon rejected the API key while listing splits.")
        resp.raise_for_status()
        payload = resp.json()
        for row in payload.get("results") or []:
            ticker = str(row.get("ticker") or "").strip()
            try:
                executed = parse_yyyy_mm_dd(str(row.get("execution_date")))
                split_from = float(row.get("split_from"))
                split_to = float(row.get("split_to"))
            except (TypeError, ValueError):
                continue
            if not ticker or split_from <= 0 or split_to <= 0:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "execution_date": executed,
                    "split_from": split_from,
                    "split_to": split_to,
                }
            )
        url = payload.get("next_url") or None
        params = {"apiKey": api_key}
    return rows


def fetch_ticker_daily(
    api_key: str,
    ticker: str,
    start: date,
    end: date,
    adjusted: bool = True,
) -> list[dict[str, Any]]:
    """Full daily history for one ticker over a date range."""
    bars: list[dict[str, Any]] = []
    url: str | None = TICKER_DAILY_URL.format(
        ticker=ticker, start=start.isoformat(), end=end.isoformat()
    )
    params: dict[str, Any] = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    while url:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code in {401, 403}:
            raise RuntimeError(f"Polygon rejected the API key while fetching {ticker} history.")
        resp.raise_for_status()
        payload = resp.json()
        bars.extend(payload.get("results") or [])
        url = payload.get("next_url") or None
        params = {"apiKey": api_key}
    return bars


def rescale_pre_split_rows(
    rows: list[dict[str, Any]],
    seam_date_int: int,
    split_from: float,
    split_to: float,
) -> int:
    """
    Divide out a split still sitting in the rows before `seam_date_int`.

    `seam_date_int` is where the stale region ends: the split's execution date,
    or the start of whatever range Polygon just refetched when that is earlier,
    because refetching part of the history moves the seam to the edge of the
    served range rather than removing it.

    The rescale only happens when the seam is measurably there, so it is a no-op
    on history Polygon served whole. That is what makes a second pass over an
    already-correct file safe. Returns the number of rows restated.
    """
    seam_ratio = split_to / split_from
    if abs(seam_ratio - 1.0) < SPLIT_MIN_RATIO_DISTANCE:
        return 0

    seam_idx = next(
        (idx for idx, row in enumerate(rows) if row["date_int"] >= seam_date_int),
        None,
    )
    if not seam_idx:
        # None means the history ends before the seam; 0 means it starts after
        # it. Either way there is nothing on the old factor to restate.
        return 0

    before = rows[seam_idx - 1]["close"]
    after = rows[seam_idx]["close"]
    if before <= 0 or after <= 0:
        return 0
    if abs((before / after) / seam_ratio - 1.0) > SPLIT_SEAM_TOLERANCE:
        return 0

    price_factor = split_from / split_to
    for row in rows[:seam_idx]:
        for field in ("open", "high", "low", "close"):
            row[field] = row[field] * price_factor
        row["volume"] = row["volume"] / price_factor
    return seam_idx


def repair_symbol_split_history(
    path: Path,
    ticker: str,
    execution_date: date,
    split_from: float,
    split_to: float,
    api_key: str,
    dry_run: bool,
) -> dict[str, int] | None:
    """
    Restate one symbol's stored history onto the current split factor.

    Polygon's own history is authoritative for whatever range the plan serves,
    so that range is replaced outright. Anything older than what it returns is
    rescaled instead, and only when the seam is still detectable.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_lines = [line for line in lines if row_date(line) is None]
    existing: dict[int, dict[str, Any]] = {}
    for line in lines:
        parsed = parse_stooq_row(line)
        if parsed is not None:
            existing[parsed["date_int"]] = parsed
    if not existing:
        return None

    symbol = existing[min(existing)]["symbol"]
    bars = fetch_ticker_daily(
        api_key, ticker, date_int_to_date(min(existing)), date_int_to_date(max(existing))
    )

    replaced = 0
    served: list[int] = []
    for bar in bars:
        trading_date = bar_trading_date(bar.get("t"))
        if trading_date is None:
            continue
        date_int = int(trading_date.strftime("%Y%m%d"))
        try:
            row = {
                "symbol": symbol,
                "date_int": date_int,
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar.get("v") or 0),
                "openint": "0",
            }
        except (KeyError, TypeError, ValueError):
            continue
        if date_int in existing:
            replaced += 1
        existing[date_int] = row
        served.append(date_int)

    ordered = [existing[key] for key in sorted(existing)]
    # Rows Polygon served are authoritative, so the stale region ends wherever
    # its range starts -- which is earlier than the split whenever the plan's
    # history window does not reach all the way back through the file.
    execution_int = int(execution_date.strftime("%Y%m%d"))
    seam_date_int = min(min(served), execution_int) if served else execution_int
    rescaled = rescale_pre_split_rows(ordered, seam_date_int, split_from, split_to)

    if not dry_run:
        out = header_lines + [format_stooq_row(row) for row in ordered]
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"replaced": replaced, "rescaled": rescaled, "rows": len(ordered)}


def repair_split_adjusted_history(args: argparse.Namespace, root: Path, api_key: str) -> None:
    """Restate stored history for every tracked symbol that split recently."""
    if args.no_split_repair or args.split_repair_days <= 0:
        return

    since = date.today() - timedelta(days=args.split_repair_days)
    try:
        splits = fetch_splits_since(api_key, since)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Warning: could not list recent splits from Polygon ({exc}); history left as is.")
        return

    state = load_split_repair_state(root)
    symbol_paths = build_file_map(root)
    mode = "DRY RUN: " if args.dry_run else ""
    pending = [
        split
        for split in splits
        if f"{split['ticker']}@{split['execution_date'].isoformat()}" not in state
        and normalize_polygon_symbol(split["ticker"]).upper() in symbol_paths
    ]
    if not pending:
        print(
            f"Split repair: nothing to restate since {since.isoformat()} "
            f"({len(splits)} splits seen, all already handled or untracked)."
        )
        return

    print(f"{mode}Split repair: restating {len(pending)} symbol(s) since {since.isoformat()}.")
    repaired = 0
    for idx, split in enumerate(pending):
        ticker = split["ticker"]
        ratio = f"{ratio_leg_text(split['split_from'])}:{ratio_leg_text(split['split_to'])}"
        path = symbol_paths[normalize_polygon_symbol(ticker).upper()]
        try:
            outcome = repair_symbol_split_history(
                path,
                ticker,
                split["execution_date"],
                split["split_from"],
                split["split_to"],
                api_key,
                dry_run=args.dry_run,
            )
        except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
            print(f"Warning: could not restate {ticker} after its {ratio} split ({exc}).")
            continue
        if outcome is None:
            print(f"Skipped {ticker}: stored file has no usable daily rows.")
            continue

        repaired += 1
        print(
            f"{mode}Restated {ticker} for its {split['execution_date'].isoformat()} {ratio} split: "
            f"{outcome['replaced']} rows refetched, {outcome['rescaled']} older rows rescaled, "
            f"{outcome['rows']} rows total."
        )
        state[f"{ticker}@{split['execution_date'].isoformat()}"] = {
            "ticker": ticker,
            "execution_date": split["execution_date"].isoformat(),
            "ratio": ratio,
            "repaired_at": datetime.now().isoformat(timespec="seconds"),
            "rows_refetched": outcome["replaced"],
            "rows_rescaled": outcome["rescaled"],
        }
        if args.rate_limit_sleep > 0 and idx < len(pending) - 1:
            time.sleep(args.rate_limit_sleep)

    if repaired and not args.dry_run:
        # Keep the window wider than the lookback so an entry cannot be pruned
        # while its split is still listed, which would repair it a second time.
        save_split_repair_state(root, state, keep_since=since - timedelta(days=args.split_repair_days))
    print(f"{mode}Split repair complete -> restated={repaired}, skipped={len(pending) - repaired}.")


def backfill_recent_days(args: argparse.Namespace, root: Path, new_symbols_dir: Path | None, api_key: str) -> None:
    dates = backfill_candidate_dates(args.backfill_days, include_today=args.include_today)
    if not dates:
        raise SystemExit("--backfill-days did not produce any weekdays to fetch.")

    symbol_paths = build_file_map(root)
    total = {"added": 0, "updated": 0, "unchanged": 0, "skipped": 0, "no_data": 0}
    mode = "DRY RUN: " if args.dry_run else ""
    print(
        f"{mode}Backfilling Polygon grouped daily bars from {dates[0].isoformat()} "
        f"to {dates[-1].isoformat()} ({len(dates)} weekdays)."
    )

    for idx, trading_date in enumerate(dates, start=1):
        bars = fetch_grouped_daily_with_retries(
            api_key,
            trading_date,
            adjusted=not args.unadjusted,
            include_otc=args.include_otc,
            rate_limit_sleep=args.rate_limit_sleep,
        )
        if not bars:
            total["no_data"] += 1
            print(f"[{idx}/{len(dates)}] {trading_date.isoformat()}: no data")
            continue
        counts = upsert_grouped_bars(
            bars,
            trading_date,
            symbol_paths=symbol_paths,
            new_symbols_dir=new_symbols_dir,
            dry_run=args.dry_run,
        )
        merge_counts(total, counts)
        print(
            f"[{idx}/{len(dates)}] {trading_date.isoformat()}: "
            f"added={counts['added']}, updated={counts['updated']}, "
            f"unchanged={counts['unchanged']}, skipped_new_or_invalid={counts['skipped']}"
        )

    print(
        f"{mode}Backfill complete -> added={total['added']}, updated={total['updated']}, "
        f"unchanged={total['unchanged']}, skipped_new_or_invalid={total['skipped']}, "
        f"days_no_data={total['no_data']}"
    )


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    new_symbols_dir = resolve_path(args.new_symbols_dir) if args.new_symbols_dir else None
    api_key = args.api_key.strip()
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY or pass --api-key.")
    if api_key.lower() in {"your_polygon_key", "your_key_here"}:
        raise SystemExit("POLYGON_API_KEY is still set to a placeholder. Replace it with your real Polygon API key.")

    if args.bootstrap:
        bootstrap_history(args, root, api_key)
        return

    if args.backfill_days:
        if args.backfill_days <= 0:
            raise SystemExit("--backfill-days must be > 0.")
        backfill_recent_days(args, root, new_symbols_dir, api_key)
        ensure_benchmark_history(
            root,
            api_key,
            calendar_days=args.ensure_benchmark_history_days,
            adjusted=not args.unadjusted,
            include_today=args.include_today,
            dry_run=args.dry_run,
        )
        # Last, so it restates history the backfill above has already written.
        repair_split_adjusted_history(args, root, api_key)
        return

    if args.date:
        trading_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        bars = fetch_grouped_daily(api_key, trading_date, adjusted=not args.unadjusted, include_otc=args.include_otc)
        if not bars:
            raise SystemExit(f"Polygon returned no grouped rows for {trading_date.isoformat()}.")
    else:
        trading_date, bars = find_latest_grouped_data(
            api_key,
            lookback_days=args.lookback_days,
            adjusted=not args.unadjusted,
            include_otc=args.include_otc,
            include_today=args.include_today,
        )

    symbol_paths = build_file_map(root)
    counts = upsert_grouped_bars(
        bars,
        trading_date,
        symbol_paths=symbol_paths,
        new_symbols_dir=new_symbols_dir,
        dry_run=args.dry_run,
    )

    mode = "DRY RUN: " if args.dry_run else ""
    print(
        f"{mode}Polygon {trading_date.isoformat()} -> "
        f"added={counts['added']}, updated={counts['updated']}, "
        f"unchanged={counts['unchanged']}, skipped_new_or_invalid={counts['skipped']}"
    )
    repair_split_adjusted_history(args, root, api_key)


if __name__ == "__main__":
    main()
