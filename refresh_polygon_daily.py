from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


GROUPED_DAILY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
REFERENCE_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"
STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"


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
        help="Fetch and upsert every weekday in the last N calendar days, ending yesterday.",
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
        help="Bootstrap end date, YYYY-MM-DD. Defaults to yesterday.",
    )
    ap.add_argument(
        "--bootstrap-universe",
        choices=["nyse", "all"],
        default="nyse",
        help="Universe to write during bootstrap. Default: NYSE common stocks plus ETFs needed for benchmarks.",
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
    return ap.parse_args()


def normalize_polygon_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(".US"):
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


def candidate_dates(lookback_days: int) -> list[date]:
    # Start at yesterday so a scheduled morning run does not ask for today's incomplete bar.
    start = date.today() - timedelta(days=1)
    return [start - timedelta(days=i) for i in range(max(1, lookback_days))]


def backfill_candidate_dates(backfill_days: int) -> list[date]:
    start = date.today() - timedelta(days=max(1, backfill_days))
    end = date.today() - timedelta(days=1)
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


def fetch_reference_symbols(api_key: str, rate_limit_sleep: float) -> tuple[set[str], set[str]]:
    """
    Return (NYSE common stock symbols, ETF symbols) normalized to *.US.
    """
    stock_symbols: set[str] = set()
    etf_symbols: set[str] = set()
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
            if security_type == "ETF":
                etf_symbols.add(symbol)
            elif primary_exchange == "XNYS" and security_type in {"CS", "ADRC", "ADRP"}:
                stock_symbols.add(symbol)

        next_url = payload.get("next_url")
        params = {"apiKey": api_key} if next_url else None
        if next_url:
            time.sleep(rate_limit_sleep)

    etf_symbols.add("SPY.US")
    return stock_symbols, etf_symbols


def bootstrap_date_range(args: argparse.Namespace) -> list[date]:
    end = parse_yyyy_mm_dd(args.bootstrap_end) if args.bootstrap_end else date.today() - timedelta(days=1)
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
    stock_symbols: set[str],
    etf_symbols: set[str],
    universe: str,
) -> Path | None:
    if symbol in etf_symbols:
        return root / "etfs" / symbol_to_file_name(symbol)
    if universe == "all" or not stock_symbols:
        return root / "nyse stocks" / symbol_to_file_name(symbol)
    if symbol in stock_symbols:
        return root / "nyse stocks" / symbol_to_file_name(symbol)
    return None


def bootstrap_history(args: argparse.Namespace, root: Path, api_key: str) -> None:
    existing_files = list(root.rglob("*.txt")) if root.exists() else []
    if existing_files and not args.replace_existing:
        raise SystemExit(
            f"{root} already contains data. Use --replace-existing with --bootstrap to rebuild it."
        )

    if existing_files and args.replace_existing and not args.dry_run:
        shutil.rmtree(root)

    stock_symbols: set[str] = set()
    etf_symbols: set[str] = {"SPY.US"}
    if args.bootstrap_universe == "nyse":
        print("Fetching Polygon ticker reference for NYSE common stocks...")
        stock_symbols, etf_symbols = fetch_reference_symbols(api_key, args.rate_limit_sleep)
        print(f"Reference universe: {len(stock_symbols)} NYSE common stocks, {len(etf_symbols)} ETFs.")

    dates = bootstrap_date_range(args)
    if not dates:
        raise SystemExit("No weekdays found in requested bootstrap date range.")

    print(
        f"Bootstrapping Polygon history from {dates[0].isoformat()} to {dates[-1].isoformat()} "
        f"({len(dates)} weekdays)."
    )
    if args.bootstrap_universe == "nyse":
        print("Free-tier rate limits make the first build slow; this is expected.")

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
                path = bootstrap_target_path(root, symbol, stock_symbols, etf_symbols, args.bootstrap_universe)
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
) -> tuple[date, list[dict[str, Any]]]:
    errors: list[str] = []
    for d in candidate_dates(lookback_days):
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


def backfill_recent_days(args: argparse.Namespace, root: Path, new_symbols_dir: Path | None, api_key: str) -> None:
    dates = backfill_candidate_dates(args.backfill_days)
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


if __name__ == "__main__":
    main()
