from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


GROUPED_DAILY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
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
    return ap.parse_args()


def normalize_polygon_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(".US"):
        symbol = f"{symbol}.US"
    return symbol


def symbol_to_file_name(symbol: str) -> str:
    return f"{symbol.lower()}.txt"


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


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    new_symbols_dir = resolve_path(args.new_symbols_dir) if args.new_symbols_dir else None
    api_key = args.api_key.strip()
    if not api_key:
        raise SystemExit("Set POLYGON_API_KEY or pass --api-key.")
    if api_key.lower() in {"your_polygon_key", "your_key_here"}:
        raise SystemExit("POLYGON_API_KEY is still set to a placeholder. Replace it with your real Polygon API key.")

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

        action = upsert_daily_row(path, row, date_int, dry_run=args.dry_run)
        counts[action] += 1

    mode = "DRY RUN: " if args.dry_run else ""
    print(
        f"{mode}Polygon {trading_date.isoformat()} -> "
        f"added={counts['added']}, updated={counts['updated']}, "
        f"unchanged={counts['unchanged']}, skipped_new_or_invalid={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
