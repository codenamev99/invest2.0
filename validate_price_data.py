"""
Sanity-checks Stooq-format daily OHLCV files under data 2/daily/us for two
classes of problems that flow silently into screen_stooq.py's indicators and
simulated fills if left undetected:

  1. Internal OHLC consistency (bad ticks): O/H/L/C relationships that are
     physically impossible, non-positive prices, negative volume, duplicate
     date rows with conflicting values.
  2. Calendar gaps: trading days present in the benchmark (SPY) but missing
     from a ticker's file. A file can look "fine" (sorted, valid rows) while
     still having a multi-month hole in the middle that indicators compute
     straight across as if it were one trading day, because the loaders in
     screen_stooq.py read the last N *rows*, not the last N *calendar days*.

Also flags large day-over-day moves (default >35%) between calendar-adjacent
rows as a heads-up for possible unadjusted splits/dividends or bad ticks;
this is a cheap early-warning signal, not a substitute for a dedicated
split/dividend reconciliation against Polygon's reference endpoints.

This script only reads data 2/; it never writes to it.

Usage:
  python3 validate_price_data.py --root "data 2/daily/us"
  python3 validate_price_data.py --root "data 2/daily/us" --symbols AAPL,MSFT
  python3 validate_price_data.py --root "data 2/daily/us" --out report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def resolve_path(p: str) -> Path:
    p = (p or "").strip()
    if "${workspaceFolder}" in p:
        p = p.replace("${workspaceFolder}", str(Path.cwd()))
    p = os.path.expandvars(os.path.expanduser(p))
    return Path(p).resolve()


def date_from_int(d: int) -> date:
    return date(d // 10000, (d // 100) % 100, d % 100)


def symbol_from_path(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".us.txt"):
        return name[:-4].upper()
    return path.stem.upper()


@dataclass
class Row:
    date_i: int
    open_: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Finding:
    symbol: str
    severity: str  # "ERROR" | "WARNING"
    check: str
    detail: str
    date_range: str = ""


def parse_file(path: Path) -> tuple[list[Row], list[str], list[str]]:
    """Returns (ordered rows incl. duplicates, malformed lines, raw duplicate-date lines)."""
    rows: list[Row] = []
    malformed: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("<TICKER>"):
                continue
            parts = ln.split(",")
            if len(parts) < 9 or parts[1] != "D":
                malformed.append(ln)
                continue
            try:
                date_i = int(parts[2])
                o = float(parts[4])
                h = float(parts[5])
                l = float(parts[6])
                c = float(parts[7])
                v = float(parts[8])
            except ValueError:
                malformed.append(ln)
                continue
            rows.append(Row(date_i, o, h, l, c, v))
    return rows, malformed, []


def load_calendar(benchmark_path: Path) -> list[int]:
    rows, _, _ = parse_file(benchmark_path)
    return sorted({r.date_i for r in rows})


def group_contiguous_missing(missing_idx: list[int], calendar: list[int]) -> list[tuple[int, int, int]]:
    """Group indices (into `calendar`) that are consecutive into (first_date, last_date, count) runs."""
    runs: list[tuple[int, int]] = []
    start = prev = None
    for idx in missing_idx:
        if start is None:
            start = idx
        elif idx != prev + 1:
            runs.append((start, prev))
            start = idx
        prev = idx
    if start is not None:
        runs.append((start, prev))
    return [(calendar[s], calendar[e], e - s + 1) for s, e in runs]


def validate_file(
    path: Path,
    calendar: list[int],
    calendar_set: set[int],
    jump_threshold: float,
    max_gap_days_for_jump_check: int,
) -> list[Finding]:
    symbol = symbol_from_path(path)
    findings: list[Finding] = []

    rows, malformed, _ = parse_file(path)
    for ln in malformed:
        findings.append(Finding(symbol, "ERROR", "unparseable_row", ln[:120]))

    if not rows:
        if malformed:
            findings.append(Finding(symbol, "ERROR", "empty_file", "no valid data rows"))
        return findings

    by_date: dict[int, list[Row]] = {}
    for r in rows:
        by_date.setdefault(r.date_i, []).append(r)

    for date_i, dup_rows in by_date.items():
        if len(dup_rows) <= 1:
            continue
        distinct = {(r.open_, r.high, r.low, r.close, r.volume) for r in dup_rows}
        d = date_from_int(date_i).isoformat()
        if len(distinct) > 1:
            findings.append(Finding(symbol, "ERROR", "duplicate_date_conflict",
                                     f"{len(dup_rows)} rows for {d} with differing OHLCV values", d))
        else:
            findings.append(Finding(symbol, "WARNING", "duplicate_date_row",
                                     f"{len(dup_rows)} identical rows for {d}", d))

    unique_rows = sorted((dup_rows[-1] for dup_rows in by_date.values()), key=lambda r: r.date_i)

    for r in unique_rows:
        d = date_from_int(r.date_i).isoformat()
        if r.open_ <= 0 or r.high <= 0 or r.low <= 0 or r.close <= 0:
            findings.append(Finding(symbol, "ERROR", "non_positive_price",
                                     f"O={r.open_} H={r.high} L={r.low} C={r.close} on {d}", d))
        if r.high < r.low:
            findings.append(Finding(symbol, "ERROR", "high_lt_low",
                                     f"H={r.high} < L={r.low} on {d}", d))
        elif r.open_ > r.high or r.open_ < r.low:
            findings.append(Finding(symbol, "ERROR", "open_outside_range",
                                     f"O={r.open_} outside [L={r.low}, H={r.high}] on {d}", d))
        elif r.close > r.high or r.close < r.low:
            findings.append(Finding(symbol, "ERROR", "close_outside_range",
                                     f"C={r.close} outside [L={r.low}, H={r.high}] on {d}", d))
        if r.volume < 0:
            findings.append(Finding(symbol, "ERROR", "negative_volume", f"V={r.volume} on {d}", d))
        elif r.volume == 0:
            findings.append(Finding(symbol, "WARNING", "zero_volume", f"V=0 on {d}", d))

    first_date, last_date = unique_rows[0].date_i, unique_rows[-1].date_i
    ticker_dates = {r.date_i for r in unique_rows}

    lo = bisect_left(calendar, first_date)
    hi = bisect_right(calendar, last_date)
    expected_idx = range(lo, hi)
    missing_idx = [i for i in expected_idx if calendar[i] not in ticker_dates]
    for start_d, end_d, count in group_contiguous_missing(missing_idx, calendar):
        if start_d == end_d:
            detail = f"missing 1 trading day: {date_from_int(start_d).isoformat()}"
            dr = date_from_int(start_d).isoformat()
        else:
            detail = (f"missing {count} trading days from "
                      f"{date_from_int(start_d).isoformat()} to {date_from_int(end_d).isoformat()}")
            dr = f"{date_from_int(start_d).isoformat()}..{date_from_int(end_d).isoformat()}"
        findings.append(Finding(symbol, "WARNING", "calendar_gap", detail, dr))

    extra_dates = sorted(d for d in ticker_dates if d not in calendar_set)
    if extra_dates:
        sample = ", ".join(date_from_int(d).isoformat() for d in extra_dates[:5])
        more = f" (+{len(extra_dates) - 5} more)" if len(extra_dates) > 5 else ""
        findings.append(Finding(symbol, "WARNING", "date_not_in_benchmark_calendar",
                                 f"{len(extra_dates)} date(s) not found in benchmark calendar: {sample}{more}"))

    for prev_r, r in zip(unique_rows, unique_rows[1:]):
        gap_days = (date_from_int(r.date_i) - date_from_int(prev_r.date_i)).days
        if gap_days > max_gap_days_for_jump_check:
            continue  # already covered by calendar_gap; a jump across a gap isn't a split signal
        if prev_r.close <= 0:
            continue
        ratio = r.close / prev_r.close
        if ratio <= (1 - jump_threshold) or ratio >= (1 + jump_threshold):
            pct = (ratio - 1) * 100
            d = date_from_int(r.date_i).isoformat()
            findings.append(Finding(
                symbol, "WARNING", "large_day_over_day_move",
                f"{pct:+.1f}% move on {d} (prev_close={prev_r.close}, close={r.close}) -- "
                "verify manually; possible unadjusted split/dividend or bad tick", d,
            ))

    return findings


def iter_data_files(root: Path, dirs: list[str], symbols: set[str] | None) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        sub = root / d
        if not sub.exists():
            continue
        for p in sorted(sub.glob("*.txt")):
            if symbols is not None and symbol_from_path(p) not in symbols:
                continue
            files.append(p)
    return files


def find_benchmark_path(root: Path, dirs: list[str], benchmark: str) -> Path:
    target = benchmark.lower()
    if not target.endswith(".us"):
        target = f"{target}.us"
    for d in dirs:
        candidate = root / d / f"{target}.txt"
        if candidate.exists():
            return candidate
    for d in dirs:
        sub = root / d
        if not sub.exists():
            continue
        for p in sub.glob("*.txt"):
            if symbol_from_path(p) == benchmark.upper() or symbol_from_path(p) == f"{benchmark.upper()}.US":
                return p
    raise SystemExit(f"Could not find benchmark file for {benchmark!r} under {root} in {dirs}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="${workspaceFolder}/data 2/daily/us",
                     help='Root folder containing Stooq-style US folders, e.g. "data 2/daily/us".')
    ap.add_argument("--dirs", default="nyse stocks,nasdaq stocks,etfs",
                     help="Comma-separated subfolders under --root to scan.")
    ap.add_argument("--benchmark", default="SPY.US",
                     help="Symbol used to derive the expected trading-day calendar.")
    ap.add_argument("--symbols", default="",
                     help="Comma-separated symbols to restrict validation to (default: all files).")
    ap.add_argument("--jump-threshold", type=float, default=0.35,
                     help="Flag day-over-day close moves larger than this fraction (default 0.35 = 35%%).")
    ap.add_argument("--max-gap-days-for-jump-check", type=int, default=10,
                     help="Skip the large-move check across calendar-adjacent rows more than this many "
                          "calendar days apart (already covered by the calendar_gap check).")
    ap.add_argument("--out", default="validation_report.csv",
                     help="Path to write the detailed CSV report.")
    ap.add_argument("--fail-on-warning", action="store_true",
                     help="Exit non-zero if any warning is found, not just errors.")
    args = ap.parse_args()

    root = resolve_path(args.root)
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    symbols = None
    if args.symbols.strip():
        symbols = {s.strip().upper() if s.strip().upper().endswith(".US") else f"{s.strip().upper()}.US"
                   for s in args.symbols.split(",") if s.strip()}

    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    benchmark_path = find_benchmark_path(root, dirs, args.benchmark)
    calendar = load_calendar(benchmark_path)
    if not calendar:
        raise SystemExit(f"Benchmark file {benchmark_path} has no valid rows; cannot build a calendar.")
    calendar_set = set(calendar)
    print(f"Calendar: {len(calendar)} trading days from {date_from_int(calendar[0])} "
          f"to {date_from_int(calendar[-1])} (from {benchmark_path.name}).")

    files = iter_data_files(root, dirs, symbols)
    if not files:
        raise SystemExit(f"No .txt files found under {root} in {dirs}.")
    print(f"Scanning {len(files)} file(s)...")

    all_findings: list[Finding] = []
    for path in files:
        all_findings.extend(validate_file(
            path, calendar, calendar_set, args.jump_threshold, args.max_gap_days_for_jump_check,
        ))

    out_path = resolve_path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "severity", "check", "date_range", "detail"])
        for finding in all_findings:
            writer.writerow([finding.symbol, finding.severity, finding.check, finding.date_range, finding.detail])

    errors = [f for f in all_findings if f.severity == "ERROR"]
    warnings_ = [f for f in all_findings if f.severity == "WARNING"]

    by_check: dict[str, int] = {}
    symbols_with_error: set[str] = set()
    symbols_with_warning: set[str] = set()
    for f in all_findings:
        by_check[f.check] = by_check.get(f.check, 0) + 1
        if f.severity == "ERROR":
            symbols_with_error.add(f.symbol)
        else:
            symbols_with_warning.add(f.symbol)

    print()
    print(f"Files scanned:        {len(files)}")
    print(f"Symbols with errors:   {len(symbols_with_error)}")
    print(f"Symbols with warnings: {len(symbols_with_warning)}")
    print(f"Total errors:   {len(errors)}")
    print(f"Total warnings: {len(warnings_)}")
    if by_check:
        print()
        print("By check type:")
        for check, count in sorted(by_check.items(), key=lambda kv: -kv[1]):
            print(f"  {check:30s} {count}")
    print()
    print(f"Full detail written to {out_path}")

    if errors or (args.fail_on_warning and warnings_):
        sys.exit(1)


if __name__ == "__main__":
    main()
