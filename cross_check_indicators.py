"""
Cross-checks screen_stooq.py's hand-rolled, lfilter-vectorized RSI/ATR/MACD/beta
against independent, plain-loop reference implementations of the exact same
documented formulas (Wilder RSI/ATR with SMA-seeded smoothing, EMA seeded at
the first close for MACD, beta via covariance/variance also re-derived with
scipy's least-squares regression).

The point isn't to relitigate which Wilder convention is "more standard" --
it's to catch implementation bugs in the vectorization itself (off-by-one
seeding, wrong alpha, wrong initial IIR filter state) by comparing against a
completely different, easy-to-eyeball code path computing the same spec.

This script only reads data 2/; it never writes to it. It imports
screen_stooq.py as a module to test the actual functions used in production
(safe: screen_stooq.py's own driver code is guarded by `if __name__ ==
"__main__"`).

Usage:
  python3 cross_check_indicators.py --root "data 2/daily/us"
  python3 cross_check_indicators.py --root "data 2/daily/us" --symbols AAPL,MSFT
  python3 cross_check_indicators.py --root "data 2/daily/us" --sample-size 50 --out report.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress

import screen_stooq as ss


def resolve_path(p: str) -> Path:
    p = (p or "").strip()
    if "${workspaceFolder}" in p:
        p = p.replace("${workspaceFolder}", str(Path.cwd()))
    p = os.path.expandvars(os.path.expanduser(p))
    return Path(p).resolve()


def symbol_from_path(path: Path) -> str:
    name = path.name
    return name[:-4].upper() if name.lower().endswith(".us.txt") else path.stem.upper()


# -----------------------------
# Independent reference implementations (plain loops, no lfilter)
# -----------------------------
def ref_ema(x: np.ndarray, span: int) -> np.ndarray:
    n = len(x)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ref_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)
    if n < period + 2:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    rsi[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


def ref_atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return atr

    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    avg = float(np.mean(tr[1: period + 1]))
    atr[period] = avg
    for i in range(period + 1, n):
        avg = (avg * (period - 1) + tr[i]) / period
        atr[i] = avg

    return atr


def ref_macd(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray]:
    macd_line = ref_ema(close, fast) - ref_ema(close, slow)
    signal_line = ref_ema(macd_line, signal)
    return macd_line, signal_line


def ref_beta(stock_close: np.ndarray, bench_close: np.ndarray) -> float:
    if len(stock_close) < 35 or len(bench_close) < 35:
        return float("nan")
    s_ret = np.diff(stock_close) / stock_close[:-1]
    b_ret = np.diff(bench_close) / bench_close[:-1]
    mask = np.isfinite(s_ret) & np.isfinite(b_ret)
    s_ret, b_ret = s_ret[mask], b_ret[mask]
    if len(b_ret) < 30 or np.var(b_ret, ddof=1) == 0:
        return float("nan")
    # Independent code path: least-squares slope of stock returns on bench returns.
    slope, _intercept, _r, _p, _se = linregress(b_ret, s_ret)
    return float(slope)


# -----------------------------
# Diffing
# -----------------------------
def diff_series(actual: np.ndarray, ref: np.ndarray) -> tuple[float, float, int]:
    """Returns (max_abs_diff, max_rel_diff, nan_pattern_mismatches) over the shorter length."""
    n = min(len(actual), len(ref))
    a, r = actual[:n], ref[:n]
    a_nan, r_nan = np.isnan(a), np.isnan(r)
    mismatches = int(np.sum(a_nan != r_nan))
    both_valid = (~a_nan) & (~r_nan)
    if not np.any(both_valid):
        return 0.0, 0.0, mismatches
    diff = np.abs(a[both_valid] - r[both_valid])
    denom = np.maximum(np.abs(r[both_valid]), 1e-12)
    return float(np.max(diff)), float(np.max(diff / denom)), mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="${workspaceFolder}/data 2/daily/us")
    ap.add_argument("--dirs", default="nyse stocks,etfs")
    ap.add_argument("--benchmark", default="SPY.US")
    ap.add_argument("--symbols", default="", help="Comma-separated symbols to test (default: random sample).")
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--beta-lookback", type=int, default=252)
    ap.add_argument("--abs-tolerance", type=float, default=1e-6)
    ap.add_argument("--rel-tolerance", type=float, default=1e-6)
    ap.add_argument("--out", default="cross_check_report.csv")
    args = ap.parse_args()

    root = resolve_path(args.root)
    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    all_files: dict[str, Path] = {}
    for d in dirs:
        sub = root / d
        if not sub.exists():
            continue
        for p in sub.glob("*.txt"):
            all_files[symbol_from_path(p)] = p

    benchmark_symbol = args.benchmark.upper()
    if not benchmark_symbol.endswith(".US"):
        benchmark_symbol = f"{benchmark_symbol}.US"
    bench_path = all_files.get(benchmark_symbol)
    if bench_path is None:
        raise SystemExit(f"Benchmark {benchmark_symbol} not found under {root}.")
    bd, _bo, _bh, _bl, bc = ss.load_ohlc_from_file(bench_path)
    bench_map = {int(d): float(c) for d, c in zip(bd, bc)}

    if args.symbols.strip():
        symbols = [s.strip().upper() if s.strip().upper().endswith(".US") else f"{s.strip().upper()}.US"
                   for s in args.symbols.split(",") if s.strip()]
        missing = [s for s in symbols if s not in all_files]
        if missing:
            raise SystemExit(f"Symbols not found: {missing}")
    else:
        candidates = sorted(s for s in all_files if s != benchmark_symbol)
        rng = random.Random(args.seed)
        symbols = rng.sample(candidates, min(args.sample_size, len(candidates)))

    print(f"Cross-checking {len(symbols)} symbol(s) against benchmark {benchmark_symbol}.")

    rows: list[dict] = []
    tolerance_failures = 0

    for symbol in symbols:
        d, o, h, l, c = ss.load_ohlc_from_file(all_files[symbol])
        if len(c) < args.rsi_period + 2:
            continue

        checks = []

        rsi_actual = ss.rsi_wilder(c, period=args.rsi_period)
        rsi_ref = ref_rsi_wilder(c, period=args.rsi_period)
        checks.append(("RSI", rsi_actual, rsi_ref))

        atr_actual = ss.atr_wilder(h, l, c, period=args.atr_period)
        atr_ref = ref_atr_wilder(h, l, c, period=args.atr_period)
        checks.append(("ATR", atr_actual, atr_ref))

        macd_actual, sig_actual = ss.macd(c, args.macd_fast, args.macd_slow, args.macd_signal)
        macd_ref, sig_ref = ref_macd(c, args.macd_fast, args.macd_slow, args.macd_signal)
        checks.append(("MACD_line", macd_actual, macd_ref))
        checks.append(("MACD_signal", sig_actual, sig_ref))

        for name, actual, ref in checks:
            max_abs, max_rel, nan_mismatches = diff_series(actual, ref)
            fails = nan_mismatches > 0 or (max_abs > args.abs_tolerance and max_rel > args.rel_tolerance)
            if fails:
                tolerance_failures += 1
            rows.append({
                "symbol": symbol, "check": name, "max_abs_diff": max_abs,
                "max_rel_diff": max_rel, "nan_pattern_mismatches": nan_mismatches,
                "status": "FAIL" if fails else "ok",
            })

        stock_aligned, bench_aligned = [], []
        for di, ci in zip(d, c):
            bc_ = bench_map.get(int(di))
            if bc_ is None:
                continue
            stock_aligned.append(float(ci))
            bench_aligned.append(float(bc_))
        if len(stock_aligned) >= args.beta_lookback + 1:
            s_arr = np.array(stock_aligned[-(args.beta_lookback + 1):], dtype=float)
            b_arr = np.array(bench_aligned[-(args.beta_lookback + 1):], dtype=float)
            beta_actual = ss.beta_from_aligned_closes(s_arr, b_arr)
            beta_ref = ref_beta(s_arr, b_arr)
            if np.isfinite(beta_actual) and np.isfinite(beta_ref):
                abs_diff = abs(beta_actual - beta_ref)
                rel_diff = abs_diff / max(abs(beta_ref), 1e-12)
                fails = abs_diff > args.abs_tolerance and rel_diff > args.rel_tolerance
                if fails:
                    tolerance_failures += 1
                rows.append({
                    "symbol": symbol, "check": "beta", "max_abs_diff": abs_diff,
                    "max_rel_diff": rel_diff, "nan_pattern_mismatches": 0,
                    "status": "FAIL" if fails else "ok",
                })

    out_path = resolve_path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "check", "max_abs_diff", "max_rel_diff",
                                                 "nan_pattern_mismatches", "status"])
        writer.writeheader()
        writer.writerows(rows)

    by_check: dict[str, list[float]] = {}
    for row in rows:
        by_check.setdefault(row["check"], []).append(row["max_abs_diff"])

    print()
    print(f"{'Check':14s} {'n':>5s} {'max_abs_diff':>14s}")
    for check, diffs in by_check.items():
        print(f"{check:14s} {len(diffs):5d} {max(diffs):14.3e}")
    print()
    print(f"Total comparisons: {len(rows)}")
    print(f"Failures (exceed both abs and rel tolerance, or NaN-pattern mismatch): {tolerance_failures}")
    print(f"Full detail written to {out_path}")

    if tolerance_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
