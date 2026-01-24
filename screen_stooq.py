from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------
# Portable path handling
# -----------------------------
def resolve_path(p: str) -> Path:
    """
    Makes paths more portable by supporting:
      - ${workspaceFolder}  (Cursor/VS Code style) -> current working directory
      - ~ home expansion
      - environment variables like $HOME
      - relative paths (resolved from current working directory)
    """
    p = (p or "").strip()
    if "${workspaceFolder}" in p:
        p = p.replace("${workspaceFolder}", str(Path.cwd()))
    p = os.path.expandvars(os.path.expanduser(p))
    return Path(p).resolve()


# -----------------------------
# Fast-ish file tail reader
# -----------------------------
def read_last_lines(path: Path, n: int = 600) -> list[str]:
    """
    Read the last ~n lines of a text file efficiently by seeking from the end.
    """
    block = 64 * 1024  # 64KB
    chunks: list[bytes] = []

    with path.open("rb") as f:
        f.seek(0, 2)  # end
        pos = f.tell()

        lines: list[bytes] = []
        while pos > 0 and len(lines) < n + 20:
            read_size = min(block, pos)
            pos -= read_size
            f.seek(pos)
            chunks.append(f.read(read_size))
            data = b"".join(reversed(chunks))
            lines = data.splitlines()

    tail = lines[-(n + 20):]
    return [ln.decode("utf-8", errors="ignore") for ln in tail]


def load_series_from_file(path: Path, need_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dates_int, close, volume) sorted by date.
    date_int is YYYYMMDD (int).
    """
    lines = read_last_lines(path, n=need_rows)

    rows: list[tuple[int, float, float]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("<TICKER>"):
            continue

        parts = ln.split(",")
        if len(parts) < 9:
            continue

        # TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL, ...
        if parts[1] != "D":
            continue

        try:
            date_i = int(parts[2])
            close = float(parts[7])
            vol = float(parts[8])
        except ValueError:
            continue

        rows.append((date_i, close, vol))

    if not rows:
        return np.array([], dtype=np.int32), np.array([], dtype=float), np.array([], dtype=float)

    rows.sort(key=lambda x: x[0])
    d = np.array([r[0] for r in rows], dtype=np.int32)
    c = np.array([r[1] for r in rows], dtype=float)
    v = np.array([r[2] for r in rows], dtype=float)
    return d, c, v


# -----------------------------
# Indicators
# -----------------------------
def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Standard Wilder RSI.
    Returns an array the same length as close, with NaNs for early periods.
    """
    close = close.astype(float)
    if len(close) < period + 2:
        return np.full_like(close, np.nan, dtype=float)

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(close.shape, np.nan, dtype=float)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # First RSI value corresponds to close index = period
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        idx = i + 1  # gains[i] affects close[i+1]
        if avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray]:
    close = close.astype(float)
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def beta_from_aligned_closes(stock_close: np.ndarray, bench_close: np.ndarray) -> float:
    """
    Beta via daily returns: cov(stock, bench) / var(bench)
    Arrays must already be aligned by date.
    """
    if len(stock_close) < 35 or len(bench_close) < 35:
        return np.nan

    s = stock_close.astype(float)
    b = bench_close.astype(float)

    s_ret = np.diff(s) / s[:-1]
    b_ret = np.diff(b) / b[:-1]

    mask = np.isfinite(s_ret) & np.isfinite(b_ret)
    s_ret = s_ret[mask]
    b_ret = b_ret[mask]
    if len(b_ret) < 30:
        return np.nan

    var_b = np.var(b_ret, ddof=1)
    if var_b == 0:
        return np.nan

    cov = np.cov(s_ret, b_ret, ddof=1)[0, 1]
    return float(cov / var_b)


def month_key(date_int: int) -> int:
    """
    YYYYMM for an int date in YYYYMMDD.
    """
    return (date_int // 10000) * 100 + (date_int // 100) % 100


def monthly_closes(dates: np.ndarray, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce daily series to month-end closes.
    Assumes dates are sorted ascending.
    """
    if len(dates) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=float)

    months: list[int] = []
    vals: list[float] = []

    current_month = None
    current_close = None
    for di, ci in zip(dates, closes):
        mk = month_key(int(di))
        if mk != current_month:
            if current_month is not None:
                months.append(int(current_month))
                vals.append(float(current_close))
            current_month = mk
        current_close = float(ci)

    if current_month is not None:
        months.append(int(current_month))
        vals.append(float(current_close))

    return np.array(months, dtype=np.int32), np.array(vals, dtype=float)


# -----------------------------
# Utilities
# -----------------------------
def build_file_map(root: Path) -> dict[str, Path]:
    """
    Map 'MBSX.US' -> /path/to/mbsx.us.txt for all *.us.txt under root.
    """
    mp: dict[str, Path] = {}
    for p in root.rglob("*.us.txt"):
        sym = p.name.replace(".txt", "").upper()  # MBSX.US
        if not sym.endswith(".US"):
            sym = sym.replace(".US", "") + ".US"
        mp[sym] = p
    return mp


def find_symbol_file(root: Path, sym: str) -> Path | None:
    """
    Find a single symbol file under root (early-exit).
    """
    target = f"{sym.lower()}.txt"
    for p in root.rglob(target):
        return p
    return None


def load_tickers_csv(path: Path) -> list[str]:
    """
    Accepts:
      - CSV with header column named 'symbol' or 'ticker'
      - OR single-column CSV (no header)
    Normalizes to STQ format: 'XYZ.US'
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    header = [c.strip().lower() for c in rows[0]]
    if "symbol" in header:
        col_idx = header.index("symbol")
        data_rows = rows[1:]
    elif "ticker" in header:
        col_idx = header.index("ticker")
        data_rows = rows[1:]
    else:
        col_idx = 0
        data_rows = rows

    out: set[str] = set()
    for r in data_rows:
        if not r:
            continue
        s = r[col_idx].strip().upper()
        if not s:
            continue
        if not s.endswith(".US"):
            s = f"{s}.US"
        out.add(s)

    return sorted(out)


def display_symbol(sym: str) -> str:
    """
    Output symbol without the trailing ".US" portion (case-insensitive).
    Keeps any other dots (e.g., BRK.B.US -> BRK.B).
    """
    s = sym.strip()
    if s.upper().endswith(".US"):
        return s[:-3]
    return s


def fmt2(x: Any) -> str:
    """
    Format numbers to at most 2 decimal places.
    - NaN/None -> empty string
    - ints/floats -> rounded to 2 decimals, always shown with 2 decimals
    """
    if x is None:
        return ""
    try:
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return ""
        val = float(x)
        return f"{val:.2f}"
    except Exception:
        return str(x)


_WORKER_PARAMS: dict[str, Any] = {}
_BENCH_MAP: dict[int, float] = {}
_NEED_ROWS = 0


def _init_worker(bench_map: dict[int, float], params: dict[str, Any], need_rows: int) -> None:
    global _WORKER_PARAMS, _BENCH_MAP, _NEED_ROWS
    _WORKER_PARAMS = params
    _BENCH_MAP = bench_map
    _NEED_ROWS = need_rows


def screen_symbol(
    sym: str,
    path: Path,
    params: dict[str, Any],
    bench_map: dict[int, float],
    need_rows: int,
) -> dict[str, Any] | None:
    d, c, v = load_series_from_file(path, need_rows=need_rows)
    if len(d) < max(params["avg_vol_days"] + 1, 60):
        return None

    # Average daily $ volume filter (close * volume) over avg_vol_days
    close_window = c[-params["avg_vol_days"]:]
    vol_window = v[-params["avg_vol_days"]:]

    avg_volume = float(np.mean(vol_window))
    avg_dollar_volume = float(np.mean(close_window * vol_window))
    if avg_dollar_volume <= params["avg_dollar_vol_min"]:
        return None

    last_close = float(c[-1])

    # RSI filter
    rsi_vals = rsi_wilder(c, period=params["rsi_period"])
    last_rsi = float(rsi_vals[-1])
    if not (params["rsi_low"] <= last_rsi <= params["rsi_high"]):
        return None

    # MACD filter
    macd_line, sig_line = macd(c, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    last_macd = float(macd_line[-1])
    last_sig = float(sig_line[-1])
    if not (last_macd > last_sig):
        return None

    if params["beta_freq"] == "monthly":
        sm, smc = monthly_closes(d, c)
        if len(sm) < params["beta_months"] + 1:
            return None

        stock_aligned: list[float] = []
        bench_aligned: list[float] = []
        for mk, ci in zip(sm, smc):
            bench_ci = bench_map.get(int(mk))
            if bench_ci is None:
                continue
            stock_aligned.append(float(ci))
            bench_aligned.append(float(bench_ci))

        if len(stock_aligned) < params["beta_months"] + 1:
            return None

        stock_close_aligned = np.array(stock_aligned[-(params["beta_months"] + 1):], dtype=float)
        bench_close_aligned = np.array(bench_aligned[-(params["beta_months"] + 1):], dtype=float)
    else:
        # Align stock closes to benchmark dates for beta (no dict/sort needed)
        stock_aligned = []
        bench_aligned = []
        for di, ci in zip(d, c):
            bench_ci = bench_map.get(int(di))
            if bench_ci is None:
                continue
            stock_aligned.append(float(ci))
            bench_aligned.append(float(bench_ci))

        if len(stock_aligned) < params["beta_lookback"] + 1:
            return None

        stock_close_aligned = np.array(stock_aligned[-(params["beta_lookback"] + 1):], dtype=float)
        bench_close_aligned = np.array(bench_aligned[-(params["beta_lookback"] + 1):], dtype=float)

    b = beta_from_aligned_closes(stock_close_aligned, bench_close_aligned)
    if not np.isfinite(b) or b <= params["beta_min"]:
        return None

    return {
        "symbol": display_symbol(sym),  # NO ".US" in output
        "last_close": last_close,
        "beta": b,
        "rsi": last_rsi,
        "macd": last_macd,
        "signal": last_sig,
        "avg_volume": avg_volume,
        "avg_dollar_volume": avg_dollar_volume,
    }


def _screen_symbol_worker(task: tuple[str, Path]) -> dict[str, Any] | None:
    sym, path = task
    return screen_symbol(sym, path, _WORKER_PARAMS, _BENCH_MAP, _NEED_ROWS)


# -----------------------------
# Main screener
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    tickers_group = ap.add_mutually_exclusive_group(required=True)
    tickers_group.add_argument("--tickers", help="CSV with tickers (column: symbol or ticker)")
    tickers_group.add_argument(
        "--tickers_dir",
        help='Directory containing *.us.txt files to screen (e.g. "${workspaceFolder}/data/daily/us/nyse stocks")',
    )
    ap.add_argument("--root", required=True, help='Root folder: e.g. "${workspaceFolder}/data/daily/us" or "/Users/v/Downloads/data/daily/us"')
    ap.add_argument("--out", default="results.csv", help='Output CSV path (supports ${workspaceFolder}, ~, env vars)')

    ap.add_argument("--benchmark", default="SPY.US")

    ap.add_argument(
        "--beta_lookback",
        type=int,
        default=252,
        help="Beta lookback in trading days (default: 252). Used when --beta_freq daily.",
    )
    ap.add_argument(
        "--beta_freq",
        choices=["daily", "monthly"],
        default="daily",
        help="Beta frequency: daily or monthly (default: daily)",
    )
    ap.add_argument(
        "--beta_months",
        type=int,
        default=60,
        help="Beta lookback in months (default: 60). Used when --beta_freq monthly.",
    )
    ap.add_argument("--beta_min", type=float, default=1.2, help="Minimum beta (default: 1.2)")

    ap.add_argument("--rsi_low", type=float, default=50.0)
    ap.add_argument("--rsi_high", type=float, default=70.0)
    ap.add_argument("--rsi_period", type=int, default=14)

    ap.add_argument("--macd_fast", type=int, default=12)
    ap.add_argument("--macd_slow", type=int, default=26)
    ap.add_argument("--macd_signal", type=int, default=9)

    ap.add_argument("--avg_vol_days", type=int, default=20, help="Window for avg volume (shares) (default: 20)")

    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for symbol screening (0=auto, 1=disable parallelism)",
    )

    # Average daily dollar volume filter (close * volume)
    ap.add_argument(
        "--avg_dollar_vol_min",
        type=float,
        default=5_000_000.0,
        help="Minimum average daily $ volume over avg_vol_days (close * volume). Default: 5,000,000",
    )

    args = ap.parse_args()

    root = resolve_path(args.root)
    out_path = resolve_path(args.out)

    if args.tickers:
        tickers_path = resolve_path(args.tickers)
        tickers = load_tickers_csv(tickers_path)
        symbol_paths = build_file_map(root)
    else:
        tickers_root = resolve_path(args.tickers_dir)
        symbol_paths = build_file_map(tickers_root)
        tickers = sorted(symbol_paths.keys())

    bench_sym = args.benchmark.strip().upper()
    if not bench_sym.endswith(".US"):
        bench_sym += ".US"

    bench_path = symbol_paths.get(bench_sym)
    if bench_path is None:
        bench_path = find_symbol_file(root, bench_sym)

    if bench_path is None:
        raise SystemExit(
            f"Benchmark {bench_sym} not found under {root}.\n"
            f"Tip: point --root at a parent folder that includes both stocks + ETFs."
        )

    beta_rows = args.beta_lookback + 10
    if args.beta_freq == "monthly":
        beta_rows = args.beta_months * 23 + 10

    # Read enough rows from each file to compute beta + indicators + averages.
    need_rows = max(
        beta_rows,
        args.macd_slow + args.macd_signal + 30,
        args.avg_vol_days + 30,
        220,
    )

    # Load benchmark series (tail), build date->close map or month->close map
    bd, bc, _bv = load_series_from_file(bench_path, need_rows=need_rows)
    if args.beta_freq == "monthly":
        bm, bmc = monthly_closes(bd, bc)
        if len(bm) < args.beta_months + 1:
            raise SystemExit(
                f"Not enough benchmark history for {bench_sym} "
                f"(have {len(bm)} months, need {args.beta_months + 1})."
            )
        bm = bm[-(args.beta_months + 1):]
        bmc = bmc[-(args.beta_months + 1):]
        bench_map = {int(mi): float(ci) for mi, ci in zip(bm, bmc)}
    else:
        if len(bd) < args.beta_lookback + 1:
            raise SystemExit(f"Not enough benchmark history for {bench_sym} (have {len(bd)} rows).")

        bd = bd[-(args.beta_lookback + 10):]
        bc = bc[-(args.beta_lookback + 10):]
        bench_map = {int(di): float(ci) for di, ci in zip(bd, bc)}

    tasks: list[tuple[str, Path]] = []
    for sym in tickers:
        p = symbol_paths.get(sym)
        if not p:
            continue
        tasks.append((sym, p))

    params = {
        "beta_freq": args.beta_freq,
        "beta_lookback": args.beta_lookback,
        "beta_months": args.beta_months,
        "beta_min": args.beta_min,
        "rsi_low": args.rsi_low,
        "rsi_high": args.rsi_high,
        "rsi_period": args.rsi_period,
        "macd_fast": args.macd_fast,
        "macd_slow": args.macd_slow,
        "macd_signal": args.macd_signal,
        "avg_vol_days": args.avg_vol_days,
        "avg_dollar_vol_min": args.avg_dollar_vol_min,
    }

    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers > 0 else max(1, cpu_count - 1)
    workers = min(workers, len(tasks)) if tasks else 1

    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(bench_map, params, need_rows),
        ) as ex:
            for res in ex.map(_screen_symbol_worker, tasks, chunksize=chunksize):
                if res:
                    results.append(res)
    else:
        for sym, p in tasks:
            res = screen_symbol(sym, p, params, bench_map, need_rows)
            if res:
                results.append(res)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["symbol", "last_close", "beta", "rsi", "macd", "signal", "avg_volume", "avg_dollar_volume"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(results, key=lambda x: str(x["symbol"])):
            # Round all numeric fields to 2 decimals (max)
            w.writerow(
                {
                    "symbol": row["symbol"],
                    "last_close": fmt2(row["last_close"]),
                    "beta": fmt2(row["beta"]),
                    "rsi": fmt2(row["rsi"]),
                    "macd": fmt2(row["macd"]),
                    "signal": fmt2(row["signal"]),
                    "avg_volume": fmt2(row["avg_volume"]),
                    "avg_dollar_volume": fmt2(row["avg_dollar_volume"]),
                }
            )

    print(f"Wrote {len(results)} matches to {out_path}")


if __name__ == "__main__":
    main()
