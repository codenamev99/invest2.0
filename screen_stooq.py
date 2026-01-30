from __future__ import annotations

import argparse
import calendar
import csv
import os
from datetime import date
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


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


def date_from_int(date_int: int) -> date:
    """
    Convert YYYYMMDD int to date.
    """
    year = date_int // 10000
    month = (date_int // 100) % 100
    day = date_int % 100
    return date(year, month, day)


def date_to_int(d: date) -> int:
    """
    Convert date to YYYYMMDD int.
    """
    return d.year * 10000 + d.month * 100 + d.day


def shift_months(d: date, months: int) -> date:
    """
    Shift date by N months, clamping the day if needed.
    """
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


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


def pct_change(current: float, previous: float) -> float:
    """
    Percent change from previous to current.
    Returns NaN when previous is 0 or either value is non-finite.
    """
    if not (np.isfinite(current) and np.isfinite(previous)):
        return np.nan
    if previous == 0:
        return np.nan
    return (current - previous) / previous * 100.0


def unique_sheet_name(wb: Workbook, base: str) -> str:
    """
    Return a unique worksheet name (<= 31 chars) for the workbook.
    """
    base = base.strip()[:31] or "Results"
    if base not in wb.sheetnames:
        return base

    idx = 2
    while True:
        suffix = f" ({idx})"
        trimmed = base[: 31 - len(suffix)]
        name = f"{trimmed}{suffix}"
        if name not in wb.sheetnames:
            return name
        idx += 1


def is_empty_sheet(ws) -> bool:
    """
    Returns True if the worksheet has no values.
    """
    return ws.max_row == 1 and ws.max_column == 1 and ws["A1"].value is None


def auto_size_columns(ws, min_width: int = 8, max_width: int = 40) -> None:
    """
    Auto-size worksheet column widths based on cell contents.
    """
    for col_idx, col_cells in enumerate(ws.iter_cols(min_col=1, max_col=ws.max_column), start=1):
        max_len = 0
        for cell in col_cells:
            if cell.value is None:
                continue
            value_str = str(cell.value)
            if "\n" in value_str:
                value_len = max(len(line) for line in value_str.splitlines())
            else:
                value_len = len(value_str)
            if value_len > max_len:
                max_len = value_len
        if max_len == 0:
            continue
        width = min(max_width, max(min_width, max_len + 2))
        ws.column_dimensions[get_column_letter(col_idx)].width = width


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
    min_rows = 60
    if params["avg_vol_mode"] == "days":
        min_rows = max(params["avg_vol_days"] + 1, min_rows)
    if len(d) < min_rows:
        return None

    change_lookback = 5  # trading periods
    prev_idx = -(change_lookback + 1)

    # Average daily $ volume filter (close * volume) over avg volume window
    if params["avg_vol_mode"] == "months":
        last_date = date_from_int(int(d[-1]))
        cutoff_date = shift_months(last_date, -int(params["avg_vol_months"]))
        cutoff_int = date_to_int(cutoff_date)
        if d[0] > cutoff_int:
            return None
        mask = d >= cutoff_int
        if not np.any(mask):
            return None
        close_window = c[mask]
        vol_window = v[mask]
    else:
        close_window = c[-params["avg_vol_days"]:]
        vol_window = v[-params["avg_vol_days"]:]

    avg_dollar_volume = float(np.mean(close_window * vol_window))
    if avg_dollar_volume <= params["avg_dollar_vol_min"]:
        return None

    avg_dollar_volume_prev = np.nan
    if len(d) >= change_lookback + 1:
        if params["avg_vol_mode"] == "months":
            last_date_prev = date_from_int(int(d[prev_idx]))
            cutoff_date_prev = shift_months(last_date_prev, -int(params["avg_vol_months"]))
            cutoff_int_prev = date_to_int(cutoff_date_prev)
            last_date_prev_int = date_to_int(last_date_prev)
            mask_prev = (d >= cutoff_int_prev) & (d <= last_date_prev_int)
            if np.any(mask_prev):
                avg_dollar_volume_prev = float(np.mean(c[mask_prev] * v[mask_prev]))
        else:
            lookback_days = int(params["avg_vol_days"])
            if len(c) >= lookback_days + change_lookback:
                close_window_prev = c[-(lookback_days + change_lookback):-change_lookback]
                vol_window_prev = v[-(lookback_days + change_lookback):-change_lookback]
                if len(close_window_prev) and len(vol_window_prev):
                    avg_dollar_volume_prev = float(np.mean(close_window_prev * vol_window_prev))

    last_close = float(c[-1])
    prev_close = float(c[prev_idx]) if len(c) >= change_lookback + 1 else np.nan

    # RSI filter
    rsi_vals = rsi_wilder(c, period=params["rsi_period"])
    last_rsi = float(rsi_vals[-1])
    prev_rsi = float(rsi_vals[prev_idx]) if len(rsi_vals) >= change_lookback + 1 else np.nan
    if not (params["rsi_low"] <= last_rsi <= params["rsi_high"]):
        return None

    # MACD filter
    macd_line, sig_line = macd(c, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    last_macd = float(macd_line[-1])
    last_sig = float(sig_line[-1])
    prev_macd = float(macd_line[prev_idx]) if len(macd_line) >= change_lookback + 1 else np.nan
    prev_sig = float(sig_line[prev_idx]) if len(sig_line) >= change_lookback + 1 else np.nan
    if not (last_macd > last_sig):
        return None

    macd_signal_ratio = np.nan
    if np.isfinite(last_macd) and np.isfinite(last_sig) and last_sig != 0:
        macd_signal_ratio = last_macd / last_sig
    macd_signal_ratio_prev = np.nan
    if np.isfinite(prev_macd) and np.isfinite(prev_sig) and prev_sig != 0:
        macd_signal_ratio_prev = prev_macd / prev_sig

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
        beta_prev = np.nan
        if len(stock_aligned) >= params["beta_months"] + 1 + change_lookback:
            prev_stock = np.array(stock_aligned[:-change_lookback], dtype=float)
            prev_bench = np.array(bench_aligned[:-change_lookback], dtype=float)
            if len(prev_stock) >= params["beta_months"] + 1:
                beta_prev = beta_from_aligned_closes(
                    prev_stock[-(params["beta_months"] + 1):],
                    prev_bench[-(params["beta_months"] + 1):],
                )
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
        beta_prev = np.nan
        if len(stock_aligned) >= params["beta_lookback"] + 1 + change_lookback:
            prev_stock = np.array(stock_aligned[:-change_lookback], dtype=float)
            prev_bench = np.array(bench_aligned[:-change_lookback], dtype=float)
            if len(prev_stock) >= params["beta_lookback"] + 1:
                beta_prev = beta_from_aligned_closes(
                    prev_stock[-(params["beta_lookback"] + 1):],
                    prev_bench[-(params["beta_lookback"] + 1):],
                )

    b = beta_from_aligned_closes(stock_close_aligned, bench_close_aligned)
    if not np.isfinite(b) or b <= params["beta_min"]:
        return None

    close_pct_5 = pct_change(last_close, prev_close)
    beta_pct_5 = pct_change(b, beta_prev)
    rsi_pct_5 = pct_change(last_rsi, prev_rsi)
    macd_pct_5 = pct_change(last_macd, prev_macd)
    signal_pct_5 = pct_change(last_sig, prev_sig)
    macd_signal_ratio_pct_5 = pct_change(macd_signal_ratio, macd_signal_ratio_prev)
    avg_dollar_volume_pct_5 = pct_change(avg_dollar_volume, avg_dollar_volume_prev)

    return {
        "symbol": display_symbol(sym),  # NO ".US" in output
        "last_close": last_close,
        "last_close_pct_5": close_pct_5,
        "beta": b,
        "beta_pct_5": beta_pct_5,
        "rsi": last_rsi,
        "rsi_pct_5": rsi_pct_5,
        "macd": last_macd,
        "macd_pct_5": macd_pct_5,
        "signal": last_sig,
        "signal_pct_5": signal_pct_5,
        "macd_signal_ratio": macd_signal_ratio,
        "macd_signal_ratio_pct_5": macd_signal_ratio_pct_5,
        "avg_dollar_volume": avg_dollar_volume,
        "avg_dollar_volume_pct_5": avg_dollar_volume_pct_5,
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
    ap.add_argument(
        "--out",
        default="results.xlsx",
        help='Output Excel (.xlsx) path (supports ${workspaceFolder}, ~, env vars)',
    )

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
    ap.add_argument("--macd_signal", type=int, default=12)

    ap.add_argument(
        "--avg_vol_days",
        type=int,
        default=None,
        help="Window for avg volume (trading days). Overrides --avg_vol_months when set.",
    )
    ap.add_argument(
        "--avg_vol_months",
        type=int,
        default=6,
        help="Window for avg volume (calendar months) (default: 6)",
    )

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
        help="Minimum average daily $ volume over avg volume window (close * volume). Default: 5,000,000",
    )

    args = ap.parse_args()

    if args.avg_vol_days is not None and args.avg_vol_days <= 0:
        raise SystemExit("--avg_vol_days must be > 0")
    if args.avg_vol_months <= 0:
        raise SystemExit("--avg_vol_months must be > 0")

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

    if args.avg_vol_days is not None and args.avg_vol_days > 0:
        avg_vol_mode = "days"
        avg_vol_days = args.avg_vol_days
        avg_vol_months = 0
        avg_vol_need_rows = args.avg_vol_days + 30
    else:
        avg_vol_mode = "months"
        avg_vol_days = 0
        avg_vol_months = args.avg_vol_months
        avg_vol_need_rows = args.avg_vol_months * 23 + 30

    # Read enough rows from each file to compute beta + indicators + averages.
    need_rows = max(
        beta_rows,
        args.macd_slow + args.macd_signal + 30,
        avg_vol_need_rows,
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
        "avg_vol_mode": avg_vol_mode,
        "avg_vol_days": avg_vol_days,
        "avg_vol_months": avg_vol_months,
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
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")

    fieldnames = [
        "Symbol",
        "Close $",
        "Close 5D %",
        "Beta",
        "Beta 5D %",
        "RSI",
        "RSI 5D %",
        "MACD",
        "MACD 5D %",
        "Signal",
        "Signal 5D %",
        "MACD/Signal",
        "MACD/Signal 5D %",
        "Avg $ Vol",
        "Avg $ Vol 5D %",
    ]
    data_keys = [
        "symbol",
        "last_close",
        "last_close_pct_5",
        "beta",
        "beta_pct_5",
        "rsi",
        "rsi_pct_5",
        "macd",
        "macd_pct_5",
        "signal",
        "signal_pct_5",
        "macd_signal_ratio",
        "macd_signal_ratio_pct_5",
        "avg_dollar_volume",
        "avg_dollar_volume_pct_5",
    ]
    data_date = date_from_int(int(bd[-1])) if len(bd) else date.today()
    headline = data_date.strftime("%d %b %Y").upper()

    if out_path.exists():
        wb = load_workbook(out_path)
    else:
        wb = Workbook()

    sheet_name = unique_sheet_name(wb, headline)
    if len(wb.sheetnames) == 1 and is_empty_sheet(wb.active):
        ws = wb.active
        ws.title = sheet_name
    else:
        ws = wb.create_sheet(title=sheet_name)

    ws.append(fieldnames)

    if args.beta_freq == "monthly":
        beta_desc = f"{args.beta_months} months\n> {args.beta_min}"
    else:
        beta_desc = f"{args.beta_lookback} days\n> {args.beta_min}"

    if avg_vol_mode == "months":
        avg_desc = f"{avg_vol_months} months\n> ${args.avg_dollar_vol_min:,.0f}"
    else:
        avg_desc = f"{avg_vol_days} days\n> ${args.avg_dollar_vol_min:,.0f}"

    pct_desc = "5 days %"
    beta_change_desc = "5 months %" if args.beta_freq == "monthly" else pct_desc
    descriptors = [
        "",
        "",
        pct_desc,
        beta_desc,
        beta_change_desc,
        f"{args.rsi_period} days\n{args.rsi_low} to {args.rsi_high}",
        pct_desc,
        f"{args.macd_fast}/{args.macd_slow} EMA\nMACD > Signal",
        pct_desc,
        f"{args.macd_signal} days",
        pct_desc,
        "MACD / Signal",
        pct_desc,
        avg_desc,
        pct_desc,
    ]
    ws.append(descriptors)
    for row in sorted(results, key=lambda x: str(x["symbol"])):
        # Round all numeric fields to 2 decimals (max)
        ws.append(
            [
                row["symbol"],
                float(row["last_close"]),
                fmt2(row["last_close_pct_5"]),
                fmt2(row["beta"]),
                fmt2(row["beta_pct_5"]),
                fmt2(row["rsi"]),
                fmt2(row["rsi_pct_5"]),
                fmt2(row["macd"]),
                fmt2(row["macd_pct_5"]),
                fmt2(row["signal"]),
                fmt2(row["signal_pct_5"]),
                fmt2(row["macd_signal_ratio"]),
                fmt2(row["macd_signal_ratio_pct_5"]),
                float(row["avg_dollar_volume"]),
                fmt2(row["avg_dollar_volume_pct_5"]),
            ]
        )

    if ws.max_row > 1:
        is_row1_empty = all(cell.value is None for cell in ws[1])
        if is_row1_empty:
            ws.delete_rows(1, 1)

    header_fill = PatternFill(fill_type="solid", fgColor="000000")
    header_font = Font(bold=False, color="FFFFFF", size=13)
    ws.row_dimensions[1].height = 26
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    descriptor_fill = PatternFill(fill_type="solid", fgColor="D9D9D9")
    descriptor_font = Font(italic=True, color="000000", size=11)
    for cell in ws[2]:
        cell.font = descriptor_font
        cell.fill = descriptor_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.freeze_panes = "A3"

    last_close_col_idx = data_keys.index("last_close") + 1
    avg_col_idx = data_keys.index("avg_dollar_volume") + 1
    base_row_height = ws.sheet_format.defaultRowHeight or 15
    zebra_fill = PatternFill(fill_type="solid", fgColor="F7F7F7")
    pct_col_idxs = [idx + 1 for idx, key in enumerate(data_keys) if key.endswith("_pct_5")]
    thin_side = Side(style="thin", color="000000")
    thick_side = Side(style="thick", color="000000")
    vertical_border = Border(left=thin_side, right=thin_side)
    separator_border = Border(left=thin_side, right=thick_side)
    next_separator_border = Border(left=thick_side, right=thin_side)
    for row_idx in range(3, ws.max_row + 1):
        last_close_cell = ws.cell(row=row_idx, column=last_close_col_idx)
        if last_close_cell.value is not None:
            last_close_cell.number_format = "$#,##0.00"

        avg_cell = ws.cell(row=row_idx, column=avg_col_idx)
        if avg_cell.value is not None:
            avg_cell.number_format = "$#,##0.00"

        first_col_cell = ws.cell(row=row_idx, column=1)
        first_col_cell.font = Font(color="FFFFFF", size=11)
        first_col_cell.fill = header_fill

        if row_idx % 2 == 0:
            for col_idx in range(2, ws.max_column + 1):
                ws.cell(row=row_idx, column=col_idx).fill = zebra_fill

        for col_idx in range(1, ws.max_column + 1):
            if col_idx in pct_col_idxs:
                border = separator_border
            elif (col_idx - 1) in pct_col_idxs:
                border = next_separator_border
            else:
                border = vertical_border
            ws.cell(row=row_idx, column=col_idx).border = border

        current_height = ws.row_dimensions[row_idx].height or base_row_height
        ws.row_dimensions[row_idx].height = current_height + 2

    auto_size_columns(ws)

    wb.save(out_path)

    print(f"Wrote {len(results)} matches to {out_path} ({sheet_name})")


if __name__ == "__main__":
    main()
