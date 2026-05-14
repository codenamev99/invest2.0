from __future__ import annotations

import argparse
import calendar
import csv
import io
import os
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.signal import lfilter
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Color, Font, PatternFill, Side
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
    target = n + 20
    newline_count = 0

    with path.open("rb") as f:
        f.seek(0, 2)  # end
        pos = f.tell()
        while pos > 0 and newline_count < target:
            read_size = min(block, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    # Join once at the end instead of on every loop iteration
    data = b"".join(reversed(chunks))
    lines = data.splitlines()
    tail = lines[-target:]
    return [ln.decode("utf-8", errors="ignore") for ln in tail]


def load_series_from_file(
    path: Path,
    need_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dates_int, close, volume, high, low) sorted by date.
    date_int is YYYYMMDD (int).
    """
    lines = read_last_lines(path, n=need_rows)

    rows: list[tuple[int, float, float, float, float]] = []
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
            high = float(parts[5])
            low = float(parts[6])
            close = float(parts[7])
            vol = float(parts[8])
        except ValueError:
            continue

        rows.append((date_i, close, vol, high, low))

    if not rows:
        empty = np.array([], dtype=float)
        return np.array([], dtype=np.int32), empty, empty, empty, empty

    rows.sort(key=lambda x: x[0])
    d = np.array([r[0] for r in rows], dtype=np.int32)
    c = np.array([r[1] for r in rows], dtype=float)
    v = np.array([r[2] for r in rows], dtype=float)
    h = np.array([r[3] for r in rows], dtype=float)
    l = np.array([r[4] for r in rows], dtype=float)
    return d, c, v, h, l


def load_ohlc_from_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dates_int, open, high, low, close) sorted by date.
    Reads the full file so validation can look forward from older ranking dates.
    """
    rows: list[tuple[int, float, float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("<TICKER>"):
                continue

            parts = ln.split(",")
            if len(parts) < 9 or parts[1] != "D":
                continue

            try:
                date_i = int(parts[2])
                open_ = float(parts[4])
                high = float(parts[5])
                low = float(parts[6])
                close = float(parts[7])
            except ValueError:
                continue

            rows.append((date_i, open_, high, low, close))

    if not rows:
        empty = np.array([], dtype=float)
        return np.array([], dtype=np.int32), empty, empty, empty, empty

    rows.sort(key=lambda x: x[0])
    d = np.array([r[0] for r in rows], dtype=np.int32)
    o = np.array([r[1] for r in rows], dtype=float)
    h = np.array([r[2] for r in rows], dtype=float)
    l = np.array([r[3] for r in rows], dtype=float)
    c = np.array([r[4] for r in rows], dtype=float)
    return d, o, h, l, c


# -----------------------------
# Indicators
# -----------------------------
def ema(x: np.ndarray, span: int) -> np.ndarray:
    if len(x) == 0:
        return np.empty(0, dtype=float)
    x = x.astype(float)
    alpha = 2.0 / (span + 1.0)
    # IIR filter: y[n] = alpha*x[n] + (1-alpha)*y[n-1], seeded at x[0].
    # Direct Form II transposed initial state: zi = [(1-alpha)*x[0]]
    # gives y[0] = alpha*x[0] + (1-alpha)*x[0] = x[0].
    zi = np.array([(1.0 - alpha) * x[0]])
    out, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)
    return out


def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Standard Wilder RSI.
    Returns an array the same length as close, with NaNs for early periods.
    """
    close = close.astype(float)
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan, dtype=float)

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(n, np.nan, dtype=float)

    # Seed: SMA of first `period` values
    avg_gain_seed = float(np.mean(gains[:period]))
    avg_loss_seed = float(np.mean(losses[:period]))

    # First RSI value corresponds to close index = period
    if avg_loss_seed == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain_seed / avg_loss_seed)

    # Vectorized Wilder smoothing (alpha = 1/period) for gains[period:]
    # Direct Form II transposed zi: zi = [(1-alpha)*seed] initialises the filter
    # so that the first output equals Wilder(seed, gains[period]).
    gains_tail = gains[period:]
    losses_tail = losses[period:]
    if len(gains_tail) > 0:
        alpha = 1.0 / period
        zi_g = np.array([(1.0 - alpha) * avg_gain_seed])
        zi_l = np.array([(1.0 - alpha) * avg_loss_seed])
        avg_gains, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], gains_tail, zi=zi_g)
        avg_losses, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], losses_tail, zi=zi_l)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_losses != 0, avg_gains / avg_losses, np.inf)
            rsi[period + 1: period + 1 + len(gains_tail)] = np.where(
                avg_losses == 0, 100.0, 100.0 - 100.0 / (1.0 + rs)
            )

    return rsi


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range (Wilder). Returns array aligned to close with NaNs early.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    n = len(close)
    atr = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return atr

    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    # Seed: SMA of tr[1..period]
    seed = float(np.mean(tr[1: period + 1]))
    atr[period] = seed

    # Vectorized Wilder smoothing (alpha = 1/period) for tr[period+1:]
    tr_tail = tr[period + 1:]
    if len(tr_tail) > 0:
        alpha = 1.0 / period
        zi = np.array([(1.0 - alpha) * seed])
        atr_tail, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], tr_tail, zi=zi)
        atr[period + 1:] = atr_tail

    return atr


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

    dates_int = dates.astype(np.int32)
    # YYYYMM for each date
    month_keys_arr = (dates_int // 10000) * 100 + (dates_int // 100) % 100
    # Last index for each unique month (dates are sorted, so searchsorted gives last occurrence)
    unique_months = np.unique(month_keys_arr)
    last_indices = np.searchsorted(month_keys_arr, unique_months, side="right") - 1
    return unique_months, closes[last_indices].astype(float)


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


def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if not s:
        return s
    if not s.endswith(".US"):
        s = f"{s}.US"
    return s


def prompt_run_mode() -> tuple[str, str | None]:
    print("Run mode menu:")
    print("1) All tickers")
    print("2) Specific ticker")
    while True:
        choice = input("Select option (1/2): ").strip()
        if choice in ("1", "2"):
            break
        print("Please enter 1 or 2.")
    if choice == "2":
        while True:
            ticker = input("Enter ticker symbol (e.g., AAPL or AAPL.US): ").strip()
            if ticker:
                return "single", normalize_symbol(ticker)
            print("Ticker cannot be empty.")
    return "all", None


NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
}
ALPHAVANTAGE_IPO_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_API_KEY = "F7HUZ9ETATI052FB"
UPCOMING_IPOS_SHEET_NAME = "Upcoming IPOs (60D)"
UPCOMING_EARNINGS_SHEET_NAME = "Upcoming Earnings (14D)"
TOP10_OHLC_SHEET_NAME = "Top 10 OHLC Tracking"
TOP10_OHLC_HIDDEN_COLUMNS = ("F", "G", "H", "M", "P")
TOP10_OHLC_TRAILING_HIDDEN_COLUMNS = ("R",)
PROTECTED_SHEET_NAMES = {
    "Single Tickers",
    UPCOMING_IPOS_SHEET_NAME,
    UPCOMING_EARNINGS_SHEET_NAME,
    TOP10_OHLC_SHEET_NAME,
}


def _format_mmddyyyy(d: date | None) -> str:
    if not d:
        return ""
    return d.strftime("%m/%d/%Y")


def _fetch_nasdaq_day(day: date, session: requests.Session) -> tuple[date, list[dict[str, Any]]]:
    """Fetch Nasdaq earnings calendar rows for a single day (no caching)."""
    try:
        resp = session.get(
            NASDAQ_EARNINGS_URL,
            params={"date": day.isoformat()},
            headers=NASDAQ_HEADERS,
            timeout=20,
        )
        if resp.status_code != 200:
            return day, []
        payload = resp.json()
    except Exception:
        return day, []

    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    return day, (rows if isinstance(rows, list) else [])


def _nasdaq_calendar_rows(
    day: date,
    session: requests.Session,
    cache: dict[date, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cached = cache.get(day)
    if cached is not None:
        return cached
    _, rows = _fetch_nasdaq_day(day, session)
    cache[day] = rows
    return rows


def _extract_company_name(row: dict[str, Any]) -> str:
    name = (
        row.get("companyName")
        or row.get("company")
        or row.get("company_name")
        or row.get("name")
        or ""
    )
    return str(name).strip()


def fetch_nasdaq_earnings_dates(
    symbols: list[str],
    session: requests.Session,
    today: date | None = None,
    max_days: int = 180,
) -> dict[str, dict[str, str]]:
    today = today or date.today()
    target = {s.strip().upper() for s in symbols if s and s.strip()}
    if not target:
        return {}

    # All weekdays in [today-max_days, today+max_days]
    all_days = [
        today + timedelta(days=i)
        for i in range(-max_days, max_days + 1)
        if (today + timedelta(days=i)).weekday() < 5
    ]

    # Fetch all days in parallel (I/O-bound → threads)
    with ThreadPoolExecutor(max_workers=20) as ex:
        day_data: dict[date, list[dict[str, Any]]] = dict(
            ex.map(lambda d: _fetch_nasdaq_day(d, session), all_days)
        )

    last_dates: dict[str, date] = {}
    next_dates: dict[str, date] = {}
    company_names: dict[str, str] = {}

    for day, rows in day_data.items():
        for row in rows:
            sym = str(row.get("symbol", "")).strip().upper()
            if sym not in target:
                continue
            if sym not in company_names:
                name = _extract_company_name(row)
                if name:
                    company_names[sym] = name
            # Earliest future date → next earnings
            if day >= today:
                if sym not in next_dates or day < next_dates[sym]:
                    next_dates[sym] = day
            # Most recent past date → last earnings
            else:
                if sym not in last_dates or day > last_dates[sym]:
                    last_dates[sym] = day

    out: dict[str, dict[str, str]] = {}
    for sym in target:
        out[sym] = {
            "last": _format_mmddyyyy(last_dates.get(sym)),
            "next": _format_mmddyyyy(next_dates.get(sym)),
            "name": company_names.get(sym, ""),
        }
    return out


def fetch_nasdaq_upcoming_earnings(
    session: requests.Session,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """
    Fetch all Nasdaq earnings calendar rows scheduled from start_date through end_date.
    """
    if end_date < start_date:
        return []

    all_days = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]

    # Fetch all days in parallel
    with ThreadPoolExecutor(max_workers=20) as ex:
        day_data: dict[date, list[dict[str, Any]]] = dict(
            ex.map(lambda d: _fetch_nasdaq_day(d, session), all_days)
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, date]] = set()
    for earnings_date in all_days:
        for row in day_data.get(earnings_date, []):
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            key = (symbol, earnings_date)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "symbol": symbol,
                    "name": _extract_company_name(row),
                    "earnings_date": earnings_date,
                    "time": str(row.get("time") or row.get("timeOfDay") or "").strip(),
                    "eps_forecast": str(row.get("epsForecast") or row.get("eps_forecast") or "").strip(),
                    "no_of_estimates": str(row.get("noOfEsts") or row.get("no_of_estimates") or "").strip(),
                    "last_year_eps": str(row.get("lastYearEPS") or row.get("last_year_eps") or "").strip(),
                }
            )

    rows.sort(key=lambda x: (x["earnings_date"], x["symbol"]))
    return rows


def fetch_alphavantage_upcoming_ipos(
    api_key: str,
    session: requests.Session,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming IPOs from Alpha Vantage IPO calendar and keep only entries
    scheduled from start_date through end_date (inclusive).
    """
    if not api_key.strip():
        return []

    try:
        resp = session.get(
            ALPHAVANTAGE_IPO_URL,
            params={"function": "IPO_CALENDAR", "apikey": api_key.strip()},
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        body = resp.text or ""
    except Exception:
        return []

    if not body.strip():
        return []

    rows: list[dict[str, Any]] = []
    try:
        reader = csv.DictReader(io.StringIO(body))
        for row in reader:
            ipo_text = str(row.get("ipoDate", "")).strip()
            if not ipo_text:
                continue
            try:
                ipo_dt = datetime.strptime(ipo_text, "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (start_date <= ipo_dt <= end_date):
                continue

            rows.append(
                {
                    "symbol": str(row.get("symbol", "")).strip(),
                    "name": str(row.get("name", "")).strip(),
                    "ipo_date": ipo_dt,
                    "price_range": str(row.get("priceRange", "")).strip(),
                    "exchange": str(row.get("exchange", "")).strip(),
                    "currency": str(row.get("currency", "")).strip(),
                    "shares_offered": str(row.get("sharesOffered", "")).strip(),
                    "estimated_volume": str(row.get("estimatedVolume", "")).strip(),
                }
            )
    except Exception:
        return []

    rows.sort(key=lambda x: (x["ipo_date"], x["symbol"]))
    return rows


def write_upcoming_ipos_sheet(
    wb: Workbook,
    ipo_rows: list[dict[str, Any]],
    start_date: date,
    end_date: date,
) -> None:
    """
    Create or replace a workbook tab with upcoming IPOs for the selected window.
    """
    sheet_name = UPCOMING_IPOS_SHEET_NAME
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        merged_ranges = list(ws.merged_cells.ranges)
        for rng in merged_ranges:
            ws.unmerge_cells(str(rng))
        if ws.max_row > 0:
            ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet(title=sheet_name)

    ws.append([f"Upcoming IPOs from {start_date.isoformat()} to {end_date.isoformat()}"])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
    ws["A1"].font = Font(name="Calibri", size=13, bold=True, color=Color(indexed=9))
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws["A1"].fill = PatternFill(fill_type="solid", fgColor=Color(indexed=8))

    headers = [
        "Symbol",
        "Company",
        "IPO Date",
        "Price Range",
        "Exchange",
        "Currency",
        "Shares Offered",
        "Estimated Volume",
    ]
    ws.append(headers)
    ws.row_dimensions[1].height = 50.85
    ws.row_dimensions[2].height = 34.85
    descriptor_fill = PatternFill(fill_type="solid", fgColor=Color(indexed=9))
    descriptor_font = Font(name="Calibri", size=11, bold=True, italic=True, color=Color(indexed=8))
    descriptor_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_red = Side(style="thin", color=Color(indexed=10))
    thick_black = Side(style="thick", color=Color(indexed=8))

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.fill = descriptor_fill
        cell.font = descriptor_font
        cell.alignment = descriptor_align
        cell.border = Border(
            left=thin_red if col_idx == 1 else thick_black,
            right=thick_black,
        )

    if ipo_rows:
        for row in ipo_rows:
            ws.append(
                [
                    row.get("symbol", ""),
                    row.get("name", ""),
                    row.get("ipo_date"),
                    row.get("price_range", ""),
                    row.get("exchange", ""),
                    row.get("currency", ""),
                    row.get("shares_offered", ""),
                    row.get("estimated_volume", ""),
                ]
            )
    else:
        ws.append(["No upcoming IPOs found in this date range.", "", "", "", "", "", "", ""])

    for row_idx in range(3, ws.max_row + 1):
        ws.cell(row=row_idx, column=3).number_format = "mmm d, yyyy"

    auto_size_columns(ws, min_width=10, max_width=45)


def write_upcoming_earnings_sheet(
    wb: Workbook,
    earnings_rows: list[dict[str, Any]],
    start_date: date,
    end_date: date,
    qualified_dates: dict[str, date],
) -> None:
    """
    Create or replace a workbook tab with upcoming earnings and current-run qualification.
    """
    sheet_name = UPCOMING_EARNINGS_SHEET_NAME
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        merged_ranges = list(ws.merged_cells.ranges)
        for rng in merged_ranges:
            ws.unmerge_cells(str(rng))
        if ws.max_row > 0:
            ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet(title=sheet_name)

    ws.append([f"Upcoming earnings from {start_date.isoformat()} to {end_date.isoformat()}"])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=9)
    ws["A1"].font = Font(name="Calibri", size=13, bold=True, color=Color(indexed=9))
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws["A1"].fill = PatternFill(fill_type="solid", fgColor=Color(indexed=8))

    headers = [
        "Symbol",
        "Company",
        "Earnings Date",
        "Time",
        "EPS Forecast",
        "No. of Estimates",
        "Last Year EPS",
        "Qualified Results?",
        "Qualified Date",
    ]
    ws.append(headers)
    ws.row_dimensions[1].height = 50.85
    ws.row_dimensions[2].height = 34.85
    descriptor_fill = PatternFill(fill_type="solid", fgColor=Color(indexed=9))
    descriptor_font = Font(name="Calibri", size=11, bold=True, italic=True, color=Color(indexed=8))
    descriptor_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_red = Side(style="thin", color=Color(indexed=10))
    thick_black = Side(style="thick", color=Color(indexed=8))

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.fill = descriptor_fill
        cell.font = descriptor_font
        cell.alignment = descriptor_align
        cell.border = Border(
            left=thin_red if col_idx == 1 else thick_black,
            right=thick_black,
        )

    if earnings_rows:
        for row in earnings_rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            qualified_date = qualified_dates.get(symbol)
            ws.append(
                [
                    symbol,
                    row.get("name", ""),
                    row.get("earnings_date"),
                    row.get("time", ""),
                    row.get("eps_forecast", ""),
                    row.get("no_of_estimates", ""),
                    row.get("last_year_eps", ""),
                    "Yes" if qualified_date else "No",
                    qualified_date,
                ]
            )
    else:
        ws.append(["No upcoming earnings found in this date range.", "", "", "", "", "", "", "", ""])

    for row_idx in range(3, ws.max_row + 1):
        ws.cell(row=row_idx, column=3).number_format = "mmm d, yyyy"
        ws.cell(row=row_idx, column=9).number_format = "mmm d, yyyy"

    auto_size_columns(ws, min_width=10, max_width=45)


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


def max_close_and_days_ago(
    dates: np.ndarray,
    closes: np.ndarray,
    last_date: date,
    cutoff_int: int | None = None,
) -> tuple[float | None, int | None]:
    """
    Max close (most recent if tied) and calendar days since that date.
    Optionally restrict to dates >= cutoff_int (YYYYMMDD).
    """
    if len(dates) == 0 or len(closes) == 0:
        return None, None
    if cutoff_int is not None:
        mask = dates >= cutoff_int
        if not np.any(mask):
            return None, None
        dates = dates[mask]
        closes = closes[mask]
    max_close = float(np.max(closes))
    idxs = np.where(closes == max_close)[0]
    if len(idxs) == 0:
        return None, None
    max_date_int = int(dates[idxs[-1]])
    days_ago = (last_date - date_from_int(max_date_int)).days
    return max_close, int(days_ago)


def last_close_5pct_higher_info(
    dates: np.ndarray,
    closes: np.ndarray,
    last_close: float,
    last_date: date,
) -> tuple[str | None, int | None]:
    """
    Last date when close >= 5% above the most recent close.
    Returns (MM/DD/YYYY, days_ago). Excludes the most recent close.
    """
    if len(closes) < 2 or not np.isfinite(last_close):
        return "", None
    threshold = last_close * 1.05
    mask = closes[:-1] >= threshold
    if not np.any(mask):
        return "", None
    idx = int(np.where(mask)[0][-1])
    dt = date_from_int(int(dates[idx]))
    return _format_mmddyyyy(dt), int((last_date - dt).days)


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


def append_single_ticker_section(
    ws,
    heading: str,
    header_row: list[Any],
    descriptors: list[Any],
    data_row: list[Any],
) -> None:
    if not is_empty_sheet(ws):
        ws.append([])

    ws.append([heading])
    heading_row = ws.max_row
    ws.merge_cells(start_row=heading_row, start_column=1, end_row=heading_row, end_column=len(header_row))
    heading_cell = ws.cell(row=heading_row, column=1)
    heading_cell.font = Font(name="Calibri", size=12, bold=True)
    heading_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws.append([h if h is not None else "" for h in header_row])
    ws.append([d if d is not None else "" for d in descriptors])
    ws.append(data_row)

    header_row_idx = heading_row + 1
    descriptor_row_idx = heading_row + 2
    for col_idx in range(1, len(header_row) + 1):
        cell = ws.cell(row=header_row_idx, column=col_idx)
        cell.font = Font(name="Calibri", size=11, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col_idx in range(1, len(descriptors) + 1):
        cell = ws.cell(row=descriptor_row_idx, column=col_idx)
        cell.font = Font(name="Calibri", size=10, italic=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def count_recent_symbol_occurrences(wb: Workbook, max_runs: int = 5) -> dict[str, int]:
    """
    Count how many of the last N sheets each symbol appeared in.
    """
    if max_runs <= 0 or not wb.sheetnames:
        return {}

    run_sheet_names = [name for name in wb.sheetnames if name not in PROTECTED_SHEET_NAMES]
    recent_names = run_sheet_names[-max_runs:]
    counts: dict[str, int] = {}
    for name in recent_names:
        ws = wb[name]
        if is_empty_sheet(ws):
            continue
        symbols_in_sheet: set[str] = set()
        for (val,) in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=1, max_col=1, values_only=True):
            if val is None:
                continue
            sym = str(val).strip().upper()
            if sym:
                symbols_in_sheet.add(sym)
        for sym in symbols_in_sheet:
            counts[sym] = counts.get(sym, 0) + 1

    return counts


def _parse_run_sheet_date(sheet_name: str) -> date | None:
    base_name = sheet_name.split(" (", 1)[0].strip()
    try:
        return datetime.strptime(base_name, "%d %b %Y").date()
    except Exception:
        return None


def collect_qualified_result_dates(
    wb: Workbook,
    current_results: list[dict[str, Any]] | None = None,
    current_date: date | None = None,
) -> dict[str, date]:
    """
    Map each symbol that appeared in a regular results sheet to its latest qualifying date.
    """
    qualified_dates: dict[str, date] = {}

    for name in wb.sheetnames:
        if name in PROTECTED_SHEET_NAMES:
            continue
        sheet_date = _parse_run_sheet_date(name)
        if sheet_date is None:
            continue
        ws = wb[name]
        if is_empty_sheet(ws):
            continue
        for (val,) in ws.iter_rows(min_row=3, max_row=ws.max_row, min_col=1, max_col=1, values_only=True):
            if val is None:
                continue
            sym = str(val).strip().upper()
            if not sym:
                continue
            existing = qualified_dates.get(sym)
            if existing is None or sheet_date > existing:
                qualified_dates[sym] = sheet_date

    if current_results and current_date is not None:
        for row in current_results:
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                continue
            existing = qualified_dates.get(sym)
            if existing is None or current_date > existing:
                qualified_dates[sym] = current_date

    return qualified_dates


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            continue
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def collect_top_ranked_cohorts(wb: Workbook, top_n: int = 10) -> list[dict[str, Any]]:
    """
    Read workbook run sheets and return ranked ticker cohorts for OHLC tracking.
    """
    cohorts: list[dict[str, Any]] = []
    run_sheets_by_date: dict[date, str] = {}
    for name in wb.sheetnames:
        if name in PROTECTED_SHEET_NAMES:
            continue
        rank_date = _parse_run_sheet_date(name)
        if rank_date is None:
            continue
        # If a workbook has duplicate run sheets for the same date, use the last
        # one in workbook order and avoid double-counting its top ranked tickers.
        run_sheets_by_date[rank_date] = name

    seen_cohorts: set[tuple[date, int, str]] = set()
    for rank_date, name in run_sheets_by_date.items():
        ws = wb[name]
        if is_empty_sheet(ws) or ws.max_row < 3:
            continue

        headers = [
            str(cell.value).strip().lower() if cell.value is not None else ""
            for cell in ws[1]
        ]
        try:
            rank_idx = headers.index("rank")
        except ValueError:
            continue
        close_idx = next(
            (idx for idx, header in enumerate(headers) if "close" in header and "$" in header),
            None,
        )
        if close_idx is None:
            continue

        for row in ws.iter_rows(min_row=3, values_only=True):
            rank = _coerce_int(row[rank_idx] if rank_idx < len(row) else None)
            if rank is None or rank < 1 or rank > top_n:
                continue
            symbol = str(row[0] if row and row[0] is not None else "").strip().upper()
            if not symbol:
                continue
            cohort_key = (rank_date, rank, symbol)
            if cohort_key in seen_cohorts:
                continue
            seen_cohorts.add(cohort_key)
            cohorts.append(
                {
                    "rank_date": rank_date,
                    "rank": rank,
                    "symbol": symbol,
                    "rank_close": _coerce_float(row[close_idx] if close_idx < len(row) else None),
                }
            )

    cohorts.sort(key=lambda r: (r["rank_date"], int(r["rank"]), str(r["symbol"])))
    return cohorts


def build_top10_ohlc_tracking_rows(
    cohorts: list[dict[str, Any]],
    symbol_paths: dict[str, Path],
    root: Path,
    follow_days: int = 5,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    ohlc_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def pct_vs_rank(value: float, rank_close: float | None) -> float | None:
        if rank_close is None or rank_close == 0:
            return None
        return float(f"{pct_change(value, rank_close):.2f}")

    for cohort in cohorts:
        symbol = str(cohort["symbol"]).strip().upper()
        normalized = normalize_symbol(symbol)
        path = symbol_paths.get(normalized)
        if path is None:
            path = find_symbol_file(root, normalized)
        if path is None:
            continue

        if normalized not in ohlc_cache:
            ohlc_cache[normalized] = load_ohlc_from_file(path)
        dates, opens, highs, lows, closes = ohlc_cache[normalized]
        if len(dates) == 0:
            continue

        rank_date = cohort["rank_date"]
        rank_int = date_to_int(rank_date)
        start_idx = int(np.searchsorted(dates, rank_int, side="right"))
        rank_close = _coerce_float(cohort.get("rank_close"))
        for offset, idx in enumerate(range(start_idx, min(start_idx + follow_days, len(dates))), start=1):
            open_val = float(opens[idx])
            high_val = float(highs[idx])
            low_val = float(lows[idx])
            close_val = float(closes[idx])
            hit_target = ""
            hit_stop = ""
            if rank_close is not None:
                hit_target = "Yes" if high_val >= rank_close * 1.02 else "No"
                hit_stop = "Yes" if low_val <= rank_close * 0.99 else "No"
            rows.append(
                [
                    rank_date,
                    int(cohort["rank"]),
                    symbol,
                    offset,
                    date_from_int(int(dates[idx])),
                    open_val,
                    high_val,
                    low_val,
                    close_val,
                    rank_close,
                    high_val,
                    pct_vs_rank(high_val, rank_close),
                    pct_vs_rank(low_val, rank_close),
                    pct_vs_rank(close_val, rank_close),
                    hit_target,
                    hit_stop,
                ]
            )

    return rows


def write_top10_ohlc_tracking_sheet(
    wb: Workbook,
    symbol_paths: dict[str, Path],
    root: Path,
    top_n: int = 10,
) -> None:
    cohorts = collect_top_ranked_cohorts(wb, top_n=top_n)
    refresh_dates = {cohort["rank_date"] for cohort in cohorts}
    headers = [
        "Rank Date",
        "Rank",
        "Symbol",
        "Days Since Rank",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Rank Close",
        "Tracking Day High",
        "High vs Initial Rank Close %",
        "Low vs Rank Close %",
        "Close vs Initial Rank Close %",
        "Hit +2%?",
        "Hit -1%?",
        "VV",
    ]

    preserved_rows: list[list[Any]] = []
    preserved_row_keys: set[tuple[Any, ...]] = set()
    if TOP10_OHLC_SHEET_NAME in wb.sheetnames:
        ws = wb[TOP10_OHLC_SHEET_NAME]
        existing_headers = [str(cell.value or "").strip() for cell in ws[1]]
        existing_header_map = {header: idx for idx, header in enumerate(existing_headers) if header}
        for row in ws.iter_rows(min_row=2, values_only=True):
            rank_date = _coerce_date(row[0] if row else None)
            if rank_date is not None and rank_date in refresh_dates:
                continue
            if any(value is not None for value in row):
                preserved_row = [
                    row[existing_header_map[header]]
                    if header in existing_header_map and existing_header_map[header] < len(row)
                    else None
                    for header in headers
                ]
                row_key = tuple(preserved_row[:5])
                if row_key in preserved_row_keys:
                    continue
                preserved_row_keys.add(row_key)
                preserved_rows.append(preserved_row)
        ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet(title=TOP10_OHLC_SHEET_NAME)

    ws.append(headers)
    for row in preserved_rows:
        ws.append(row + [""] * (len(headers) - len(row)))
    for row in build_top10_ohlc_tracking_rows(cohorts, symbol_paths, root):
        ws.append(row)

    tracking_day_high_col = headers.index("Tracking Day High") + 1
    high_col = headers.index("High") + 1
    for row_idx in range(2, ws.max_row + 1):
        tracking_day_high_cell = ws.cell(row=row_idx, column=tracking_day_high_col)
        if tracking_day_high_cell.value in (None, ""):
            tracking_day_high_cell.value = ws.cell(row=row_idx, column=high_col).value

    header_fill = PatternFill(fill_type="solid", fgColor=Color(indexed=8))
    header_font = Font(name="Calibri", size=11, bold=True, color=Color(indexed=9))
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    data_align = Alignment(horizontal="center", vertical="center")
    hit_target_fill = PatternFill(fill_type="solid", fgColor="C6EFCE")
    hit_target_vertical_border = Side(style="thin", color="A6A6A6")
    date_group_border = Side(style="thick", color="808080")
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_align

    success_label_cell = ws["S1"]
    success_label_cell.value = "Next-Day +2% Success Rate"
    success_label_cell.fill = header_fill
    success_label_cell.font = header_font
    success_label_cell.alignment = header_align

    success_rate_cell = ws["T1"]
    success_rate_cell.value = '=IFERROR(COUNTIFS($D:$D,1,$O:$O,"Yes")/COUNTIFS($D:$D,1,$O:$O,"<>"),"")'
    success_rate_cell.fill = header_fill
    success_rate_cell.font = header_font
    success_rate_cell.alignment = header_align
    success_rate_cell.number_format = "0.00%"

    data_last_row = max(ws.max_row, 2)
    overall_success_label_cell = ws["U1"]
    overall_success_label_cell.value = "Overall +2% Success Rate"
    overall_success_label_cell.fill = header_fill
    overall_success_label_cell.font = header_font
    overall_success_label_cell.alignment = header_align

    overall_success_rate_cell = ws["V1"]
    overall_success_rate_cell.value = (
        f'=IFERROR(LET(keys,$A$2:$A${data_last_row}&"|"&$B$2:$B${data_last_row}&"|"&$C$2:$C${data_last_row},'
        f'valid,$O$2:$O${data_last_row}<>"",'
        f'uniqueKeys,UNIQUE(FILTER(keys,valid)),'
        f'hits,IFERROR(UNIQUE(FILTER(keys,$O$2:$O${data_last_row}="Yes")),""),'
        f'SUM(--ISNUMBER(XMATCH(uniqueKeys,hits)))/ROWS(uniqueKeys)),"")'
    )
    overall_success_rate_cell.fill = header_fill
    overall_success_rate_cell.font = header_font
    overall_success_rate_cell.alignment = header_align
    overall_success_rate_cell.number_format = "0.00%"

    pct_literal_format = '0.00"%"'
    date_cols = {1, 5}
    price_cols = {6, 7, 8, 9, 10, 11}
    pct_cols = {12, 13, 14}
    previous_rank_date = None
    for row_idx in range(2, ws.max_row + 1):
        for col_idx in range(1, len(headers) + 1):
            ws.cell(row=row_idx, column=col_idx).alignment = data_align
        hit_target = ws.cell(row=row_idx, column=15).value
        if isinstance(hit_target, str) and hit_target.strip().lower() == "yes":
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.fill = hit_target_fill
                cell.border = Border(
                    left=hit_target_vertical_border,
                    right=hit_target_vertical_border,
                    top=cell.border.top,
                    bottom=cell.border.bottom,
                    diagonal=cell.border.diagonal,
                    diagonal_direction=cell.border.diagonal_direction,
                    diagonalUp=cell.border.diagonalUp,
                    diagonalDown=cell.border.diagonalDown,
                    outline=cell.border.outline,
                    vertical=cell.border.vertical,
                    horizontal=cell.border.horizontal,
                )
        for col_idx in date_cols:
            ws.cell(row=row_idx, column=col_idx).number_format = "mmm d, yyyy"
        for col_idx in price_cols:
            ws.cell(row=row_idx, column=col_idx).number_format = '"$"#,##0.00'
        for col_idx in pct_cols:
            ws.cell(row=row_idx, column=col_idx).number_format = pct_literal_format
        rank_date = ws.cell(row=row_idx, column=1).value
        if previous_rank_date is not None and rank_date != previous_rank_date:
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = Border(
                    left=cell.border.left,
                    right=cell.border.right,
                    top=date_group_border,
                    bottom=cell.border.bottom,
                    diagonal=cell.border.diagonal,
                    diagonal_direction=cell.border.diagonal_direction,
                    diagonalUp=cell.border.diagonalUp,
                    diagonalDown=cell.border.diagonalDown,
                    outline=cell.border.outline,
                    vertical=cell.border.vertical,
                    horizontal=cell.border.horizontal,
                )
        previous_rank_date = rank_date

    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.alignment = header_align if cell.row == 1 else data_align

    ws.freeze_panes = "A2"
    auto_size_columns(ws, min_width=9, max_width=22)
    for col_idx in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col_idx)].hidden = False
    # Set to an empty tuple to show all OHLC tracking columns again.
    for col_letter in TOP10_OHLC_HIDDEN_COLUMNS:
        ws.column_dimensions[col_letter].hidden = True
    # Excel always has additional blank columns; keep the first trailing one out of view.
    for col_letter in TOP10_OHLC_TRAILING_HIDDEN_COLUMNS:
        ws.column_dimensions[col_letter].hidden = True


def prune_old_run_sheets(
    wb: Workbook,
    keep_runs: int = 7,
    protected_names: set[str] | None = None,
) -> None:
    """
    Keep only the most recent run-result sheets in workbook order.
    """
    if keep_runs < 0:
        keep_runs = 0
    keep_protected = protected_names or PROTECTED_SHEET_NAMES
    run_sheet_names = [name for name in wb.sheetnames if name not in keep_protected]
    remove_count = len(run_sheet_names) - keep_runs
    if remove_count <= 0:
        return
    for name in run_sheet_names[:remove_count]:
        del wb[name]


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
    apply_filters: bool = True,
) -> dict[str, Any] | None:
    d, c, v, h, l = load_series_from_file(path, need_rows=need_rows)
    if len(d) == 0:
        return None
    min_rows = 60
    if params["avg_vol_mode"] == "days":
        min_rows = max(params["avg_vol_days"] + 1, min_rows)
    min_rows = max(min_rows, params["rsi_period"] + 2, params["atr_period"] + 2)
    if apply_filters and len(d) < min_rows:
        return None

    change_lookback = 5  # trading periods
    prev_idx = -(change_lookback + 1)

    # Average daily $ volume filter (close * volume) over avg volume window
    if params["avg_vol_mode"] == "months":
        last_date = date_from_int(int(d[-1]))
        cutoff_date = shift_months(last_date, -int(params["avg_vol_months"]))
        cutoff_int = date_to_int(cutoff_date)
        if apply_filters and d[0] > cutoff_int:
            return None
        mask = d >= cutoff_int
        if apply_filters and not np.any(mask):
            return None
        if np.any(mask):
            close_window = c[mask]
            vol_window = v[mask]
        else:
            close_window = c
            vol_window = v
    else:
        if params["avg_vol_days"] > 0:
            close_window = c[-params["avg_vol_days"]:]
            vol_window = v[-params["avg_vol_days"]:]
        else:
            close_window = c
            vol_window = v

    avg_dollar_volume = float(np.mean(close_window * vol_window)) if len(close_window) else np.nan
    if apply_filters and (not np.isfinite(avg_dollar_volume) or avg_dollar_volume <= params["avg_dollar_vol_min"]):
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
    last_date = date_from_int(int(d[-1]))
    prev_close = float(c[prev_idx]) if len(c) >= change_lookback + 1 else np.nan

    cutoff_52_date = last_date - timedelta(days=364)
    cutoff_52_int = date_to_int(cutoff_52_date)
    high_52_close, high_52_days = max_close_and_days_ago(d, c, last_date, cutoff_52_int)
    high_all_close, high_all_days = max_close_and_days_ago(d, c, last_date, None)
    last_5pct_date, last_5pct_days = last_close_5pct_higher_info(d, c, last_close, last_date)

    # RSI filter
    rsi_vals = rsi_wilder(c, period=params["rsi_period"])
    last_rsi = float(rsi_vals[-1])
    prev_rsi = float(rsi_vals[prev_idx]) if len(rsi_vals) >= change_lookback + 1 else np.nan
    if apply_filters and not (params["rsi_low"] <= last_rsi <= params["rsi_high"]):
        return None

    # ATR filter (percent of last close)
    atr_vals = atr_wilder(h, l, c, period=params["atr_period"])
    last_atr = float(atr_vals[-1])
    atr_pct = np.nan
    if np.isfinite(last_atr) and last_close > 0:
        atr_pct = (last_atr / last_close) * 100.0
    if apply_filters and (not np.isfinite(atr_pct) or atr_pct <= params["atr_min_pct"]):
        return None

    # MACD filter
    macd_line, sig_line = macd(c, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    last_macd = float(macd_line[-1])
    last_sig = float(sig_line[-1])
    prev_macd = float(macd_line[prev_idx]) if len(macd_line) >= change_lookback + 1 else np.nan
    prev_sig = float(sig_line[prev_idx]) if len(sig_line) >= change_lookback + 1 else np.nan
    if apply_filters and not (last_macd > last_sig):
        return None

    macd_signal_ratio = np.nan
    if np.isfinite(last_macd) and np.isfinite(last_sig) and last_sig != 0:
        macd_signal_ratio = last_macd / last_sig
    macd_signal_ratio_prev = np.nan
    if np.isfinite(prev_macd) and np.isfinite(prev_sig) and prev_sig != 0:
        macd_signal_ratio_prev = prev_macd / prev_sig

    stock_close_aligned = np.array([], dtype=float)
    bench_close_aligned = np.array([], dtype=float)
    beta_prev = np.nan

    if params["beta_freq"] == "monthly":
        sm, smc = monthly_closes(d, c)
        if len(sm) < params["beta_months"] + 1:
            if apply_filters:
                return None
        else:
            stock_aligned: list[float] = []
            bench_aligned: list[float] = []
            for mk, ci in zip(sm, smc):
                bench_ci = bench_map.get(int(mk))
                if bench_ci is None:
                    continue
                stock_aligned.append(float(ci))
                bench_aligned.append(float(bench_ci))

            if len(stock_aligned) < params["beta_months"] + 1:
                if apply_filters:
                    return None
            else:
                stock_close_aligned = np.array(stock_aligned[-(params["beta_months"] + 1):], dtype=float)
                bench_close_aligned = np.array(bench_aligned[-(params["beta_months"] + 1):], dtype=float)
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
            if apply_filters:
                return None
        else:
            stock_close_aligned = np.array(stock_aligned[-(params["beta_lookback"] + 1):], dtype=float)
            bench_close_aligned = np.array(bench_aligned[-(params["beta_lookback"] + 1):], dtype=float)
            if len(stock_aligned) >= params["beta_lookback"] + 1 + change_lookback:
                prev_stock = np.array(stock_aligned[:-change_lookback], dtype=float)
                prev_bench = np.array(bench_aligned[:-change_lookback], dtype=float)
                if len(prev_stock) >= params["beta_lookback"] + 1:
                    beta_prev = beta_from_aligned_closes(
                        prev_stock[-(params["beta_lookback"] + 1):],
                        prev_bench[-(params["beta_lookback"] + 1):],
                    )

    b = beta_from_aligned_closes(stock_close_aligned, bench_close_aligned)
    if apply_filters and (not np.isfinite(b) or b <= params["beta_min"]):
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
        "high_52w_close": high_52_close,
        "high_52w_days_ago": high_52_days,
        "high_all_close": high_all_close,
        "high_all_days_ago": high_all_days,
        "last_5pct_higher_date": last_5pct_date,
        "last_5pct_higher_days_ago": last_5pct_days,
        "beta": b,
        "beta_pct_5": beta_pct_5,
        "rsi": last_rsi,
        "rsi_pct_5": rsi_pct_5,
        "atr": last_atr,
        "atr_pct": atr_pct,
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
    tickers_group = ap.add_mutually_exclusive_group(required=False)
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
    ap.add_argument(
        "--alphavantage_api_key",
        default=ALPHAVANTAGE_API_KEY,
        help="Alpha Vantage API key (defaults to hardcoded project key).",
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

    ap.add_argument("--atr_period", type=int, default=14, help="ATR period in days (default: 14)")
    ap.add_argument(
        "--atr_min_pct",
        type=float,
        default=2.0,
        help="Minimum ATR as percent of last close (default: 2.0)",
    )

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
    ap.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top-ranked rows to flag (default: 10)",
    )
    ap.add_argument(
        "--score_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Top-N scoring and ranking columns (default: True)",
    )

    args = ap.parse_args()

    if args.avg_vol_days is not None and args.avg_vol_days <= 0:
        raise SystemExit("--avg_vol_days must be > 0")
    if args.avg_vol_months <= 0:
        raise SystemExit("--avg_vol_months must be > 0")
    if args.atr_period <= 0:
        raise SystemExit("--atr_period must be > 0")
    if args.atr_min_pct < 0:
        raise SystemExit("--atr_min_pct must be >= 0")

    run_mode, single_symbol = prompt_run_mode()

    root = resolve_path(args.root)
    out_path = resolve_path(args.out)

    tickers: list[str] = []
    symbol_paths: dict[str, Path] = {}
    single_path: Path | None = None

    if run_mode == "all":
        if not args.tickers and not args.tickers_dir:
            raise SystemExit("Provide --tickers or --tickers_dir for All tickers mode.")
        if args.tickers:
            tickers_path = resolve_path(args.tickers)
            tickers = load_tickers_csv(tickers_path)
            symbol_paths = build_file_map(root)
        else:
            tickers_root = resolve_path(args.tickers_dir)
            symbol_paths = build_file_map(tickers_root)
            tickers = sorted(symbol_paths.keys())
    else:
        if not single_symbol:
            raise SystemExit("Single ticker selection required.")

    bench_sym = normalize_symbol(args.benchmark)

    bench_path = None
    if run_mode == "all":
        bench_path = symbol_paths.get(bench_sym)
        if bench_path is None:
            bench_path = find_symbol_file(root, bench_sym)
    else:
        bench_path = find_symbol_file(root, bench_sym)
        if bench_path is None and args.tickers_dir:
            bench_path = find_symbol_file(resolve_path(args.tickers_dir), bench_sym)

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
        args.atr_period + 30,
        260,
    )

    # Load benchmark series (tail), build date->close map or month->close map
    bd, bc, _bv, _bh, _bl = load_series_from_file(bench_path, need_rows=need_rows)
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
    if run_mode == "all":
        for sym in tickers:
            p = symbol_paths.get(sym)
            if not p:
                continue
            tasks.append((sym, p))
    else:
        single_path = find_symbol_file(root, single_symbol)
        if single_path is None and args.tickers_dir:
            single_path = find_symbol_file(resolve_path(args.tickers_dir), single_symbol)
        if single_path is None:
            raise SystemExit(f"Ticker {single_symbol} not found under {root}.")
        tasks.append((single_symbol, single_path))

    params = {
        "beta_freq": args.beta_freq,
        "beta_lookback": args.beta_lookback,
        "beta_months": args.beta_months,
        "beta_min": args.beta_min,
        "rsi_low": args.rsi_low,
        "rsi_high": args.rsi_high,
        "rsi_period": args.rsi_period,
        "atr_period": args.atr_period,
        "atr_min_pct": args.atr_min_pct,
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
    if run_mode == "all":
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
    else:
        sym, p = tasks[0]
        res = screen_symbol(sym, p, params, bench_map, need_rows, apply_filters=False)
        if res:
            results.append(res)
        else:
            raise SystemExit(f"No data available for {display_symbol(sym)}.")

    if results:
        symbols = [str(row.get("symbol", "")).strip().upper() for row in results]
        session = requests.Session()
        earnings_map = fetch_nasdaq_earnings_dates(symbols, session, today=date.today())
        for row in results:
            symbol = str(row.get("symbol", "")).strip().upper()
            info = earnings_map.get(symbol, {})
            row["last_earnings_date"] = info.get("last", "")
            row["next_earnings_date"] = info.get("next", "")
            row["company_name"] = info.get("name", "")

    def parse_earnings_date(value: Any) -> date | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%m/%d/%Y").date()
        except Exception:
            return None

    def to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if np.isfinite(out) else None

    def percentile_rank(values: list[float], value: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return 1.0
        left = bisect_left(sorted_vals, value)
        right = bisect_right(sorted_vals, value)
        idx = (left + right - 1) / 2.0
        return idx / (n - 1)

    def score_row(row: dict[str, Any], liquidity_values: list[float]) -> float | None:
        avg_dollar_volume = to_float(row.get("avg_dollar_volume"))
        atr_pct = to_float(row.get("atr_pct"))
        rsi = to_float(row.get("rsi"))
        if avg_dollar_volume is None or atr_pct is None or rsi is None:
            return None
        liquidity_score = percentile_rank(liquidity_values, avg_dollar_volume)
        if 3.0 <= atr_pct <= 5.5:
            atr_score = 1.0
        else:
            distance = (3.0 - atr_pct) if atr_pct < 3.0 else (atr_pct - 5.5)
            atr_score = max(0.0, 1.0 - distance / 3.0)
        rsi_score = max(0.0, min(1.0, 1.0 - abs(rsi - 55.0) / 15.0))
        macd_ratio_pct = to_float(row.get("macd_signal_ratio_pct_5"))
        momentum_score = 1.0 if macd_ratio_pct is not None and macd_ratio_pct > 0 else 0.0
        prev_runs = to_float(row.get("prev_5_runs")) or 0.0
        consistency_score = min(prev_runs / 5.0, 1.0)
        return (
            0.30 * liquidity_score
            + 0.25 * atr_score
            + 0.20 * rsi_score
            + 0.15 * momentum_score
            + 0.10 * consistency_score
        )

    def is_score_eligible(row: dict[str, Any], today: date) -> bool:
        next_earnings = parse_earnings_date(row.get("next_earnings_date"))
        if next_earnings is not None:
            days_out = (next_earnings - today).days
            if 0 <= days_out <= 10:
                return False
        atr_pct = to_float(row.get("atr_pct"))
        if atr_pct is None or atr_pct < 2.5:
            return False
        avg_dollar_volume = to_float(row.get("avg_dollar_volume"))
        if avg_dollar_volume is None or avg_dollar_volume < 10_000_000:
            return False
        rsi = to_float(row.get("rsi"))
        if rsi is None or rsi >= 68:
            return False
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")

    header_row = [
        None,
        "Company",
        "Prev 5x",
        " Close $",
        None,
        "52W High",
        None,
        "All-Time High",
        None,
        "Last 5% Higher Close",
        None,
        "Beta",
        None,
        "RSI",
        None,
        "ATR %",
        "MACD",
        None,
        "Signal",
        None,
        "MACD/Signal",
        None,
        "Avg $ Vol",
        None,
        "Earnings",
        None,
    ]
    data_keys = [
        "symbol",
        "company_name",
        "prev_5_runs",
        "last_close",
        "last_close_pct_5",
        "high_52w_close",
        "high_52w_days_ago",
        "high_all_close",
        "high_all_days_ago",
        "last_5pct_higher_date",
        "last_5pct_higher_days_ago",
        "beta",
        "beta_pct_5",
        "rsi",
        "rsi_pct_5",
        "atr_pct",
        "macd",
        "macd_pct_5",
        "signal",
        "signal_pct_5",
        "macd_signal_ratio",
        "macd_signal_ratio_pct_5",
        "avg_dollar_volume",
        "avg_dollar_volume_pct_5",
        "last_earnings_date",
        "next_earnings_date",
    ]
    if args.score_enable:
        header_row = header_row[:2] + ["Rank"] + header_row[2:]
        data_keys = data_keys[:2] + ["rank"] + data_keys[2:]
    data_date = date_from_int(int(bd[-1])) if len(bd) else date.today()
    headline = data_date.strftime("%d %b %Y").upper()

    if out_path.exists():
        wb = load_workbook(out_path)
        prev_counts = count_recent_symbol_occurrences(wb, max_runs=5) if run_mode == "all" else {}
    else:
        wb = Workbook()
        prev_counts = {}

    top10: list[dict[str, Any]] = []
    if results:
        for row in results:
            symbol_key = str(row.get("symbol", "")).strip().upper()
            row["prev_5_runs"] = int(prev_counts.get(symbol_key, 0))
            if args.score_enable:
                row["total_score"] = None
                row["rank"] = None
    if results and args.score_enable:
        score_today = date.today()
        eligible = [row for row in results if is_score_eligible(row, score_today)]
        liquidity_values = [to_float(row.get("avg_dollar_volume")) for row in eligible]
        liquidity_values = [val for val in liquidity_values if val is not None]
        for row in eligible:
            row["total_score"] = score_row(row, liquidity_values)
        scored = sorted(
            [row for row in eligible if row.get("total_score") is not None],
            key=lambda r: float(r.get("total_score", 0.0)),
            reverse=True,
        )
        top_n = max(args.top_n, 0)
        top10 = scored[:top_n] if top_n else []
        for idx, row in enumerate(top10, start=1):
            row["rank"] = idx

    qualified_dates = collect_qualified_result_dates(
        wb,
        current_results=results if run_mode == "all" else None,
        current_date=data_date if run_mode == "all" else None,
    )

    if run_mode == "single":
        if "Single Tickers" in wb.sheetnames:
            ws = wb["Single Tickers"]
        elif len(wb.sheetnames) == 1 and is_empty_sheet(wb.active):
            ws = wb.active
            ws.title = "Single Tickers"
        else:
            ws = wb.create_sheet(title="Single Tickers")
    else:
        sheet_name = unique_sheet_name(wb, headline)
        if len(wb.sheetnames) == 1 and is_empty_sheet(wb.active):
            ws = wb.active
            ws.title = sheet_name
        else:
            ws = wb.create_sheet(title=sheet_name)

        ws.append(header_row)

    if args.beta_freq == "monthly":
        beta_desc = f"{args.beta_months} months\n> {args.beta_min}"
    else:
        beta_desc = f"{args.beta_lookback} days\n> {args.beta_min}"

    if avg_vol_mode == "months":
        avg_desc = f"{avg_vol_months} months\n> ${args.avg_dollar_vol_min:,.0f}"
    else:
        avg_desc = f"{avg_vol_days} days\n> ${args.avg_dollar_vol_min:,.0f}"

    pct_change_days = 5
    pct_change_desc = f"{pct_change_days} days \nchange %"
    pct_desc = f"{pct_change_days} days %"
    beta_change_desc = (
        f"{pct_change_days} months \nchange %" if args.beta_freq == "monthly" else pct_change_desc
    )
    descriptors = [
        None,
        None,
        "Last 5 runs",
        None,
        pct_change_desc,
        None,
        "Days ago",
        None,
        "Days ago",
        None,
        "Days ago",
        beta_desc,
        beta_change_desc,
        f"{args.rsi_period} days\n{args.rsi_low} to {args.rsi_high}",
        pct_desc,
        f"{args.atr_period} days\n> {args.atr_min_pct}% of close",
        f"{args.macd_fast}/{args.macd_slow} EMA\nMACD > Signal",
        pct_desc,
        f"{args.macd_signal} days",
        pct_desc,
        "MACD / Signal",
        pct_desc,
        avg_desc,
        pct_desc,
        "Last",
        "Next",
    ]
    if args.score_enable:
        descriptors = descriptors[:2] + [None] + descriptors[2:]
    if run_mode == "all":
        ws.append(descriptors)

    def parse_mmddyyyy(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        try:
            return datetime.strptime(str(value), "%m/%d/%Y")
        except Exception:
            return None

    def fmt_pct_value(x: Any) -> float | None:
        if x is None:
            return None
        try:
            if isinstance(x, (float, np.floating)) and not np.isfinite(x):
                return None
            return float(f"{float(x):.2f}")
        except Exception:
            return None

    def build_output_row(row: dict[str, Any], prev_count: int | None) -> list[Any]:
        high_52_close = row.get("high_52w_close")
        high_all_close = row.get("high_all_close")
        symbol_display = str(row.get("symbol", "")).strip()
        company_name = str(row.get("company_name", "")).strip()
        if company_name:
            company_name = company_name[:10]
        last_close_val = row.get("last_close")
        last_close = float(last_close_val) if last_close_val is not None and np.isfinite(last_close_val) else None
        avg_dollar_volume = row.get("avg_dollar_volume")
        avg_dollar_volume_val = (
            float(avg_dollar_volume) if avg_dollar_volume is not None and np.isfinite(avg_dollar_volume) else None
        )
        rank_val = row.get("rank") if args.score_enable else None
        output_row = [
            symbol_display,
            company_name,
        ]
        if args.score_enable:
            output_row.append(rank_val)
        output_row.extend(
            [
                prev_count,
                last_close,
                fmt_pct_value(row.get("last_close_pct_5")),
                float(high_52_close) if high_52_close is not None else None,
                row.get("high_52w_days_ago"),
                float(high_all_close) if high_all_close is not None else None,
                row.get("high_all_days_ago"),
                parse_mmddyyyy(row.get("last_5pct_higher_date")),
                row.get("last_5pct_higher_days_ago"),
                to_float(row.get("beta")),
                fmt_pct_value(row.get("beta_pct_5")),
                to_float(row.get("rsi")),
                fmt_pct_value(row.get("rsi_pct_5")),
                fmt_pct_value(row.get("atr_pct")),
                to_float(row.get("macd")),
                fmt_pct_value(row.get("macd_pct_5")),
                to_float(row.get("signal")),
                fmt_pct_value(row.get("signal_pct_5")),
                fmt2(row.get("macd_signal_ratio")),
                to_float(row.get("macd_signal_ratio")),
                avg_dollar_volume_val,
                fmt_pct_value(row.get("avg_dollar_volume_pct_5")),
                parse_mmddyyyy(row.get("last_earnings_date")),
                parse_mmddyyyy(row.get("next_earnings_date")),
            ]
        )
        return output_row

    if run_mode == "single":
        data_row = build_output_row(results[0], None)
        append_single_ticker_section(ws, headline, header_row, descriptors, data_row)
        ipo_start = date.today()
        ipo_end = ipo_start + timedelta(days=60)
        session = requests.Session()
        ipo_rows = fetch_alphavantage_upcoming_ipos(args.alphavantage_api_key, session, ipo_start, ipo_end)
        write_upcoming_ipos_sheet(wb, ipo_rows, ipo_start, ipo_end)
        earnings_start = date.today()
        earnings_end = earnings_start + timedelta(days=14)
        earnings_rows = fetch_nasdaq_upcoming_earnings(session, earnings_start, earnings_end)
        write_upcoming_earnings_sheet(wb, earnings_rows, earnings_start, earnings_end, qualified_dates)
        auto_size_columns(ws)
        wb.save(out_path)
        print(f"Wrote single ticker to {out_path} (Single Tickers)")
        return

    def daily_result_sort_key(row: dict[str, Any]) -> tuple[bool, int, str]:
        rank = _coerce_int(row.get("rank")) if args.score_enable else None
        symbol = str(row.get("symbol", "")).strip().upper()
        return (rank is None, rank if rank is not None else 0, symbol)

    for row in sorted(results, key=daily_result_sort_key):
        symbol_display = str(row.get("symbol", "")).strip()
        symbol_key = symbol_display.upper()
        prev_count = int(prev_counts.get(symbol_key, 0))
        ws.append(build_output_row(row, prev_count))

    black = Color(indexed=8)
    white = Color(indexed=9)
    red = Color(indexed=10)
    blue = Color(indexed=12)
    yellow = Color(indexed=13)

    header_fill = PatternFill(fill_type="solid", fgColor=black)
    descriptor_fill = PatternFill(fill_type="solid", fgColor=blue)
    white_fill = PatternFill(fill_type="solid", fgColor=white)
    black_fill = PatternFill(fill_type="solid", fgColor=black)

    header_font = Font(name="Calibri", size=13, bold=True, color=white)
    descriptor_font = Font(name="Calibri", size=11, bold=True, italic=True, color=black)
    symbol_font = Font(name="Calibri", size=13, bold=True, color=white)
    bold_font = Font(name="Calibri", size=11, bold=True, color=black)

    header_align = Alignment(horizontal="center", vertical="center")
    descriptor_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_center_align = Alignment(horizontal="left", vertical="center")
    center_bottom_align = Alignment(horizontal="center", vertical="bottom")
    bottom_align = Alignment(vertical="bottom")

    thin_red = Side(style="thin", color=red)
    thin_black = Side(style="thin", color=black)
    thick_black = Side(style="thick", color=black)

    score_shift = 1 if args.score_enable else 0

    def shift_col_idx(idx: int) -> int:
        if score_shift == 0 or idx <= 2:
            return idx
        return idx + score_shift

    def shift_cols(cols: set[int]) -> set[int]:
        return {shift_col_idx(col) for col in cols}

    header_style_cols = shift_cols({1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25})
    header_value_cols = shift_cols({2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25})
    if args.score_enable:
        header_style_cols.update({3})
        header_value_cols.update({3})
    left_thick_cols = shift_cols({4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 25})
    right_thick_cols = shift_cols({5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26})
    header_right_cols = {shift_col_idx(25), shift_col_idx(26)} if args.score_enable else {25, 26}

    # Column widths from the sample spreadsheet
    base_column_widths = {
        1: 8.0,
        2: 12.0,
        3: 8.0,
        4: 9.0,
        5: 12.0,
        6: 13.5,
        7: 8.44531,
        8: 13.1562,
        9: 9.39844,
        10: 10.6641,
        11: 13.0,
        12: 10.0,
        13: 11.0,
        14: 14.0,
        15: 10.0,
        16: 17.0,
        17: 15.0,
        18: 11.0,
        19: 9.0,
        20: 13.0,
        21: 12.2812,
        22: 14.7578,
        23: 15.6797,
        24: 12.5312,
        25: 14.2734,
        26: 13.0,
    }
    column_widths = (
        {shift_col_idx(k): v for k, v in base_column_widths.items()} if args.score_enable else base_column_widths
    )
    if args.score_enable:
        column_widths[3] = 6.0
    for col_idx, width in column_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.row_dimensions[1].height = 50.85
    ws.row_dimensions[2].height = 34.85
    for row_idx in range(3, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 17.0

    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        if col_idx in header_style_cols:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
        if col_idx in header_value_cols:
            cell.number_format = "@"
        cell.border = Border(
            left=thin_red if col_idx == 1 else None,
            right=thin_red if col_idx in header_right_cols else None,
            top=thin_red,
        )

    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.fill = white_fill
        cell.font = descriptor_font
        cell.alignment = descriptor_align
        if cell.value is not None:
            cell.number_format = "@"
        cell.border = Border(
            left=thin_red if col_idx == 1 else (thick_black if col_idx in left_thick_cols else None),
            right=thick_black if col_idx in right_thick_cols else None,
        )

    bold_cols = shift_cols({4, 6, 8})
    left_center_cols = shift_cols({7, 9, 11})
    center_bottom_cols = shift_cols({17, 19, 21})
    # Percentage-like values are already computed as percent points (e.g. 21.59),
    # so use a literal percent-sign format to avoid Excel multiplying by 100.
    pct_literal_format = '0.00"%"'
    base_number_formats = {
        1: "@",
        2: "@",
        3: "0",
        4: '"$"#,##0.00',
        5: pct_literal_format,
        6: '"$"#,##0.00',
        7: "General",
        8: '"$"#,##0.00',
        9: "General",
        10: "mmm d, yyyy",
        11: "General",
        12: "0.00",
        13: pct_literal_format,
        14: "0.00",
        15: pct_literal_format,
        16: pct_literal_format,
        17: "0.00",
        18: pct_literal_format,
        19: "0.00",
        20: pct_literal_format,
        21: "0.00",
        22: pct_literal_format,
        23: '"$"#,##0.00',
        24: pct_literal_format,
        25: "mmm d, yyyy",
        26: "mmm d, yyyy",
    }
    number_formats = (
        {shift_col_idx(k): v for k, v in base_number_formats.items()}
        if args.score_enable
        else base_number_formats
    )
    if args.score_enable:
        number_formats[3] = "0"

    for row_idx in range(3, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.fill = black_fill if col_idx == 1 else white_fill
            if col_idx == 1:
                cell.font = symbol_font
            elif col_idx in bold_cols:
                cell.font = bold_font
            if col_idx in left_center_cols:
                cell.alignment = left_center_align
            elif col_idx in center_bottom_cols:
                cell.alignment = center_bottom_align
            else:
                cell.alignment = bottom_align
            if col_idx in number_formats:
                cell.number_format = number_formats[col_idx]
            cell.border = Border(
                left=thick_black if col_idx in left_thick_cols else thin_black,
                right=thick_black if col_idx in right_thick_cols else thin_black,
            )

    merge_pairs = [
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
        (25, 26),
    ]
    if args.score_enable:
        merge_pairs = [(shift_col_idx(start), shift_col_idx(end)) for start, end in merge_pairs]
    for start_col, end_col in merge_pairs:
        ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)

    ws.freeze_panes = None

    ipo_start = date.today()
    ipo_end = ipo_start + timedelta(days=60)
    session = requests.Session()
    ipo_rows = fetch_alphavantage_upcoming_ipos(args.alphavantage_api_key, session, ipo_start, ipo_end)
    write_upcoming_ipos_sheet(wb, ipo_rows, ipo_start, ipo_end)
    earnings_start = date.today()
    earnings_end = earnings_start + timedelta(days=14)
    earnings_rows = fetch_nasdaq_upcoming_earnings(session, earnings_start, earnings_end)
    write_upcoming_earnings_sheet(wb, earnings_rows, earnings_start, earnings_end, qualified_dates)
    write_top10_ohlc_tracking_sheet(wb, symbol_paths, root, top_n=10)
    prune_old_run_sheets(wb, keep_runs=15)

    wb.save(out_path)

    print(f"Wrote {len(results)} matches to {out_path} ({sheet_name})")


if __name__ == "__main__":
    main()
