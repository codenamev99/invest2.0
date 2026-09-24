"""
Manage real Alpaca paper positions opened by alpaca_pm_entries.py: once an
entry fills, submit a +2%/-1% OCO exit mirroring PM Simulation's own
target/stop, and force-close after the same 5-trading-day timeout the
simulation uses (build_investment_simulation_rows, screen_stooq.py).

Runs once regular hours are open (~9:35am ET) since Alpaca doesn't accept
OCO/stop orders in extended hours -- the entry fills in the prior evening's
after-hours session, but the exit can only be placed the next morning.

State lives in the same alpaca_state/<rank_date>.json files alpaca_pm_entries.py
writes, extended in place with exit_order_id/exit_status/exit fields. Each run
scans a rolling window of recent files rather than "just yesterday" so a
skipped or failed run self-heals on the next one.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest

import screen_stooq as s

GAIN_PCT = 0.02
LOSS_PCT = 0.01
FOLLOW_DAYS = 5
STATE_DIR = Path("alpaca_state")
SCAN_WINDOW_DAYS = 10

# An entry order in any of these statuses never became a position.
TERMINAL_WITHOUT_FILL = {"canceled", "expired", "rejected", "done_for_day"}


def iter_pending_rows(scan_window_days: int = SCAN_WINDOW_DAYS):
    today = date.today()
    for offset in range(scan_window_days):
        day = today - timedelta(days=offset)
        path = STATE_DIR / f"{day.isoformat()}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                rows = json.load(f)
        except (OSError, ValueError):
            continue
        for row in rows:
            if row.get("gate") != "submitted":
                continue
            if row.get("exit_status") in {"target_hit", "stop_hit", "timeout_closed", "entry_unfilled"}:
                continue
            # Rows are yielded alongside the list they came from, so callers
            # mutate in place and save the whole file once at the end.
            yield path, rows, row


def trading_days_elapsed(symbol: str, rank_date: date, root: Path, symbol_paths: dict[str, Path]) -> int | None:
    normalized = s.normalize_symbol(symbol)
    path = symbol_paths.get(normalized) or s.find_symbol_file(root, normalized)
    if path is None:
        return None
    dates, _opens, _highs, _lows, _closes = s.load_ohlc_from_file(path)
    if len(dates) == 0:
        return None
    rank_idx = int(np.searchsorted(dates, s.date_to_int(rank_date), side="left"))
    today_idx = int(np.searchsorted(dates, s.date_to_int(date.today()), side="right") - 1)
    if rank_idx >= len(dates) or today_idx < rank_idx:
        return None
    return today_idx - rank_idx


def resolve_oco(exit_order) -> tuple[str, float, Any] | None:
    """
    Read an OCO exit order's real outcome.

    The order alpaca_pm_exits.py submits and stores as exit_order_id is the
    take-profit (limit) leg; Alpaca creates the stop-loss leg as its sibling.
    If the take-profit leg fills, its own status is FILLED. If the stop-loss
    leg fills instead, OCO semantics cancel its sibling -- so the *tracked*
    take-profit order shows status CANCELED, and the fill has to be read off
    the sibling leg (exit_order.legs), not the tracked order itself.

    Returns (outcome, exit_price, filled_at) or None if still unresolved.
    """
    status = str(exit_order.status).split(".")[-1].lower()
    if status == "filled":
        return "target_hit", float(exit_order.filled_avg_price), exit_order.filled_at

    if status == "canceled":
        for leg in exit_order.legs or []:
            if str(leg.status).split(".")[-1].lower() == "filled":
                return "stop_hit", float(leg.filled_avg_price), leg.filled_at
        # Canceled with no filled sibling -- not something this job's own
        # flow produces (the timeout path cancels and marks terminal in the
        # same step); likely a manual cancellation. Leave unresolved rather
        # than guess, so it surfaces for a human to look at.
        return None

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data 2/daily/us")
    args = ap.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY are required.")

    root = Path(args.root)
    symbol_paths = s.build_file_map(root)
    trading = TradingClient(api_key, secret_key, paper=True)

    touched_files: dict[Path, list[dict[str, Any]]] = {}
    for path, rows, row in iter_pending_rows():
        symbol = row["symbol"]
        rank_date = date.fromisoformat(row["rank_date"])
        touched_files.setdefault(path, rows)

        exit_status = row.get("exit_status")

        if exit_status is None:
            # Entry order status hasn't been checked yet.
            entry_order = trading.get_order_by_id(row["alpaca_order_id"])
            filled_qty = float(entry_order.filled_qty or 0)
            if filled_qty <= 0:
                if str(entry_order.status).split(".")[-1].lower() in TERMINAL_WITHOUT_FILL:
                    row["exit_status"] = "entry_unfilled"
                # else: still resting, nothing to do yet.
                continue

            fill_price = float(entry_order.filled_avg_price)
            target_price = round(fill_price * (1 + GAIN_PCT), 2)
            stop_price = round(fill_price * (1 - LOSS_PCT), 2)
            oco = LimitOrderRequest(
                symbol=symbol,
                qty=filled_qty,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=target_price),
                stop_loss=StopLossRequest(stop_price=stop_price),
                client_order_id=f"pm-exit-{rank_date.isoformat()}-{symbol}",
            )
            exit_order = trading.submit_order(oco)
            row["fill_price"] = fill_price
            row["fill_qty"] = filled_qty
            row["target_price"] = target_price
            row["stop_price"] = stop_price
            row["exit_order_id"] = str(exit_order.id)
            row["exit_status"] = "oco_open"
            continue

        if exit_status == "oco_open":
            exit_order = trading.get_order_by_id(row["exit_order_id"])
            resolved = resolve_oco(exit_order)
            if resolved is not None:
                outcome, exit_price, filled_at = resolved
                row["exit_status"] = outcome
                row["exit_price"] = exit_price
                row["exit_at"] = (filled_at or datetime.now(s.EASTERN_TZ)).isoformat()
                continue

            still_open = str(exit_order.status).split(".")[-1].lower() not in {
                "canceled",
                "expired",
                "rejected",
            }
            if not still_open:
                # resolve_oco() couldn't explain this (no filled sibling
                # leg) -- most likely a manual cancellation outside this
                # job. Leave it for a human rather than guess or crash on a
                # cancel-of-an-already-canceled-order.
                continue

            elapsed = trading_days_elapsed(symbol, rank_date, root, symbol_paths)
            if elapsed is not None and elapsed >= FOLLOW_DAYS:
                trading.cancel_order_by_id(row["exit_order_id"])
                closed = trading.close_position(symbol)
                row["exit_status"] = "timeout_closed"
                row["exit_price"] = float(closed.filled_avg_price) if getattr(closed, "filled_avg_price", None) else None
                row["exit_at"] = datetime.now(s.EASTERN_TZ).isoformat()
            # else: still within the window, leave resting.

    for path, rows in touched_files.items():
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"Checked {sum(len(r) for r in touched_files.values())} rows across {len(touched_files)} file(s).")


if __name__ == "__main__":
    main()
