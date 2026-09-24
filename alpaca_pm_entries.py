"""
Submit real Alpaca paper-trading limit orders for the day's top-10 ranked
cohort, while the PM extended-hours session (4:00-8:00pm ET) is still open.

Reads the JSON written by `screen_stooq.py --rank_only` and reuses the exact
regime gate (evaluate_market_regime) and headroom formula screen_stooq.py's
PM Simulation applies, so a real order is only submitted for rows that would
count as "Good" in the spreadsheet's own totals.

No fill exists yet at submission time, so the live quote fetched here stands
in both as the limit-price base (+ LIMIT_BUFFER) and as the headroom
formula's entry-price proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

import screen_stooq as s

# Must match build_investment_simulation_rows' defaults in screen_stooq.py.
GAIN_PCT = 0.02
POSITION_SIZE = 10_000.0
LIMIT_BUFFER = 0.05


def load_top10(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top10", default="top10.json")
    ap.add_argument("--root", default="data 2/daily/us")
    ap.add_argument("--state_out", default="")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute gating and limit prices but do not submit orders.",
    )
    args = ap.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        sys.exit("ALPACA_API_KEY / ALPACA_SECRET_KEY are required.")

    top10 = load_top10(Path(args.top10))
    if not top10:
        print("No ranked rows to process.")
        return
    rank_date = date.fromisoformat(top10[0]["rank_date"])

    root = Path(args.root)
    symbol_paths = s.build_file_map(root)
    market_regime = s.evaluate_market_regime(rank_date, symbol_paths, root, {})
    entry_allowed = bool(market_regime.get("entry_allowed", True))

    trading = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    state_out = (
        Path(args.state_out)
        if args.state_out
        else Path("alpaca_state") / f"{rank_date.isoformat()}.json"
    )
    state_out.parent.mkdir(parents=True, exist_ok=True)

    # A re-run must not clobber exit-tracking fields alpaca_pm_exits.py may
    # have already added to this same day's file (normally impossible --
    # exits only ever runs the morning after entries -- but cheap insurance
    # against a manual re-trigger).
    existing_by_symbol: dict[str, dict[str, Any]] = {}
    if state_out.exists():
        try:
            with open(state_out) as f:
                for prior_row in json.load(f):
                    sym = str(prior_row.get("symbol", "")).strip().upper()
                    if sym:
                        existing_by_symbol[sym] = prior_row
        except (OSError, ValueError):
            pass

    results: list[dict[str, Any]] = []
    for row in top10:
        symbol = row["symbol"]
        outcome: dict[str, Any] = {
            "rank": row["rank"],
            "symbol": symbol,
            "rank_date": rank_date.isoformat(),
            "gate": None,
            "limit_price": None,
            "qty": None,
            "alpaca_order_id": None,
            "submitted_at": None,
        }

        if not entry_allowed:
            outcome["gate"] = f"blocked-regime: {market_regime.get('reason', '')}"
            results.append(outcome)
            continue

        normalized = s.normalize_symbol(symbol)
        path = symbol_paths.get(normalized) or s.find_symbol_file(root, normalized)
        if path is None:
            outcome["gate"] = "skipped: no local daily price file"
            results.append(outcome)
            continue
        dates, _opens, highs, lows, _closes = s.load_ohlc_from_file(path)
        variance_4m = s.average_daily_variance(dates, highs, lows, rank_date)

        try:
            quote = data_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol)
            )
            last_price = float(quote[symbol].price)
        except Exception as exc:
            outcome["gate"] = f"skipped: quote fetch failed ({exc})"
            results.append(outcome)
            continue

        # Mirrors the headroom test in build_investment_simulation_rows
        # (screen_stooq.py:2856-2868): the live quote stands in for
        # entry_price since no fill exists yet, and rank_date's own close
        # (written by --rank_only) stands in for prev_close.
        prev_close = float(row.get("closing_price") or 0.0)
        if prev_close > 0 and variance_4m is not None:
            move_from_close = (last_price / prev_close) - 1.0
            run_up = max(0.0, move_from_close)
            headroom = variance_4m - run_up
            if headroom < GAIN_PCT:
                outcome["gate"] = (
                    f"excluded-headroom: headroom {headroom:.4f} < target {GAIN_PCT}"
                )
                results.append(outcome)
                continue

        limit_price = round(last_price + LIMIT_BUFFER, 2)
        # Extended-hours orders are whole-share only.
        qty = int(POSITION_SIZE // limit_price)
        if qty < 1:
            outcome["gate"] = "skipped: position size below one share at this price"
            results.append(outcome)
            continue

        outcome["limit_price"] = limit_price
        outcome["qty"] = qty

        if args.dry_run:
            outcome["gate"] = "dry-run: not submitted"
            results.append(outcome)
            continue

        order_req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            extended_hours=True,
            # Deterministic per (rank_date, symbol) so a retried/duplicate CI
            # run can't double-buy -- Alpaca rejects a repeated order id.
            client_order_id=f"pm-{rank_date.isoformat()}-{symbol}",
        )
        try:
            order = trading.submit_order(order_req)
            outcome["gate"] = "submitted"
            outcome["alpaca_order_id"] = str(order.id)
            outcome["submitted_at"] = datetime.now(s.EASTERN_TZ).isoformat()
        except APIError as exc:
            # A retried/duplicate run hits Alpaca's own client_order_id
            # uniqueness check -- that's the idempotency guard working, not a
            # failure. Look the original order back up so the state file
            # still reflects "submitted" with its real id, instead of
            # clobbering a prior successful submission with an error row
            # that alpaca_pm_exits.py would then never pick up.
            if "client_order_id must be unique" in str(exc):
                try:
                    existing = trading.get_order_by_client_id(order_req.client_order_id)
                    outcome["gate"] = "submitted"
                    outcome["alpaca_order_id"] = str(existing.id)
                    outcome["submitted_at"] = (
                        existing.submitted_at or datetime.now(s.EASTERN_TZ)
                    ).isoformat()
                except Exception as lookup_exc:
                    outcome["gate"] = f"error: duplicate order, lookup failed ({lookup_exc})"
            else:
                outcome["gate"] = f"error: {exc}"
        prior_row = existing_by_symbol.get(symbol)
        if prior_row:
            for key, value in prior_row.items():
                if key not in outcome:
                    outcome[key] = value
        results.append(outcome)

    with open(state_out, "w") as f:
        json.dump(results, f, indent=2)
    submitted = sum(1 for r in results if r["gate"] == "submitted")
    print(f"Wrote {state_out}: {submitted}/{len(results)} orders submitted")


if __name__ == "__main__":
    main()
