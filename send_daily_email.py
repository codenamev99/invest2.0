from __future__ import annotations

import html
import os
import smtplib
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


RESULTS_PATH = Path(os.environ.get("RESULTS_FILE", "results.xlsx"))
UPCOMING_IPOS_SHEET_NAME = "Upcoming IPOs (60D)"
PROTECTED_SHEETS = {
    "Single Tickers",
    UPCOMING_IPOS_SHEET_NAME,
    "Upcoming Earnings (14D)",
    "Top 10 OHLC Tracking",
    "Summary",
    "Investment Dashboard",
}


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def parse_run_sheet_date(sheet_name: str) -> date | None:
    try:
        return datetime.strptime(sheet_name.strip(), "%d %b %Y").date()
    except ValueError:
        return None


def latest_run_sheet_name(sheet_names: list[str]) -> str | None:
    candidates = [
        (parsed, name)
        for name in sheet_names
        if name not in PROTECTED_SHEETS
        for parsed in [parse_run_sheet_date(name)]
        if parsed is not None
    ]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    for name in reversed(sheet_names):
        if name not in PROTECTED_SHEETS:
            return name
    return None


def format_value(value: Any, number_format: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%b %-d, %Y")
    if isinstance(value, date):
        return value.strftime("%b %-d, %Y")
    if isinstance(value, float):
        if "0.00%" in number_format or '0.00"%"' in number_format:
            return f"{value:.2f}%"
        if "0%" in number_format and abs(value) <= 1:
            return f"{value:.0%}"
        if "$" in number_format:
            return f"${value:,.2f}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        if "$" in number_format:
            return f"${value:,.2f}"
        return f"{value:,}"
    return str(value)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p>No rows found.</p>"

    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_html.append(f"<tr>{cells}</tr>")

    return f"""
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{''.join(body_html)}</tbody>
    </table>
    """


def ranked_stocks_table(wb) -> tuple[str, str]:
    sheet_name = latest_run_sheet_name(wb.sheetnames)
    if not sheet_name:
        return "Latest Ranked Stocks", "<p>No ranked stock sheet was found.</p>"

    ws = wb[sheet_name]
    rank_col = 3 if str(ws.cell(row=1, column=3).value or "").strip().lower() == "rank" else None
    selected_cols = [
        ("Symbol", 1),
        ("Company", 2),
    ]
    if rank_col:
        selected_cols.append(("Rank", rank_col))
    selected_cols.extend(
        [
            ("Close", 5 if rank_col else 4),
            ("ATR %", 17 if rank_col else 16),
            ("Avg $ Vol", 24 if rank_col else 23),
            ("Next Earnings", 27 if rank_col else 26),
        ]
    )

    rows: list[list[str]] = []
    for row_idx in range(3, ws.max_row + 1):
        symbol = ws.cell(row=row_idx, column=1).value
        if not symbol:
            continue
        rows.append(
            [
                format_value(ws.cell(row=row_idx, column=col_idx).value, ws.cell(row=row_idx, column=col_idx).number_format)
                for _, col_idx in selected_cols
            ]
        )
        if len(rows) >= 10:
            break

    headers = [label for label, _ in selected_cols]
    title = f"Latest Ranked Stocks ({html.escape(sheet_name)})"
    return title, html_table(headers, rows)


def upcoming_ipos_table(wb) -> str:
    if UPCOMING_IPOS_SHEET_NAME not in wb.sheetnames:
        return "<p>No upcoming IPO sheet was found.</p>"

    ws = wb[UPCOMING_IPOS_SHEET_NAME]
    header_row = None
    for row_idx in range(1, min(ws.max_row, 10) + 1):
        values = [ws.cell(row=row_idx, column=col_idx).value for col_idx in range(1, ws.max_column + 1)]
        if any(values):
            header_row = row_idx
            break
    if header_row is None:
        return "<p>No upcoming IPO rows found.</p>"

    headers = [
        str(ws.cell(row=header_row, column=col_idx).value or "").strip() or f"Column {col_idx}"
        for col_idx in range(1, ws.max_column + 1)
    ]
    rows: list[list[str]] = []
    for row_idx in range(header_row + 1, ws.max_row + 1):
        values = [ws.cell(row=row_idx, column=col_idx).value for col_idx in range(1, ws.max_column + 1)]
        if not any(values):
            continue
        rows.append(
            [
                format_value(ws.cell(row=row_idx, column=col_idx).value, ws.cell(row=row_idx, column=col_idx).number_format)
                for col_idx in range(1, ws.max_column + 1)
            ]
        )
        if len(rows) >= 20:
            break

    return html_table(headers, rows)


def build_email_html() -> tuple[str, str]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Results workbook not found: {RESULTS_PATH}")

    wb = load_workbook(RESULTS_PATH, data_only=True)
    ranked_title, ranked_html = ranked_stocks_table(wb)
    ipo_html = upcoming_ipos_table(wb)
    run_date = datetime.now().strftime("%b %-d, %Y")
    subject_prefix = os.environ.get("EMAIL_SUBJECT_PREFIX", "Daily Screener").strip() or "Daily Screener"
    subject = f"{subject_prefix} - {run_date}"

    body = f"""
    <!doctype html>
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; color: #111; }}
          table {{ border-collapse: collapse; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #bbb; padding: 6px 8px; text-align: left; }}
          th {{ background: #f0f0f0; }}
        </style>
      </head>
      <body>
        <p>Attached is the latest <code>results.xlsx</code> workbook.</p>
        <h2>{ranked_title}</h2>
        {ranked_html}
        <h2>Upcoming IPOs</h2>
        {ipo_html}
      </body>
    </html>
    """
    return subject, body


def send_email(subject: str, html_body: str) -> None:
    smtp_host = required_env("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = required_env("SMTP_USERNAME")
    smtp_password = required_env("SMTP_PASSWORD")
    email_from = os.environ.get("EMAIL_FROM", smtp_username).strip() or smtp_username
    recipients = [addr.strip() for addr in required_env("EMAIL_TO").replace(";", ",").split(",") if addr.strip()]
    if not recipients:
        raise SystemExit("EMAIL_TO did not contain any recipient addresses.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_from
    message["To"] = ", ".join(recipients)
    message.set_content("Your email client does not support HTML. See the attached results workbook.")
    message.add_alternative(html_body, subtype="html")

    attach_results = os.environ.get("EMAIL_ATTACH_RESULTS", "true").strip().lower() not in {"0", "false", "no"}
    if attach_results:
        with RESULTS_PATH.open("rb") as f:
            message.add_attachment(
                f.read(),
                maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=RESULTS_PATH.name,
            )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as smtp:
        smtp.starttls()
        smtp.login(smtp_username, smtp_password)
        smtp.send_message(message)


def main() -> None:
    subject, html_body = build_email_html()
    send_email(subject, html_body)
    print("Daily screener email sent.")


if __name__ == "__main__":
    main()
