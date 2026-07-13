from __future__ import annotations

import html
import os
import smtplib
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


RESULTS_PATH = Path(os.environ.get("RESULTS_FILE", "results.xlsx"))
UPCOMING_IPOS_SHEET_NAME = "Upcoming IPOs (60D)"
UPCOMING_EARNINGS_SHEET_NAME = "Upcoming Earnings (14D)"
SIMULATION_SHEET_NAME = "Simulation"
LEGACY_SIMULATION_SHEET_NAME = "Summary"
PROTECTED_SHEETS = {
    "Single Tickers",
    UPCOMING_IPOS_SHEET_NAME,
    UPCOMING_EARNINGS_SHEET_NAME,
    "Top 10 OHLC Tracking",
    SIMULATION_SHEET_NAME,
    LEGACY_SIMULATION_SHEET_NAME,
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
        if '0.00"%"' in number_format:
            return f"{value:.2f}%"
        if "0.00%" in number_format:
            return f"{value:.2%}"
        if "0%" in number_format and abs(value) <= 1:
            return f"{value:.0%}"
        if "$" in number_format:
            return f"${value:,.2f}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        if '0.00"%"' in number_format:
            return f"{value:.2f}%"
        if "0.00%" in number_format:
            return f"{value:.2%}"
        if "0%" in number_format and abs(value) <= 1:
            return f"{value:.0%}"
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


def worksheet_table(wb, sheet_name: str, max_rows: int | None = None) -> str:
    if sheet_name not in wb.sheetnames:
        return f"<p>No {html.escape(sheet_name)} sheet was found.</p>"

    ws = wb[sheet_name]
    header_row = None
    for row_idx in range(1, min(ws.max_row, 10) + 1):
        values = [ws.cell(row=row_idx, column=col_idx).value for col_idx in range(1, ws.max_column + 1)]
        if sum(value is not None for value in values) > 1:
            header_row = row_idx
            break
    if header_row is None:
        return f"<p>No rows found in {html.escape(sheet_name)}.</p>"

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
        if max_rows is not None and len(rows) >= max_rows:
            break

    return html_table(headers, rows)


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
    return worksheet_table(wb, UPCOMING_IPOS_SHEET_NAME, max_rows=20)


def parse_cell_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        value = value.strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return None


def upcoming_earnings_table(wb) -> str:
    if UPCOMING_EARNINGS_SHEET_NAME not in wb.sheetnames:
        return f"<p>No {html.escape(UPCOMING_EARNINGS_SHEET_NAME)} sheet was found.</p>"

    ws = wb[UPCOMING_EARNINGS_SHEET_NAME]
    header_row = None
    for row_idx in range(1, min(ws.max_row, 10) + 1):
        headers = [
            str(ws.cell(row=row_idx, column=col_idx).value or "").strip()
            for col_idx in range(1, ws.max_column + 1)
        ]
        normalized = {header.lower(): col_idx for col_idx, header in enumerate(headers, start=1)}
        if {"symbol", "company", "earnings date"}.issubset(normalized):
            header_row = row_idx
            symbol_col = normalized["symbol"]
            company_col = normalized["company"]
            date_col = normalized["earnings date"]
            break
    else:
        return f"<p>No earnings table was found in {html.escape(UPCOMING_EARNINGS_SHEET_NAME)}.</p>"

    today = date.today()
    end_date = today + timedelta(days=7)
    rows: list[list[str]] = []
    for row_idx in range(header_row + 1, ws.max_row + 1):
        earnings_date = parse_cell_date(ws.cell(row=row_idx, column=date_col).value)
        if earnings_date is None or earnings_date < today or earnings_date > end_date:
            continue

        company = format_value(
            ws.cell(row=row_idx, column=company_col).value,
            ws.cell(row=row_idx, column=company_col).number_format,
        )
        symbol = format_value(
            ws.cell(row=row_idx, column=symbol_col).value,
            ws.cell(row=row_idx, column=symbol_col).number_format,
        )
        rows.append([format_value(earnings_date), company, symbol])

    rows.sort(key=lambda row: (parse_cell_date(row[0]) or end_date, row[2]))
    return html_table(["Date", "Company", "Symbol"], rows)



def simulation_totals_html(wb) -> str:
    sheet_name = next(
        (name for name in (SIMULATION_SHEET_NAME, LEGACY_SIMULATION_SHEET_NAME) if name in wb.sheetnames),
        None,
    )
    if sheet_name is None:
        return "<p>Simulated Portfolio totals: Simulation sheet not found.</p>"

    ws = wb[sheet_name]
    total_label_row = None
    for row_idx in range(1, ws.max_row + 1):
        labels = {
            str(ws.cell(row=row_idx, column=7).value or "").strip().upper(),
            str(ws.cell(row=row_idx, column=8).value or "").strip().upper(),
        }
        if "TOTAL" in labels:
            total_label_row = row_idx
            break

    if total_label_row is None:
        return "<p>Simulated Portfolio totals: TOTAL row not found.</p>"

    total_formula_row = total_label_row + 1
    dollar_value = ws.cell(row=total_formula_row, column=7).value
    percent_value = ws.cell(row=total_formula_row, column=8).value

    if not isinstance(dollar_value, (int, float)):
        dollar_value = sum(
            float(ws.cell(row=row_idx, column=7).value or 0)
            for row_idx in range(2, total_label_row)
            if isinstance(ws.cell(row=row_idx, column=7).value, (int, float))
        )
    if not isinstance(percent_value, (int, float)):
        percent_value = sum(
            float(ws.cell(row=row_idx, column=8).value or 0)
            for row_idx in range(2, total_label_row)
            if isinstance(ws.cell(row=row_idx, column=8).value, (int, float))
        )

    dollar_total = format_value(dollar_value, '"$"#,##0.00')
    percent_total = format_value(percent_value, "0.00%")
    return f"<p><strong>Simulated Portfolio totals:</strong> {html.escape(dollar_total)} and {html.escape(percent_total)}.</p>"


def build_email_html() -> tuple[str, str]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Results workbook not found: {RESULTS_PATH}")

    wb = load_workbook(RESULTS_PATH, data_only=True)
    ranked_title, ranked_html = ranked_stocks_table(wb)
    simulation_totals = simulation_totals_html(wb)
    ipo_html = upcoming_ipos_table(wb)
    earnings_html = upcoming_earnings_table(wb)
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
        {simulation_totals}
        <h2>{ranked_title}</h2>
        {ranked_html}
        <h2>Upcoming IPOs</h2>
        {ipo_html}
        <h2>Upcoming Earnings</h2>
        {earnings_html}
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
