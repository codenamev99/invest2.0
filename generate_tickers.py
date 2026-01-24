from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


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


def collect_tickers(root: Path) -> list[str]:
    """
    Collect tickers from *.txt files under root.
    """
    out: set[str] = set()
    for p in root.rglob("*.txt"):
        sym = p.name.replace(".txt", "").upper()
        if not sym.endswith(".US"):
            sym = f"{sym}.US"
        out.add(sym)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="${workspaceFolder}/data/daily/us/nyse stocks",
        help="Folder containing Stooq NYSE *.txt files",
    )
    ap.add_argument(
        "--out",
        default="${workspaceFolder}/nyse_tickers.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    root = resolve_path(args.dir)
    out_path = resolve_path(args.out)

    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")

    tickers = collect_tickers(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol"])
        for t in tickers:
            w.writerow([t])

    print(f"Wrote {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    main()
