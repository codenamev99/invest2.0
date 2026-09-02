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


def collect_tickers(roots: list[Path]) -> list[str]:
    """
    Collect tickers from *.txt files under every root that exists.

    A missing folder is skipped rather than fatal: a NYSE-only checkout has no
    "nasdaq stocks" directory, and a run that covered one venue should not fail
    just because the other has not been bootstrapped yet.
    """
    out: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"Skipping missing folder: {root}")
            continue
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
        nargs="+",
        default=[
            "${workspaceFolder}/data/daily/us/nyse stocks",
            "${workspaceFolder}/data/daily/us/nasdaq stocks",
        ],
        help="One or more folders of Stooq-format *.txt files to pool into the universe",
    )
    ap.add_argument(
        "--out",
        default="${workspaceFolder}/us_tickers.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    roots = [resolve_path(d) for d in args.dir]
    out_path = resolve_path(args.out)

    if not any(root.exists() for root in roots):
        raise SystemExit(
            "None of the requested folders exist: " + ", ".join(str(r) for r in roots)
        )

    tickers = collect_tickers(roots)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol"])
        for t in tickers:
            w.writerow([t])

    print(f"Wrote {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    main()
