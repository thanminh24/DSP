"""Prepare weak-supervision dataset cache instructions.

This script intentionally does not download data automatically. WRENCH and similar
datasets have multiple distribution formats; cache one as CSV/Parquet with columns:
`label` for weak/noisy labels and `gold` for clean evaluation labels.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data/weak-supervision")
CANDIDATES = ["bank-marketing", "bioresponse", "mushroom", "phishing", "spambase"]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache weak-supervision datasets under: {DATA_DIR.resolve()}")
    print("Required schema: feature columns + label + gold")
    print("Candidate datasets:")
    for name in CANDIDATES:
        print(f"  - {name}.csv or {name}.parquet")
    print("\nNo network download performed by this script.")


if __name__ == "__main__":
    main()
