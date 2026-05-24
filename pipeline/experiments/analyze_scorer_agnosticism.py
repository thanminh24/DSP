"""Analyze Run B: scorer-agnosticism-sweep.csv → scorer comparison tables."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "scorer-agnosticism-sweep.csv"


def _g_mean(df):
    return np.sqrt(df["minority_recall"].clip(lower=0) * df["majority_recall"].clip(lower=0))


def main():
    if not OUTPUT_CSV.exists():
        print(f"ERROR: {OUTPUT_CSV} not found. Run Run B first.", flush=True)
        sys.exit(1)
    df = pd.read_csv(OUTPUT_CSV)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()
    df["g_mean"] = _g_mean(df)

    print(f"Loaded {len(df)} valid rows from {OUTPUT_CSV}\n")

    # --- Table 1: Mean BA per scorer ---
    print("=" * 70)
    print("Table 1 — Mean BA per scorer (n=50 pairs each)")
    print("=" * 70)
    methods = ["no_cleaning", "class_proportional", "cwms_msbs", "cwms_msbs_knn", "cwms_msbs_crossfamily"]
    print(f"{'Method':<25} {'BA':>8} {'G-mean':>8} {'Min.Recall':>10}")
    print("-" * 70)
    for method in methods:
        mdf = df[df["method"] == method]
        if len(mdf) == 0:
            continue
        print(f"{method:<25} {mdf['balanced_accuracy'].mean():8.4f} {mdf['g_mean'].mean():8.4f} {mdf['minority_recall'].mean():10.4f}")

    # --- Table 2: Gains vs class_proportional ---
    print("\n" + "=" * 70)
    print("Table 2 — ΔBA vs class_proportional with p-values")
    print("=" * 70)
    cp = df[df["method"] == "class_proportional"].set_index(
        ["dataset", "model", "seed"])["balanced_accuracy"]
    for method in ["cwms_msbs", "cwms_msbs_knn", "cwms_msbs_crossfamily"]:
        m_vals = df[df["method"] == method].set_index(
            ["dataset", "model", "seed"])["balanced_accuracy"]
        common_idx = cp.index.intersection(m_vals.index)
        if len(common_idx) < 5:
            continue
        cp_v = cp.loc[common_idx].values
        m_v = m_vals.loc[common_idx].values
        delta = np.mean(m_v - cp_v) * 100
        _, p = wilcoxon(m_v, cp_v)
        wins = int(np.sum(m_v > cp_v))
        print(f"  {method:<25}: ΔBA={delta:+5.2f}pp  p={p:.1e}  wins={wins}/{len(common_idx)}")

    # --- Table 3: Head-to-head cwms_msbs vs knn vs crossfamily ---
    print("\n" + "=" * 70)
    print("Table 3 — Head-to-head: cwms_msbs vs knn vs crossfamily")
    print("=" * 70)
    base = df[df["method"] == "cwms_msbs"].set_index(
        ["dataset", "model", "seed"])["balanced_accuracy"]
    for method in ["cwms_msbs_knn", "cwms_msbs_crossfamily"]:
        m_vals = df[df["method"] == method].set_index(
            ["dataset", "model", "seed"])["balanced_accuracy"]
        common_idx = base.index.intersection(m_vals.index)
        if len(common_idx) < 5:
            continue
        base_v = base.loc[common_idx].values
        m_v = m_vals.loc[common_idx].values
        delta = np.mean(base_v - m_v) * 100
        _, p = wilcoxon(base_v, m_v)
        wins = int(np.sum(base_v > m_v))
        print(f"  cwms_msbs vs {method:<20}: ΔBA={delta:+5.2f}pp  p={p:.1e}  wins={wins}/{len(common_idx)}")

    # --- Table 4: Per-dataset breakdown ---
    print("\n" + "=" * 70)
    print("Table 4 — Per-dataset breakdown (cwms_msbs BA)")
    print("=" * 70)
    datasets = sorted(df["dataset"].unique())
    print(f"{'Dataset':<12}", end="")
    for method in methods:
        print(f" {method:>10}", end="")
    print()
    print("-" * (12 + 11 * len(methods)))
    for ds in datasets:
        ddf = df[df["dataset"] == ds]
        print(f"{ds:<12}", end="")
        for method in methods:
            vals = ddf[ddf["method"] == method]["balanced_accuracy"]
            print(f" {vals.mean():10.4f}" if len(vals) else "       N/A", end="")
        print()


if __name__ == "__main__":
    main()
