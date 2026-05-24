"""Analyze Run C: competitor-headtohead.csv → paper Table 2."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead.csv"


def _g_mean(df):
    return np.sqrt(df["minority_recall"].clip(lower=0) * df["majority_recall"].clip(lower=0))


def main():
    if not OUTPUT_CSV.exists():
        print(f"ERROR: {OUTPUT_CSV} not found. Run Run C first.", flush=True)
        sys.exit(1)
    df = pd.read_csv(OUTPUT_CSV)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()
    df["g_mean"] = _g_mean(df)

    print(f"Loaded {len(df)} valid rows from {OUTPUT_CSV}\n")

    method_order = ["no_cleaning", "smote", "class_proportional", "iw_smote", "sw_framework", "cwms_msbs"]
    method_labels = {
        "no_cleaning": "no_cleaning",
        "smote": "SMOTE [Chawla 2002]",
        "class_proportional": "class_proportional [He & Garcia 2009]",
        "iw_smote": "IW-SMOTE [Zhang et al. 2022]",
        "sw_framework": "SW-approx [X et al. 2022]",
        "cwms_msbs": "CWMS+MSBS (ours)",
    }

    # --- Table A: Mean BA/G-mean/RecMin by method ---
    print("=" * 80)
    print("Table A — Mean performance per method (n=50 pairs)")
    print("=" * 80)
    print(f"{'Method':<40} {'BA':>8} {'G-mean':>8} {'Min.Recall':>10}")
    print("-" * 80)
    for method in method_order:
        mdf = df[df["method"] == method]
        if len(mdf) == 0:
            continue
        label = method_labels.get(method, method)
        print(f"{label:<40} {mdf['balanced_accuracy'].mean():8.4f} {mdf['g_mean'].mean():8.4f} {mdf['minority_recall'].mean():10.4f}")

    # --- Table B: Wilcoxon pairwise: CWMS+MSBS vs each competitor ---
    print("\n" + "=" * 80)
    print("Table B — CWMS+MSBS vs each competitor (Wilcoxon signed-rank)")
    print("=" * 80)
    # Pairing on (dataset, seed) only — competitor CSV has single protocol.
    # Full benchmark uses (dataset, seed, noise_protocol) — see analyze_full_benchmark.py.
    cwms = df[df["method"] == "cwms_msbs"].set_index(
        ["dataset", "seed"])["balanced_accuracy"]
    for method in method_order:
        if method == "cwms_msbs":
            continue
        m_vals = df[df["method"] == method].set_index(
            ["dataset", "seed"])["balanced_accuracy"]
        common_idx = cwms.index.intersection(m_vals.index)
        if len(common_idx) < 5:
            continue
        cw_v = cwms.loc[common_idx].values
        m_v = m_vals.loc[common_idx].values
        delta = np.mean(cw_v - m_v) * 100
        # Two-sided: competitor headtohead does not pre-assume direction (unlike Table 1 one-sided).
        _, p = wilcoxon(cw_v, m_v)
        wins = int(np.sum(cw_v > m_v))
        label = method_labels.get(method, method)
        print(f"  vs {label:<35}: ΔBA={delta:+5.2f}pp  p={p:.1e}  wins={wins}/{len(common_idx)}")

    # --- Table C: Per-dataset breakdown ---
    print("\n" + "=" * 80)
    print("Table C — Per-dataset BA breakdown")
    print("=" * 80)
    datasets = sorted(df["dataset"].unique())
    print(f"{'Dataset':<12}", end="")
    for method in method_order:
        print(f" {method:>12}", end="")
    print()
    print("-" * (12 + 13 * len(method_order)))
    for ds in datasets:
        ddf = df[df["dataset"] == ds]
        print(f"{ds:<12}", end="")
        for method in method_order:
            vals = ddf[ddf["method"] == method]["balanced_accuracy"]
            if len(vals):
                print(f" {vals.mean():12.4f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        # Mark best per dataset
        best_method = None
        best_ba = -1
        for method in method_order:
            vals = ddf[ddf["method"] == method]["balanced_accuracy"]
            if len(vals) and vals.mean() > best_ba:
                best_ba = vals.mean()
                best_method = method
        if best_method:
            print(f"  ← {best_method}", end="")
        print()

    # --- G-mean summary for paper Table 2 ---
    print("\n" + "=" * 80)
    print("Paper Table 2 — External Comparison (LR, medium noise, 50 pairs)")
    print("=" * 80)
    print(f"{'Method':<35} {'Source':<25} {'BA':>8} {'G-mean':>8} {'Min.Rec':>8}")
    print("-" * 80)
    sources = {
        "no_cleaning": "—",
        "smote": "[Chawla 2002]",
        "class_proportional": "[He & Garcia 2009]",
        "iw_smote": "[Zhang et al. 2022]",
        "sw_framework": "[X et al. 2022]†",
        "cwms_msbs": "—",
    }
    for method in method_order:
        mdf = df[df["method"] == method]
        if len(mdf) == 0:
            continue
        label = method_labels.get(method, method)
        src = sources.get(method, "—")
        ba = mdf["balanced_accuracy"].mean()
        gm = mdf["g_mean"].mean()
        rec = mdf["minority_recall"].mean()
        marker = "**" if method == "cwms_msbs" else ""
        print(f"{marker}{label:<35}{marker} {src:<25} {ba:7.4f}  {gm:7.4f}  {rec:7.4f}")


if __name__ == "__main__":
    main()
