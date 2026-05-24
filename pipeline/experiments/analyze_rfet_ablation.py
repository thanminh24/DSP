"""Analyze RF/ET ablation sweep — decompose harm from cwms vs msbs components."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_full_benchmark import per_dataset_wilcoxon_stouffer

INPUT_CSV = PROJECT_ROOT / "outputs" / "rfet-ablation-sweep.csv"


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run run_rfet_ablation_sweep.py first.")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()

    print(f"Loaded {len(df)} valid rows from {INPUT_CSV}\n")

    method_order = ["no_cleaning", "class_proportional", "cwms", "msbs", "cwms_msbs"]

    print("=" * 90)
    print("RF/ET Ablation — ΔBA vs class_proportional (mean over 5 datasets × 10 seeds × 3 protocols = 150 pairs)")
    print("=" * 90)
    print(f"{'Model':<16} {'no_clean':>8} {'class_prop':>10} {'cwms_only':>10} {'msbs_only':>10} {'cwms_msbs':>10}")
    print("-" * 90)

    for model_name in ["random_forest", "extra_trees"]:
        mdf = df[df["model"] == model_name]
        vals = {}
        for m in method_order:
            v = mdf[mdf["method"] == m]["balanced_accuracy"]
            vals[m] = v.mean() if len(v) else float("nan")
        print(f"{model_name:<16} {vals['no_cleaning']:8.4f} {vals['class_proportional']:10.4f} "
              f"{vals['cwms']:10.4f} {vals['msbs']:10.4f} {vals['cwms_msbs']:10.4f}")

    print()
    print("=" * 90)
    print("ΔBA vs class_proportional (pp)")
    print("=" * 90)
    print(f"{'Model':<16} {'cwms_only':>10} {'msbs_only':>10} {'cwms_msbs':>10} {'Stouffer-Z(cwms_msbs)':>22} {'p-value':>10}")
    print("-" * 90)

    for model_name in ["random_forest", "extra_trees"]:
        mdf = df[df["model"] == model_name]
        cp_mean = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
        deltas = {}
        for m in ["cwms", "msbs", "cwms_msbs"]:
            v = mdf[mdf["method"] == m]["balanced_accuracy"].mean()
            deltas[m] = (v - cp_mean) * 100

        result = per_dataset_wilcoxon_stouffer(mdf, "cwms_msbs", "class_proportional")
        z_str = f"{result['stouffer_z']:.2f}"
        p_str = f"{result['stouffer_p']:.1e}"
        sig = f"{result['n_datasets_significant']}/{result['n_datasets_total']} sig"

        print(f"{model_name:<16} {deltas['cwms']:+9.2f}pp {deltas['msbs']:+9.2f}pp "
              f"{deltas['cwms_msbs']:+9.2f}pp {z_str:>22} {p_str:>10}  [{sig}]")

    print()
    print("=" * 90)
    print("Per-dataset breakdown (ΔBA cwms_msbs vs class_prop)")
    print("=" * 90)
    datasets = sorted(df["dataset"].unique())
    print(f"{'Dataset':<12} {'RF Δ(pp)':>10} {'ET Δ(pp)':>10}")
    print("-" * 35)
    for ds in datasets:
        row = []
        for model_name in ["random_forest", "extra_trees"]:
            mdf = df[(df["model"] == model_name) & (df["dataset"] == ds)]
            cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw = mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            delta = (cw - cp) * 100
            row.append(f"{delta:+9.2f}")
        print(f"{ds:<12} {row[0]:>10} {row[1]:>10}")

    print()
    print("Interpretation:")
    for model_name in ["random_forest", "extra_trees"]:
        mdf = df[df["model"] == model_name]
        cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
        d_cwms = (mdf[mdf["method"] == "cwms"]["balanced_accuracy"].mean() - cp) * 100
        d_msbs = (mdf[mdf["method"] == "msbs"]["balanced_accuracy"].mean() - cp) * 100
        d_full = (mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean() - cp) * 100
        harm_source = "cwms (suppression)" if abs(d_cwms) > abs(d_msbs) else "msbs (synthesis)"
        print(f"  {model_name}: primary harm from {harm_source} (Δcwms={d_cwms:+.2f}pp, Δmsbs={d_msbs:+.2f}pp, Δfull={d_full:+.2f}pp)")


if __name__ == "__main__":
    main()
