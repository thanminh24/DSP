"""Analyze failure-mode sweep — NoiSyn under symmetric and reverse-asymmetric noise."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_full_benchmark import per_dataset_wilcoxon_stouffer

INPUT_CSV = PROJECT_ROOT / "outputs" / "failure-mode-sweep.csv"


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run run_failure_mode_sweep.py first.")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()

    print(f"Loaded {len(df)} valid rows from {INPUT_CSV}\n")

    protocols = ["symmetric", "reverse_asymmetric"]
    method_order = ["no_cleaning", "class_proportional", "smote", "cwms_msbs"]

    print("=" * 80)
    print("Failure Mode Analysis — NoiSyn under non-designed noise protocols (LR)")
    print("=" * 80)
    print(f"{'Protocol':<22} {'no_clean':>8} {'class_prop':>10} {'smote':>8} {'cwms_msbs':>10} {'Δ(vs cp)':>9} {'Stouffer-Z':>11} {'p-value':>10}")
    print("-" * 80)

    for proto in protocols:
        pdf = df[df["noise_protocol"] == proto]
        if len(pdf) == 0:
            continue
        vals = {}
        for m in method_order:
            v = pdf[pdf["method"] == m]["balanced_accuracy"]
            vals[m] = v.mean() if len(v) else float("nan")

        delta = (vals["cwms_msbs"] - vals["class_proportional"]) * 100
        result = per_dataset_wilcoxon_stouffer(pdf, "cwms_msbs", "class_proportional")
        z_str = f"{result['stouffer_z']:.2f}" if not np.isnan(result["stouffer_z"]) else "N/A"
        p_str = f"{result['stouffer_p']:.1e}" if not np.isnan(result["stouffer_p"]) else "N/A"

        print(f"{proto:<22} {vals['no_cleaning']:8.4f} {vals['class_proportional']:10.4f} "
              f"{vals['smote']:8.4f} {vals['cwms_msbs']:10.4f} {delta:+8.2f}pp {z_str:>11} {p_str:>10}")

    print()
    print("=" * 80)
    print("Reference: hidden_minority protocols (from full-benchmark-solution.csv)")
    print("=" * 80)
    ref_csv = PROJECT_ROOT / "outputs" / "full-benchmark-solution.csv"
    if ref_csv.exists():
        df_ref = pd.read_csv(ref_csv)
        df_ref = df_ref[df_ref["model"] == "lr"]
        df_ref = df_ref[df_ref["balanced_accuracy"].notna()]
        for proto in ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]:
            pdf = df_ref[df_ref["noise_protocol"] == proto]
            if len(pdf) == 0:
                continue
            cp = pdf[pdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw = pdf[pdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            delta = (cw - cp) * 100
            print(f"  {proto:<28}: ΔBA={delta:+.2f}pp (NoiSyn vs class_prop, LR)")

    print()
    print("Interpretation:")
    for proto in protocols:
        pdf = df[df["noise_protocol"] == proto]
        if len(pdf) == 0:
            continue
        cp = pdf[pdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
        cw = pdf[pdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
        delta = (cw - cp) * 100
        if proto == "symmetric":
            expected = "≈0 or slightly negative"
        else:  # reverse_asymmetric
            expected = "negative (OOF scores point wrong direction)"
        sign = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"
        print(f"  {proto}: Δ={delta:+.2f}pp ({sign}, expected {expected})")


if __name__ == "__main__":
    main()
