"""Analyze expanded competitor head-to-head: LR+SVM+HGB × 3 protocols × 15 datasets."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_full_benchmark import per_dataset_wilcoxon_stouffer

INPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead-expanded.csv"

METHOD_ORDER = ["no_cleaning", "class_proportional", "smote", "iw_smote", "sw_framework", "cwms_msbs"]
METHOD_LABELS = {
    "no_cleaning":       "no_cleaning",
    "smote":             "SMOTE [Chawla 2002]",
    "class_proportional":"class_proportional",
    "iw_smote":          "IW-SMOTE [Zhang et al. 2022]",
    "sw_framework":      "SW-approx [X et al. 2022]",
    "cwms_msbs":         "NoiSyn (ours)",
}
SOURCES = {
    "no_cleaning":        "—",
    "smote":              "[Chawla 2002]",
    "class_proportional": "—",
    "iw_smote":           "[Zhang et al. 2022]",
    "sw_framework":       "[X et al. 2022]†",
    "cwms_msbs":          "—",
}
PROTOCOLS = ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]
MODELS = ["lr", "svm", "hgb"]


def _g_mean(df):
    return np.sqrt(df["minority_recall"].clip(lower=0) * df["majority_recall"].clip(lower=0))


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run run_expanded_competitor_headtohead.py first.")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()
    df["g_mean"] = _g_mean(df)

    print(f"Loaded {len(df)} valid rows from {INPUT_CSV}\n")

    # --- Table 2: Aggregate per model (all protocols combined) ---
    print("=" * 105)
    print("Table 2 — External Comparison (mean BA, all 3 protocols combined)")
    print(f"  n pairs per cell = {df['dataset'].nunique()} datasets × 10 seeds × 3 protocols = {df['dataset'].nunique() * 10 * 3}")
    print("=" * 105)
    print(f"{'Method':<30} {'LR BA':>8} {'SVM BA':>8} {'HGB BA':>8} {'LR Stouff-Z':>12} {'LR p':>10} {'LR sig_ds':>10}")
    print("-" * 105)

    for method in METHOD_ORDER:
        row_parts = [f"{METHOD_LABELS.get(method, method):<30}"]
        for model_name in MODELS:
            mdf = df[(df["method"] == method) & (df["model"] == model_name)]
            ba = mdf["balanced_accuracy"].mean() if len(mdf) else float("nan")
            row_parts.append(f"{ba:8.4f}" if not np.isnan(ba) else f"{'N/A':>8}")
        # Stouffer for LR vs class_proportional (only for cwms_msbs)
        if method == "cwms_msbs":
            lr_df = df[df["model"] == "lr"]
            res = per_dataset_wilcoxon_stouffer(lr_df, "cwms_msbs", "class_proportional")
            z_str = f"{res['stouffer_z']:.2f}" if not np.isnan(res["stouffer_z"]) else "N/A"
            p_str = f"{res['stouffer_p']:.1e}" if not np.isnan(res["stouffer_p"]) else "N/A"
            sig_str = f"{res['n_datasets_significant']}/{res['n_datasets_total']}"
            row_parts += [f"{z_str:>12}", f"{p_str:>10}", f"{sig_str:>10}"]
        else:
            row_parts += [f"{'':>12}", f"{'':>10}", f"{'':>10}"]
        print(" ".join(row_parts))

    # --- Table 2b: Per-model × per-protocol BA ---
    print("\n" + "=" * 90)
    print("Table 2b — Mean BA by model × protocol (cwms_msbs vs class_proportional)")
    print("=" * 90)
    print(f"{'Model':<8} {'Protocol':<26} {'class_prop':>11} {'cwms_msbs':>10} {'Δ(pp)':>7} {'Stouffer-Z':>11} {'p':>10} {'sig_ds':>8}")
    print("-" * 90)

    for model_name in MODELS:
        mdf = df[df["model"] == model_name]
        for proto in PROTOCOLS:
            pdf = mdf[mdf["noise_protocol"] == proto]
            cp_mean = pdf[pdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw_mean = pdf[pdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            delta = (cw_mean - cp_mean) * 100
            res = per_dataset_wilcoxon_stouffer(pdf, "cwms_msbs", "class_proportional")
            z_str = f"{res['stouffer_z']:.2f}" if not np.isnan(res["stouffer_z"]) else "N/A"
            p_str = f"{res['stouffer_p']:.1e}" if not np.isnan(res["stouffer_p"]) else "N/A"
            sig_str = f"{res['n_datasets_significant']}/{res['n_datasets_total']}"
            print(f"{model_name:<8} {proto:<26} {cp_mean:11.4f} {cw_mean:10.4f} {delta:+6.2f}pp {z_str:>11} {p_str:>10} {sig_str:>8}")

    # --- Table 2c: Pairwise comparison vs each competitor (LR, all protocols) ---
    print("\n" + "=" * 95)
    print("Table 2c — NoiSyn vs each competitor: LR, all protocols (Stouffer combination)")
    print("=" * 95)
    lr_df = df[df["model"] == "lr"]
    print(f"{'Competitor':<35} {'mean_delta(pp)':>15} {'Stouffer-Z':>12} {'p-value':>10} {'sig_ds':>8}")
    print("-" * 95)
    for method in METHOD_ORDER:
        if method == "cwms_msbs":
            continue
        mdf_comp = lr_df[lr_df["method"].isin([method, "cwms_msbs"])]
        if len(mdf_comp) == 0:
            continue
        res = per_dataset_wilcoxon_stouffer(mdf_comp, "cwms_msbs", method)
        delta_mean = np.mean(list(res["per_dataset_deltas"].values())) * 100 if res["per_dataset_deltas"] else float("nan")
        z_str = f"{res['stouffer_z']:.2f}" if not np.isnan(res["stouffer_z"]) else "N/A"
        p_str = f"{res['stouffer_p']:.1e}" if not np.isnan(res["stouffer_p"]) else "N/A"
        sig_str = f"{res['n_datasets_significant']}/{res['n_datasets_total']}"
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<35} {delta_mean:+14.2f}pp {z_str:>12} {p_str:>10} {sig_str:>8}")

    # --- Per-dataset breakdown: NoiSyn vs class_prop, LR, medium protocol ---
    print("\n" + "=" * 80)
    print("Per-dataset BA: NoiSyn vs class_prop (LR, medium, mean over 10 seeds)")
    print("=" * 80)
    medium_lr = df[(df["model"] == "lr") & (df["noise_protocol"] == "hidden_minority_medium")]
    datasets = sorted(df["dataset"].unique())
    print(f"{'Dataset':<14} {'class_prop':>11} {'cwms_msbs':>10} {'Δ(pp)':>8}")
    print("-" * 47)
    for ds in datasets:
        ddf = medium_lr[medium_lr["dataset"] == ds]
        cp = ddf[ddf["method"] == "class_proportional"]["balanced_accuracy"].mean()
        cw = ddf[ddf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
        delta = (cw - cp) * 100
        marker = " ←" if delta > 0 else ""
        print(f"{ds:<14} {cp:11.4f} {cw:10.4f} {delta:+7.2f}pp{marker}")


if __name__ == "__main__":
    main()
