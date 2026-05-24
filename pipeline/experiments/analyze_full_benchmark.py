"""Analyze Run A: full-benchmark-solution.csv → paper Table 1 + supplementary tables."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "docs" / "experiments" / "raw" / "full-benchmark-solution-v2.csv"


def _g_mean(df):
    return np.sqrt(df["minority_recall"].clip(lower=0) * df["majority_recall"].clip(lower=0))


def per_dataset_wilcoxon_stouffer(df, method_a, method_b, metric="balanced_accuracy"):
    """Per-dataset Wilcoxon + Stouffer combination. Fully defensible.

    Each dataset contributes one independent z-score via one-sided Wilcoxon over
    (seed × protocol) pairs (30 per dataset at 10 seeds × 3 protocols).
    Stouffer's method combines these into a single z-statistic.
    """
    datasets = sorted(df["dataset"].unique())
    z_scores, pvals, deltas, ds_used = [], [], [], []
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        a = sub[sub["method"] == method_a].set_index(["seed", "noise_protocol"])[metric]
        b = sub[sub["method"] == method_b].set_index(["seed", "noise_protocol"])[metric]
        shared = a.index.intersection(b.index)
        diff = (a - b)[shared]
        if len(diff) < 5:
            continue
        try:
            _, p = stats.wilcoxon(diff, alternative="greater")
            # norm.ppf(1 - p): maps one-sided p to z-score (p=0.05 → z=1.645, p=1.0 → -inf)
            # Clip at -8 so -inf doesn't propagate into Stouffer sum for all-negative datasets
            z = max(float(stats.norm.ppf(1 - p)), -8.0)
        except ValueError:
            p, z = 1.0, -8.0
        pvals.append(p)
        z_scores.append(z)
        deltas.append(float(diff.mean()))
        ds_used.append(ds)
    z_combined = sum(z_scores) / (len(z_scores) ** 0.5) if z_scores else float("nan")
    p_combined = float(1 - stats.norm.cdf(z_combined)) if z_scores else float("nan")
    return {
        "datasets": ds_used,
        "per_dataset_pvals": dict(zip(ds_used, pvals)),
        "per_dataset_deltas": dict(zip(ds_used, deltas)),
        "stouffer_z": z_combined,
        "stouffer_p": p_combined,
        "n_datasets_significant": sum(p < 0.05 for p in pvals),
        "n_datasets_total": len(pvals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Path to CSV (default: docs/experiments/raw/full-benchmark-solution-v2.csv)")
    args = parser.parse_args()

    output_csv = Path(args.input) if args.input else DEFAULT_CSV
    if not output_csv.exists():
        print(f"ERROR: {output_csv} not found. Run benchmark sweep first.", flush=True)
        sys.exit(1)
    df = pd.read_csv(output_csv)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df = df[df["balanced_accuracy"].notna()].copy()
    df["g_mean"] = _g_mean(df)
    has_pr_auc = "pr_auc" in df.columns and df["pr_auc"].notna().any()

    print(f"Loaded {len(df)} valid rows from {output_csv}\n")

    # --- Table 1: Per-model benchmark ---
    cwms_models = [m for m in df["model"].unique()
                   if m not in ("xgboost", "calibrated_lr")]
    baseline_models = ["xgboost", "calibrated_lr"]

    n_datasets = df["dataset"].nunique()
    n_pairs_per_row = n_datasets * 10 * 3  # datasets × seeds × protocols
    hdr_suffix = " {'PR-AUC':>8}" if has_pr_auc else ""
    width = 110 if has_pr_auc else 100
    print("=" * width)
    print(f"Table 1 — Internal Benchmark (mean BA, {n_pairs_per_row} pairs per row = {n_datasets} datasets × 10 seeds × 3 protocols)")
    print("=" * width)
    pr_hdr = f" {'PR-AUC':>8}" if has_pr_auc else ""
    print(f"{'Model':<16} {'no_clean':>8} {'class_prop':>10} {'smote':>8} {'cwms_msbs':>10} {'ΔBA(pp)':>8} {'Stouffer-Z':>11} {'p-value':>10} {'sig_ds':>8} {'G-mean':>8}{pr_hdr}")
    print("-" * width)

    for model_name in cwms_models:
        mdf = df[df["model"] == model_name]
        nc = mdf[mdf["method"] == "no_cleaning"]["balanced_accuracy"]
        cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"]
        sm = mdf[mdf["method"] == "smote"]["balanced_accuracy"]
        cw = mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"]

        if len(cw) == 0 or len(cp) == 0:
            continue

        delta = (cw.mean() - cp.mean()) * 100
        result = per_dataset_wilcoxon_stouffer(mdf, "cwms_msbs", "class_proportional")
        stouffer_z = result["stouffer_z"]
        stouffer_p = result["stouffer_p"]
        sig_ds = f"{result['n_datasets_significant']}/{result['n_datasets_total']}"

        nc_mean = nc.mean() if len(nc) else float("nan")
        cp_mean = cp.mean() if len(cp) else float("nan")
        sm_mean = sm.mean() if len(sm) else float("nan")
        cw_mean = cw.mean() if len(cw) else float("nan")
        gm = mdf[mdf["method"] == "cwms_msbs"]["g_mean"].mean() if len(cw) else float("nan")

        p_str = f"{stouffer_p:.1e}" if not np.isnan(stouffer_p) else "N/A"
        z_str = f"{stouffer_z:.2f}" if not np.isnan(stouffer_z) else "N/A"
        pr_str = ""
        if has_pr_auc:
            pr_vals = mdf[mdf["method"] == "cwms_msbs"]["pr_auc"]
            pr_mean = pr_vals.mean() if len(pr_vals) else float("nan")
            pr_str = f" {pr_mean:8.4f}" if not np.isnan(pr_mean) else f" {'N/A':>8}"
        print(f"{model_name:<16} {nc_mean:8.4f} {cp_mean:10.4f} {sm_mean:8.4f} {cw_mean:10.4f} {delta:+7.2f} {z_str:>11} {p_str:>10} {sig_ds:>8} {gm:8.4f}{pr_str}")

    # Baseline-only models (footnote)
    print()
    for model_name in baseline_models:
        mdf = df[df["model"] == model_name]
        nc = mdf[mdf["method"] == "no_cleaning"]["balanced_accuracy"]
        cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"]
        sm = mdf[mdf["method"] == "smote"]["balanced_accuracy"]
        if len(nc) == 0 and len(cp) == 0:
            continue
        print(f"{model_name:<16} {nc.mean():8.4f} {cp.mean():10.4f} {sm.mean():8.4f} {'(baselines only)':>10}")

    # Aggregate row (CWMS-compatible average)
    cwms_df = df[df["model"].isin(cwms_models)]
    nc_agg = cwms_df[cwms_df["method"] == "no_cleaning"]["balanced_accuracy"].mean()
    cp_agg = cwms_df[cwms_df["method"] == "class_proportional"]["balanced_accuracy"].mean()
    cw_agg = cwms_df[cwms_df["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
    print("-" * 90)
    print(f"{'Avg (CWMS-comp)':<16} {nc_agg:8.4f} {cp_agg:10.4f} {'':>8} {cw_agg:10.4f}")

    # --- Table 1b: Shuffled ablation ---
    print("\n" + "=" * 80)
    print("Table 1b — Shuffled Ablation: cwms_msbs vs cwms_msbs_shuffled")
    print("=" * 80)
    print(f"{'Model':<16} {'cwms_msbs':>10} {'shuffled':>10} {'ΔBA(pp)':>8} {'Stouffer-Z':>11} {'p-value':>10}")
    print("-" * 80)

    for model_name in cwms_models:
        mdf = df[df["model"] == model_name]
        if mdf[mdf["method"] == "cwms_msbs_shuffled"].empty:
            continue
        result = per_dataset_wilcoxon_stouffer(mdf, "cwms_msbs", "cwms_msbs_shuffled")
        cw_mean = mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
        sh_mean = mdf[mdf["method"] == "cwms_msbs_shuffled"]["balanced_accuracy"].mean()
        delta = (cw_mean - sh_mean) * 100
        z_str = f"{result['stouffer_z']:.2f}" if not np.isnan(result["stouffer_z"]) else "N/A"
        p_str = f"{result['stouffer_p']:.1e}" if not np.isnan(result["stouffer_p"]) else "N/A"
        print(f"{model_name:<16} {cw_mean:10.4f} {sh_mean:10.4f} {delta:+7.2f} {z_str:>11} {p_str:>10}")

    # --- Supplementary: Best method per model ---
    print("\n" + "=" * 70)
    print("Table S1 — Best method per model (mean BA)")
    print("=" * 70)
    methods_to_compare = ["msbs", "cwms", "cwms_msbs"]
    for model_name in cwms_models:
        mdf = df[df["model"] == model_name]
        best_method, best_ba = None, -1
        for method in methods_to_compare:
            vals = mdf[mdf["method"] == method]["balanced_accuracy"]
            if len(vals) and vals.mean() > best_ba:
                best_ba = vals.mean()
                best_method = method
        print(f"  {model_name:<16}: {best_method:<15} {best_ba:.4f}")

    # --- Supplementary: Protocol breakdown ---
    print("\n" + "=" * 70)
    print("Table S2 — Protocol breakdown: cwms_msbs ΔBA vs class_proportional")
    print("=" * 70)
    print(f"{'Model':<16} {'low':>8} {'medium':>8} {'high':>8}")
    print("-" * 70)
    for model_name in cwms_models:
        mdf = df[df["model"] == model_name]
        deltas = []
        for proto in ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]:
            pdf = mdf[mdf["noise_protocol"] == proto]
            cp_v = pdf[pdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw_v = pdf[pdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            deltas.append(f"{(cw_v - cp_v)*100:+.2f}" if not np.isnan(cw_v) else "N/A")
        print(f"{model_name:<16} {deltas[0]:>8} {deltas[1]:>8} {deltas[2]:>8}")


if __name__ == "__main__":
    main()
