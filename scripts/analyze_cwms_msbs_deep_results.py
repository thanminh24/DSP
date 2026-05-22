"""Statistical analysis for CWMS+MSBS v2 deep sweep.

Loads outputs/cwms-msbs-deep-sweep.csv, runs paired Wilcoxon tests on all metrics,
produces summary tables, per-model breakdown, cross-protocol robustness, and
boosting family aggregate for the paper.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "cwms-msbs-deep-sweep.csv"


def main():
    df = pd.read_csv(OUTPUT_CSV)
    # Exclude error/NAN rows
    if 'error' in df.columns:
        n_err = df['error'].notna().sum()
        if n_err:
            print(f"WARNING: {n_err} error rows — excluding")
        df = df[df['error'].isna()]
    df = df[pd.to_numeric(df["balanced_accuracy"], errors="coerce").notna()].copy()
    df = df.reset_index(drop=True)

    # Dedup
    dedup_cols = ['dataset', 'model', 'seed', 'noise_protocol', 'method', 'budget', 'target_ratio']
    n_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep='last')
    if len(df) < n_before:
        print(f"WARNING: removed {n_before - len(df)} duplicate rows")

    protocols = sorted(df['noise_protocol'].unique())
    df_med = df[df['noise_protocol'] == 'hidden_minority_medium'].copy()

    print(f"Data: {len(df)} rows, protocols={protocols}")
    print(f"Models: {sorted(df.model.unique())}")
    print(f"Methods per model:")
    for m in sorted(df.model.unique()):
        print(f"  {m}: {sorted(df[df.model==m].method.unique())}")
    print()

    # Available metrics
    metric_cols = ['balanced_accuracy', 'accuracy', 'macro_f1', 'weighted_f1',
                   'minority_recall', 'minority_precision', 'majority_recall']

    # === Step 1: Primary summary table (medium) ===
    print("=== PRIMARY SUMMARY TABLE (medium, all metrics) ===")
    summary = df_med.groupby('method')[metric_cols].agg(['mean', 'std']).round(4)
    for col in metric_cols:
        m = summary[col]
        print(f"\n{col}:")
        for method in m.sort_values('mean', ascending=False).index:
            print(f"  {method:25s}  {m.loc[method, 'mean']:.4f} ± {m.loc[method, 'std']:.4f}")
    print()

    # === Step 2: Overall delta table (medium) ===
    print("=== cwms_msbs vs class_proportional: DELTA TABLE (medium) ===")
    idx_cols = ['dataset', 'model', 'seed']
    pivot = df_med.pivot_table(index=idx_cols, columns='method',
                                values=metric_cols + ['n_synthetic'])
    for metric in metric_cols:
        if (target := ('cwms_msbs', metric)) not in pivot.columns:
            continue
        sub = pivot[target] - pivot[('class_proportional', metric)]
        n = len(sub.dropna())
        mean_delta = sub.mean()
        wins = int((sub > 0).sum())
        try:
            target_vals = df_med[df_med.method == 'cwms_msbs'].set_index(idx_cols)[metric]
            cp_vals = df_med[df_med.method == 'class_proportional'].set_index(idx_cols)[metric]
            common = target_vals.index.intersection(cp_vals.index)
            _, p = stats.wilcoxon(target_vals.loc[common], cp_vals.loc[common])
        except Exception:
            p = float('nan')
        print(f"  {metric:25s}  delta={mean_delta:+8.4f}  wins={wins:>4}/{n:<4} ({100*wins/n:5.1f}%)  p={p:.2e}")
    print()

    # === Step 3: Per-model BA breakdown ===
    print("=== PER-MODEL BA BREAKDOWN (medium) ===")
    model_metrics = df_med[df_med['method'].isin(['cwms_msbs', 'class_proportional', 'no_cleaning'])]
    ba_pivot = model_metrics.pivot_table(index='model', columns='method',
                                          values='balanced_accuracy', aggfunc='mean').round(4)
    if 'cwms_msbs' in ba_pivot.columns and 'class_proportional' in ba_pivot.columns:
        ba_pivot['delta_vs_cp'] = (ba_pivot['cwms_msbs'] - ba_pivot['class_proportional']).round(4)
    if 'no_cleaning' in ba_pivot.columns and 'cwms_msbs' in ba_pivot.columns:
        ba_pivot['delta_vs_nc'] = (ba_pivot['cwms_msbs'] - ba_pivot['no_cleaning']).round(4)
    print(ba_pivot.to_string())
    print()

    # Per-model Wilcoxon + win rate
    print("=== PER-MODEL WILCOXON (cwms_msbs vs class_proportional, medium) ===")
    ba_wide = df_med.pivot_table(index=idx_cols, columns='method', values='balanced_accuracy')
    for model in sorted(df_med.model.unique()):
        sub = ba_wide[ba_wide.index.get_level_values('model') == model]
        if 'cwms_msbs' not in sub.columns or 'class_proportional' not in sub.columns:
            print(f"  {model:20s}  SKIPPED (no cwms_msbs for this model)")
            continue
        s = sub[['cwms_msbs', 'class_proportional']].dropna()
        if len(s) < 5:
            continue
        wins = int((s['cwms_msbs'] > s['class_proportional']).sum())
        n = len(s)
        delta = (s['cwms_msbs'] - s['class_proportional']).mean()
        try:
            _, p = stats.wilcoxon(s['cwms_msbs'], s['class_proportional'])
        except Exception:
            p = float('nan')
        print(f"  {model:20s}  wins={wins:>3}/{n:<3} ({100*wins/n:5.1f}%)  delta={delta:+8.4f}  p={p:.2e}")
    print()

    # === Step 4: Boosting family aggregate ===
    boosting_models = ['hgb', 'xgboost', 'lightgbm', 'catboost']
    boost_data = df_med[df_med.model.isin(boosting_models)]
    if len(boost_data) > 0:
        print("=== BOOSTING FAMILY AGGREGATE (hgb+xgb+lgbm+cat) ===")
        for m in ['cwms_msbs', 'class_proportional']:
            vals = boost_data[boost_data.method == m]['balanced_accuracy']
            print(f"  {m}: BA={vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")
        delta = (boost_data[boost_data.method == 'cwms_msbs']['balanced_accuracy'].mean() -
                 boost_data[boost_data.method == 'class_proportional']['balanced_accuracy'].mean())
        print(f"  delta: {delta:+.4f}")
        print()

    # === Step 5: LR family ===
    lr_data = df_med[df_med.model == 'lr']
    if len(lr_data) > 0:
        print("=== LR FAMILY ===")
        for m in ['cwms_msbs', 'class_proportional', 'no_cleaning']:
            vals = lr_data[lr_data.method == m]['balanced_accuracy']
            print(f"  {m}: BA={vals.mean():.4f} ± {vals.std():.4f}")
        print()

    # === Step 6: Cross-protocol robustness ===
    if len(protocols) > 1:
        print("=== CROSS-PROTOCOL ROBUSTNESS (cwms_msbs vs class_proportional) ===")
        proto_methods = ['cwms_msbs', 'class_proportional']
        pt = df[df['method'].isin(proto_methods)].pivot_table(
            index='noise_protocol', columns='method',
            values='balanced_accuracy', aggfunc='mean'
        ).round(4)
        pt['delta'] = pt['cwms_msbs'] - pt['class_proportional']
        print(pt.to_string())
        print()

    # === Step 7: Per-dataset breakdown ===
    print("=== PER-DATASET cwms_msbs vs class_proportional (medium) ===")
    ds_tab = df_med[df_med['method'].isin(['cwms_msbs', 'class_proportional'])].pivot_table(
        index='dataset', columns='method', values='balanced_accuracy', aggfunc='mean'
    ).round(4)
    ds_tab['delta'] = (ds_tab['cwms_msbs'] - ds_tab['class_proportional']).round(4)
    print(ds_tab.to_string())
    print()

    # === Step 8: MSBS volume check ===
    print("=== MSBS VOLUME CHECK (medium) ===")
    msbs_rows = df_med[df_med['method'].isin(['msbs', 'cwms_msbs'])]
    print(msbs_rows.groupby('method')['n_synthetic'].describe().round(1).to_string())
    zero_synth = int((msbs_rows['n_synthetic'] == 0).sum())
    print(f"Zero-synthetic cases: {zero_synth}/{len(msbs_rows)}")
    print()

    # === Step 9: Multi-metric per-model table ===
    print("=== MULTI-METRIC PER-MODEL (cwms_msbs vs class_proportional, medium) ===")
    multi_metrics = ['balanced_accuracy', 'minority_recall', 'minority_precision', 'weighted_f1']
    for model in sorted(df_med.model.unique()):
        sub = df_med[df_med.model == model]
        if 'cwms_msbs' not in sub['method'].values:
            continue
        deltas = {}
        for metric in multi_metrics:
            cwms_val = sub[sub.method == 'cwms_msbs'][metric].mean()
            cp_val = sub[sub.method == 'class_proportional'][metric].mean()
            deltas[metric] = cwms_val - cp_val
        delta_str = "  ".join(f"{m}: {d:+.4f}" for m, d in deltas.items())
        print(f"  {model:20s}  {delta_str}")

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
