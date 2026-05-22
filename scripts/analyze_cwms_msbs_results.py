"""Statistical analysis for CWMS+MSBS full sweep results.

Loads outputs/cwms-msbs-full-sweep.csv, runs paired Wilcoxon tests,
produces summary tables and per-model breakdown for the paper.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy is required. Install it with: pip install scipy", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "cwms-msbs-full-sweep.csv"


def main():
    df = pd.read_csv(OUTPUT_CSV)
    if 'error' in df.columns:
        n_err = df['error'].notna().sum()
        if n_err:
            print(f"WARNING: {n_err} error rows — excluding")
        df = df[df['error'].isna()].copy()
    df = df.reset_index(drop=True)
    # Deduplicate in case of corrupted append
    dedup_cols = ['dataset', 'model', 'seed', 'noise_protocol', 'method', 'budget', 'target_ratio']
    n_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep='last')
    if len(df) < n_before:
        print(f"WARNING: removed {n_before - len(df)} duplicate rows")

    # Primary slice: hidden_minority_medium only
    df_med = df[df['noise_protocol'] == 'hidden_minority_medium'].copy()

    protocols = sorted(df['noise_protocol'].unique())
    print(f"Data: {len(df)} rows, protocols={protocols}")
    print(f"Models: {sorted(df.model.unique())}, datasets: {sorted(df.dataset.unique())}")
    print(f"Methods: {sorted(df.method.unique())}")
    print()

    # --- Step 1: per-method/model counts ---
    print("=== PER-METHOD/MODEL COUNTS (medium) ===")
    print(df_med.groupby(['method', 'model']).size().unstack().to_string())
    print()

    # --- Step 2: Primary summary table ---
    print("=== PRIMARY SUMMARY TABLE (medium) ===")
    metrics = ['balanced_accuracy', 'minority_recall']
    summary = df_med.groupby('method')[metrics].agg(['mean', 'std']).round(4)
    summary.columns = ['BA_mean', 'BA_std', 'recall_mean', 'recall_std']
    print(summary.sort_values('BA_mean', ascending=False).to_string())
    print()

    # --- Step 3: Paired Wilcoxon ---
    print("=== PAIRED WILCOXON (cwms_msbs vs others, medium) ===")
    idx_cols = ['dataset', 'model', 'seed']
    pivot = df_med.pivot_table(index=idx_cols, columns='method', values='balanced_accuracy')
    target = 'cwms_msbs'
    comparisons = ['no_cleaning', 'class_proportional', 'msbs', 'cwms']

    results = []
    for comp in comparisons:
        sub = pivot[[target, comp]].dropna()
        wins = int((sub[target] > sub[comp]).sum())
        n = len(sub)
        delta = (sub[target] - sub[comp]).mean()
        try:
            _, p = stats.wilcoxon(sub[target], sub[comp])
        except Exception:
            p = float('nan')
        results.append({
            'comparison': f'{target} vs {comp}',
            'wins': wins, 'total': n, 'win_pct': 100*wins/n,
            'mean_delta': delta, 'p_value': p,
        })
        print(f"  {target} vs {comp:22s} wins={wins:>4}/{n:<4} ({100*wins/n:5.1f}%)  "
              f"delta={delta:+8.4f}  p={p:.2e}")
    print()

    # Per-model Wilcoxon
    print("=== PER-MODEL WILCOXON (cwms_msbs vs class_proportional) ===")
    for model in sorted(df_med.model.unique()):
        sub = pivot[pivot.index.get_level_values('model') == model]
        if len(sub) < 5:
            continue
        wins = int((sub[target] > sub['class_proportional']).sum())
        n = len(sub)
        delta = (sub[target] - sub['class_proportional']).mean()
        try:
            _, p = stats.wilcoxon(sub[target], sub['class_proportional'])
        except Exception:
            p = float('nan')
        print(f"  {model:20s} wins={wins:>3}/{n:<3} ({100*wins/n:5.1f}%)  delta={delta:+8.4f}  p={p:.2e}")
    print()

    # --- Step 4: Per-model breakdown ---
    print("=== PER-MODEL BREAKDOWN TABLE (medium) ===")
    model_tab = df_med[df_med['method'].isin(['cwms_msbs', 'class_proportional', 'no_cleaning'])].pivot_table(
        index='model', columns='method', values='balanced_accuracy', aggfunc='mean'
    ).round(4)
    model_tab['delta_vs_cp'] = (model_tab['cwms_msbs'] - model_tab['class_proportional']).round(4)
    model_tab['delta_vs_nc'] = (model_tab['cwms_msbs'] - model_tab['no_cleaning']).round(4)
    print(model_tab.to_string())
    print()

    # --- Step 5: Robustness across noise protocols ---
    if len(protocols) > 1:
        print("=== CROSS-PROTOCOL ROBUSTNESS ===")
        proto_methods = ['cwms_msbs', 'class_proportional', 'no_cleaning']
        proto_tab = df[df['method'].isin(proto_methods)].pivot_table(
            index='noise_protocol', columns='method', values='balanced_accuracy', aggfunc='mean'
        ).round(4)
        proto_tab['delta_vs_cp'] = proto_tab['cwms_msbs'] - proto_tab['class_proportional']
        print(proto_tab.to_string())
        print()

    # --- Step 6: MSBS volume check ---
    print("=== MSBS VOLUME CHECK (medium) ===")
    msbs_rows = df_med[df_med['method'].isin(['msbs', 'cwms_msbs'])]
    print(msbs_rows.groupby('method')['n_synthetic'].describe().round(1).to_string())
    zero_synth = (msbs_rows['n_synthetic'] == 0).sum()
    print(f"Zero-synthetic cases: {zero_synth}/{len(msbs_rows)}")
    print()

    # --- Additional: per-dataset cwms_msbs vs cp ---
    print("=== PER-DATASET cwms_msbs vs class_proportional (medium) ===")
    ds_tab = df_med[df_med['method'].isin(['cwms_msbs', 'class_proportional'])].pivot_table(
        index='dataset', columns='method', values='balanced_accuracy', aggfunc='mean'
    ).round(4)
    ds_tab['delta'] = (ds_tab['cwms_msbs'] - ds_tab['class_proportional']).round(4)
    print(ds_tab.to_string())
    print()

    # --- Recall analysis ---
    print("=== RECALL ANALYSIS (medium) ===")
    recall_tab = df_med[df_med['method'].isin(['cwms_msbs', 'class_proportional', 'no_cleaning'])].pivot_table(
        index='method', values='minority_recall', aggfunc=['mean', 'std']
    ).round(4)
    print(recall_tab.to_string())
    cp_recall = df_med[df_med.method == 'class_proportional']['minority_recall'].mean()
    cwms_recall = df_med[df_med.method == 'cwms_msbs']['minority_recall'].mean()
    print(f"cwms_msbs recall vs class_proportional: {cwms_recall - cp_recall:+.4f}")

    # Paired Wilcoxon on recall
    recall_pivot = df_med.pivot_table(index=['dataset', 'model', 'seed'],
                                       columns='method', values='minority_recall')
    for comp in ['class_proportional', 'no_cleaning', 'msbs', 'cwms']:
        sub = recall_pivot[['cwms_msbs', comp]].dropna()
        if len(sub) < 5:
            continue
        wins = int((sub['cwms_msbs'] > sub[comp]).sum())
        n = len(sub)
        delta = (sub['cwms_msbs'] - sub[comp]).mean()
        try:
            _, p = stats.wilcoxon(sub['cwms_msbs'], sub[comp])
        except Exception:
            p = float('nan')
        print(f"  recall cwms_msbs vs {comp:22s}: wins={wins}/{n} ({100*wins/n:.1f}%)  "
              f"delta={delta:+.4f}  p={p:.2e}")
    print()

    # --- cwms_msbs vs standalone methods ---
    print("=== cwms_msbs vs STANDALONE METHODS ===")
    for comp_method in ['msbs', 'cwms']:
        sub = pivot[[target, comp_method]].dropna()
        wins = int((sub[target] > sub[comp_method]).sum())
        n = len(sub)
        delta = (sub[target] - sub[comp_method]).mean()
        try:
            _, p = stats.wilcoxon(sub[target], sub[comp_method])
        except Exception:
            p = float('nan')
        print(f"  vs {comp_method}: wins={wins}/{n} ({100*wins/n:.1f}%) delta={delta:+.4f} p={p:.2e}")

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
