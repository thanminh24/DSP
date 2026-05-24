"""Analyze IR=0.30 sweep vs IR=0.15 baseline — Table 3 in paper."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_full_benchmark import per_dataset_wilcoxon_stouffer

DEFAULT_IR015 = PROJECT_ROOT / "outputs" / "full-benchmark-solution-v2.csv"
DEFAULT_IR030 = PROJECT_ROOT / "outputs" / "full-benchmark-ir030-solution.csv"

CWMS_MODELS = ["lr", "svm", "random_forest", "extra_trees", "hgb", "lightgbm", "catboost"]


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    return df[df["balanced_accuracy"].notna()].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir015", default=str(DEFAULT_IR015))
    parser.add_argument("--ir030", default=str(DEFAULT_IR030))
    args = parser.parse_args()

    df015 = load_df(Path(args.ir015))
    df030 = load_df(Path(args.ir030))

    print(f"IR=0.15: {len(df015)} rows from {args.ir015}")
    print(f"IR=0.30: {len(df030)} rows from {args.ir030}\n")

    # --- Table 3: ΔBA cwms_msbs vs class_prop at IR=0.15 and IR=0.30 ---
    print("=" * 95)
    print("Table 3 — NoiSyn ΔBA vs class_proportional at different imbalance ratios")
    print("=" * 95)
    print(f"{'Model':<16} {'Δ IR=0.15(pp)':>14} {'Z IR=0.15':>10} {'sig_ds 0.15':>12} {'Δ IR=0.30(pp)':>14} {'Z IR=0.30':>10} {'sig_ds 0.30':>12}")
    print("-" * 95)

    for model_name in CWMS_MODELS:
        row015 = _model_stats(df015, model_name)
        row030 = _model_stats(df030, model_name)
        if row015 is None and row030 is None:
            continue
        r015 = row015 or {"delta": float("nan"), "z": float("nan"), "sig": "N/A"}
        r030 = row030 or {"delta": float("nan"), "z": float("nan"), "sig": "N/A"}
        z015_str = f"{r015['z']:.2f}" if not np.isnan(r015['z']) else "N/A"
        z030_str = f"{r030['z']:.2f}" if not np.isnan(r030['z']) else "N/A"
        delta015 = f"{r015['delta']:+.2f}" if not np.isnan(r015['delta']) else "N/A"
        delta030 = f"{r030['delta']:+.2f}" if not np.isnan(r030['delta']) else "N/A"
        print(f"{model_name:<16} {delta015:>13}pp {z015_str:>10} {r015['sig']:>12} {delta030:>13}pp {z030_str:>10} {r030['sig']:>12}")

    # --- Per-protocol breakdown at IR=0.30 ---
    print("\n" + "=" * 80)
    print("Table 3b — IR=0.30 breakdown by protocol (cwms_msbs ΔBA vs class_prop)")
    print("=" * 80)
    print(f"{'Model':<16} {'low':>8} {'medium':>8} {'high':>8}")
    print("-" * 50)
    for model_name in CWMS_MODELS:
        mdf = df030[df030["model"] == model_name]
        if len(mdf) == 0:
            continue
        deltas = []
        for proto in ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]:
            pdf = mdf[mdf["noise_protocol"] == proto]
            cp = pdf[pdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw = pdf[pdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            deltas.append(f"{(cw - cp)*100:+.2f}" if not np.isnan(cw) else "N/A")
        print(f"{model_name:<16} {deltas[0]:>8} {deltas[1]:>8} {deltas[2]:>8}")


def _model_stats(df: pd.DataFrame, model_name: str):
    mdf = df[df["model"] == model_name]
    cw = mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"]
    cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"]
    if len(cw) == 0 or len(cp) == 0:
        return None
    delta = (cw.mean() - cp.mean()) * 100
    res = per_dataset_wilcoxon_stouffer(mdf, "cwms_msbs", "class_proportional")
    return {
        "delta": delta,
        "z": res["stouffer_z"],
        "p": res["stouffer_p"],
        "sig": f"{res['n_datasets_significant']}/{res['n_datasets_total']}",
    }


if __name__ == "__main__":
    main()
