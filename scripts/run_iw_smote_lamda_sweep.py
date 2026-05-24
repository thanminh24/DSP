"""Sweep C — M3: IW-SMOTE lamda sensitivity.

Tests lamda=10,20,30,50,100. Current default is 30; original paper default is 100.
If BA(lamda=100) > BA(lamda=30) by >0.5pp on average, Phase 5 must use lamda=100.
Otherwise keep lamda=30 and add footnote.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import NOISE_PROTOCOLS, QUICK_SEEDS
from scripts._sweep_utils import check_gpu, load_completed
from pipeline.data.loaders import induce_imbalance, inject_noise, load_dataset
from pipeline.data.encoding import encode_train_test
from pipeline.evaluation.augment_metrics import evaluate_augmented
from pipeline.models.factories import make_model_factory
from pipeline.baselines.iw_smote import iw_smote

LAMDA_VALUES = [10, 20, 30, 50, 100]
PROTOCOL = "hidden_minority_medium"
MODEL = "lr"
DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = QUICK_SEEDS
BUDGET = 0.10
RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "iw-lamda-sweep.csv"

# 5 lamda × 5 datasets × 10 seeds = 250 rows


def _load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(zip(df["dataset"], df["seed"].astype(int), df["lamda"].astype(int)))
    except Exception:
        return set()


def run_one(dataset, seed, lamda, use_gpu=False):
    rng = np.random.default_rng(seed)
    X_raw, y_raw, cat_cols, _ = load_dataset(dataset)
    vals, counts = np.unique(y_raw, return_counts=True)
    minority_label = vals[np.argmin(counts)].item()
    majority_label = 1 if minority_label == 0 else 0

    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=seed,
    )
    X_tr_df = X_tr_df.reset_index(drop=True)
    X_te_df = X_te_df.reset_index(drop=True)
    X_tr_df, y_tr = induce_imbalance(X_tr_df, y_tr, minority_label=minority_label,
                                      target_ratio=RATIO, rng=rng)
    X_tr, X_te, cat_indices = encode_train_test(X_tr_df, X_te_df, cat_cols)
    mn, mj = NOISE_PROTOCOLS[PROTOCOL]
    y_noisy, _ = inject_noise(y_tr, minority_label, mn, mj, rng)

    factory = make_model_factory(MODEL, seed, cat_indices, balanced=False, use_gpu=use_gpu)
    budget_count = max(1, int(round(BUDGET * len(y_noisy))))
    X_aug, y_aug, n_synth = iw_smote(
        X_tr, y_noisy, minority_label, majority_label,
        budget_count=budget_count, lamda=lamda, seed=seed,
    )
    result = evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, minority_label,
                                 n_synthetic=n_synth, relabel_correctness=float("nan"))
    result.update({
        "dataset": dataset, "model": MODEL, "seed": seed,
        "noise_protocol": PROTOCOL, "lamda": lamda,
        "budget": BUDGET, "target_ratio": RATIO,
    })
    return result


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        check_gpu()

    completed = _load_completed(OUTPUT_CSV)
    rows = []

    for dataset in DATASETS:
        for seed in SEEDS:
            for lamda in LAMDA_VALUES:
                if (dataset, seed, lamda) in completed:
                    continue
                try:
                    row = run_one(dataset, seed, lamda, use_gpu=use_gpu)
                except Exception as exc:
                    row = {"dataset": dataset, "model": MODEL, "seed": seed,
                            "noise_protocol": PROTOCOL, "lamda": lamda,
                            "budget": BUDGET, "target_ratio": RATIO, "error": str(exc)}
                    print(f"FAIL {dataset}/{seed}/lamda={lamda}: {exc}", flush=True)
                rows.append(row)
                df = pd.DataFrame([row])
                write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                ba = row.get("balanced_accuracy", float("nan"))
                ba_str = f"{ba:.4f}" if isinstance(ba, float) and ba == ba else "ERR"
                print(f"  {dataset}/seed={seed}/lamda={lamda}: BA={ba_str}", flush=True)

    print(f"\nDone. {len(rows)} rows -> {OUTPUT_CSV}", flush=True)

    # Print lamda decision summary
    if OUTPUT_CSV.exists():
        df_all = pd.read_csv(OUTPUT_CSV)
        if "balanced_accuracy" in df_all.columns:
            agg = df_all.groupby("lamda")["balanced_accuracy"].mean()
            print("\nLamda sensitivity summary (mean BA over datasets × seeds):")
            for l, ba in agg.items():
                print(f"  lamda={l:3d}: BA={ba:.4f}")
            ba_30 = agg.get(30, float("nan"))
            ba_100 = agg.get(100, float("nan"))
            delta = (ba_100 - ba_30) * 100
            print(f"\nΔBA(100 vs 30) = {delta:+.2f}pp")
            if delta > 0.5:
                print("GATE: lamda=100 wins by >0.5pp — Phase 5 should use lamda=100")
            else:
                print("GATE: lamda=30 equivalent to lamda=100 — keep lamda=30, add footnote")


if __name__ == "__main__":
    main()
