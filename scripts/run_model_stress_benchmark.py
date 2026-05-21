"""Model-family stress benchmark for confidence-guided relabeling."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.models.factories import list_publication_models
from scripts.run_relabeling_viability_sweep import NOISE_PROTOCOLS, run_single_viability
from pipeline.core.config import BaseExperimentConfig

DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
OUT_CSV = "outputs/model-stress-results.csv"


def main() -> None:
    models = sys.argv[1:] or list_publication_models(include_optional=True)
    rows = []
    protocol_items = [
        item for item in NOISE_PROTOCOLS.items()
        if item[0] in {"hidden_minority_medium", "symmetric", "reverse_asymmetric"}
    ]
    combos = list(itertools.product(DATASETS, models, SEEDS, protocol_items))
    for i, (dataset, model_name, seed, (noise_name, (mn, mj))) in enumerate(combos, 1):
        cfg = BaseExperimentConfig(
            minority_to_majority_noise=mn,
            majority_to_minority_noise=mj,
        )
        try:
            batch = run_single_viability(dataset, model_name, seed, noise_name, cfg)
            for row in batch:
                row["stress_scope"] = "model_family"
            rows.extend(batch)
        except Exception as exc:
            rows.append({
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "noise_protocol": noise_name,
                "method": "ERROR",
                "error": str(exc),
                "stress_scope": "model_family",
            })
            print(f"FAIL {dataset}/{model_name}/{seed}/{noise_name}: {exc}", flush=True)
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"progress {i}/{len(combos)} rows={len(rows)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
