from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ExperimentConfig:
    test_size: float = 0.25
    minority_label: int = 0
    target_minority_ratio: float = 0.15
    minority_to_majority_noise: float = 0.30
    majority_to_minority_noise: float = 0.10
    cleaning_budget: float = 0.10
    seeds: tuple[int, ...] = (13, 29, 47)
    lambda_risk: float = 0.5


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )

def induce_training_imbalance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    minority_idx = np.where(y_train == cfg.minority_label)[0]
    majority_idx = np.where(y_train != cfg.minority_label)[0]
    target_minority_count = int(
        (cfg.target_minority_ratio / (1 - cfg.target_minority_ratio))
        * len(majority_idx)
    )
    keep_minority_count = min(len(minority_idx), max(2, target_minority_count))
    keep_minority = rng.choice(minority_idx, size=keep_minority_count, replace=False)
    keep_idx = np.concatenate([majority_idx, keep_minority])
    rng.shuffle(keep_idx)
    return x_train[keep_idx], y_train[keep_idx]

def inject_class_dependent_noise(
    y_clean: np.ndarray,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    y_noisy = y_clean.copy()
    noisy_mask = np.zeros_like(y_clean, dtype=bool)
    classes = np.unique(y_clean)
    if len(classes) != 2:
        raise ValueError("Smoke test expects binary labels.")

    majority_label = int([c for c in classes if c != cfg.minority_label][0])
    for idx, label in enumerate(y_clean):
        if label == cfg.minority_label:
            flip_prob = cfg.minority_to_majority_noise
            flipped_label = majority_label
        else:
            flip_prob = cfg.majority_to_minority_noise
            flipped_label = cfg.minority_label
        if rng.random() < flip_prob:
            y_noisy[idx] = flipped_label
            noisy_mask[idx] = True
    return y_noisy, noisy_mask

def out_of_fold_suspiciousness(
    x_train: np.ndarray,
    y_noisy: np.ndarray,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(y_noisy), 2), dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(x_train, y_noisy):
        model = make_model(seed)
        model.fit(x_train[train_idx], y_noisy[train_idx])
        probabilities[valid_idx] = model.predict_proba(x_train[valid_idx])
    clipped = np.clip(probabilities[np.arange(len(y_noisy)), y_noisy], 1e-12, 1.0)
    return -np.log(clipped)

def select_random(n_samples: int, budget_count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.arange(n_samples), size=budget_count, replace=False)

def select_global(suspiciousness: np.ndarray, budget_count: int) -> np.ndarray:
    return np.argsort(suspiciousness)[::-1][:budget_count]

def select_class_proportional(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
) -> np.ndarray:
    selected: list[int] = []
    for label in np.unique(y_noisy):
        label_idx = np.where(y_noisy == label)[0]
        class_budget = int(round(budget_count * len(label_idx) / len(y_noisy)))
        ranked = label_idx[np.argsort(suspiciousness[label_idx])[::-1]]
        selected.extend(ranked[:class_budget].tolist())
    return np.array(selected[:budget_count], dtype=int)

def select_crcc(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    cfg: ExperimentConfig,
) -> np.ndarray:
    class_caps = {
        label: int(round(budget_count * np.mean(y_noisy == label)))
        for label in np.unique(y_noisy)
    }
    class_risk = {label: 0.0 for label in np.unique(y_noisy)}
    class_risk[cfg.minority_label] = 1.0
    adjusted = np.array(
        [score - cfg.lambda_risk * class_risk[label] for score, label in zip(suspiciousness, y_noisy)]
    )
    deleted_count = {label: 0 for label in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(adjusted)[::-1]:
        label = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[label] < class_caps[label]:
            selected.append(int(idx))
            deleted_count[label] += 1
    return np.array(selected, dtype=int)

def evaluate_selection(
    method: str,
    selected: np.ndarray,
    x_train: np.ndarray,
    y_noisy: np.ndarray,
    y_clean: np.ndarray,
    noisy_mask: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
) -> dict[str, float | int | str]:
    keep_mask = np.ones(len(y_noisy), dtype=bool)
    keep_mask[selected] = False
    model = make_model(seed)
    model.fit(x_train[keep_mask], y_noisy[keep_mask])
    y_pred = model.predict(x_test)
    clean_minority = (y_clean[selected] == cfg.minority_label) & ~noisy_mask[selected]
    return {
        "seed": seed, "method": method, "deleted": int(len(selected)),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "minority_recall": recall_score(y_test, y_pred, pos_label=cfg.minority_label),
        "noise_precision_deleted": float(noisy_mask[selected].mean()) if len(selected) else 0.0,
        "clean_minority_deletion_rate": float(clean_minority.mean()) if len(selected) else 0.0,
    }

def run() -> pd.DataFrame:
    cfg = ExperimentConfig()
    data = load_breast_cancer()
    results = []
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=cfg.test_size,
            stratify=data.target,
            random_state=seed,
        )
        x_train, y_clean = induce_training_imbalance(x_train, y_train, cfg, rng)
        y_noisy, noisy_mask = inject_class_dependent_noise(y_clean, cfg, rng)
        suspiciousness = out_of_fold_suspiciousness(x_train, y_noisy, seed)
        budget_count = max(1, int(round(cfg.cleaning_budget * len(y_noisy))))
        selectors = {
            "random_deletion": select_random(len(y_noisy), budget_count, rng),
            "global_top_loss": select_global(suspiciousness, budget_count),
            "class_proportional": select_class_proportional(suspiciousness, y_noisy, budget_count),
            "crcc": select_crcc(suspiciousness, y_noisy, budget_count, cfg),
        }
        selectors["no_cleaning"] = np.array([], dtype=int)
        for method, selected in selectors.items():
            results.append(
                evaluate_selection(
                    method, selected, x_train, y_noisy, y_clean, noisy_mask, x_test, y_test, cfg, seed
                )
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    frame = run()
    frame.to_csv(output_dir / "crcc-smoke-test-results.csv", index=False)
    print(frame.groupby("method").mean(numeric_only=True).round(4).to_string())
