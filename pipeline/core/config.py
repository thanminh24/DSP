"""Shared experiment configuration — single source of truth for all constants."""

from __future__ import annotations

from dataclasses import dataclass, field

OPERATING_CONDITION = (
    "Confidence-guided OOF relabeling is designed for hidden-minority noise only: "
    "cases where minority examples are mislabeled as majority class. "
    "It does not improve and may harm performance under reverse asymmetric or "
    "symmetric noise where the minority class is already over-represented in labels."
)

LAMBDA_NAMES: dict[float, str] = {
    0.0: "crcc_p_l0",
    0.25: "crcc_p_l025",
    0.5: "crcc_p_l05",
    1.0: "crcc_p_l10",
}


@dataclass(frozen=True)
class BaseExperimentConfig:
    """Base config shared by all experiment variants.

    Override fields in subclass dataclasses for specific experiments.
    """

    datasets: tuple[str, ...] = ("pima", "credit-g", "yeast", "ecoli", "phoneme")
    seeds: tuple[int, ...] = (13, 29, 47, 61, 83)
    test_size: float = 0.25
    target_minority_ratio: float = 0.15
    minority_to_majority_noise: float = 0.30
    majority_to_minority_noise: float = 0.10
    cleaning_budget: float = 0.10
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    minority_cap_factor_m: float = 0.5
    n_cv_folds: int = 5
    minority_label: int = 1

    # Per-dataset minority label overrides (e.g. Phoneme: nasal=1 is minority
    # but the class label in the encoded y array may differ from default 1).
    minority_label_map: dict[str, int] = field(default_factory=dict)

    @property
    def method_names(self) -> list[str]:
        names = [
            "no_cleaning", "random_deletion", "global_top_loss",
            "class_proportional", "oracle_deletion",
        ]
        for lam in self.lambda_grid:
            names.append(LAMBDA_NAMES[lam])
        names.append("crcc_m")
        return names

    def get_minority_label(self, dataset_name: str) -> int:
        """Resolve the minority label for a given dataset."""
        return self.minority_label_map.get(dataset_name, self.minority_label)
