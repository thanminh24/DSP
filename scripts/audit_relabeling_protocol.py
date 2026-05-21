"""Leakage and protocol audit for confidence-guided relabeling."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "relabeling-protocol-audit.md"


def main() -> int:
    checks = [
        _check_oof_scoring(),
        _check_clean_label_isolation(),
        _check_active_runner_selection_inputs(),
        _check_split_order(),
        _check_relabeler_signature(),
        _check_baseline_fairness_surface(),
        _check_poison_pill_declared(),
        _check_archive_ignored(),
    ]
    failed = [c for c in checks if not c["passed"]]
    _write_report(checks)
    print(f"wrote {OUT}")
    if failed:
        for item in failed:
            print(f"FAIL: {item['name']} - {item['detail']}")
        return 1
    print("PASS: relabeling protocol audit")
    return 0


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _check_oof_scoring() -> dict:
    src = _read("pipeline/scoring/balanced_oof.py")
    required = [
        "StratifiedKFold",
        "folds.split(X, y_noisy)",
        "model.fit(X[tr_idx], y_noisy[tr_idx]",
        "predict_proba(X[va_idx])",
        "probs[minority_mask] = np.nan",
    ]
    return _result(
        "out-of-fold scoring isolation",
        all(token in src for token in required),
        "balanced scorer must fit on tr_idx and score va_idx only",
    )


def _check_clean_label_isolation() -> dict:
    src = _read("scripts/run_augment_experiment.py")
    active = _read("scripts/run_relabeling_viability_sweep.py")
    relabel_pos = src.find("relabel_typeA(")
    truth_pos = src.find("y_tr[rel_idx]")
    active_relabel = active.find("relabel_typeA(")
    active_truth = active.find("y_tr")
    ok = (
        relabel_pos != -1 and truth_pos != -1 and truth_pos > relabel_pos
        and active_relabel != -1 and active_truth != -1
    )
    return _result(
        "clean-label isolation",
        ok and "relabel_typeA(\n                y_noisy, scores" in src,
        "clean labels may be used only after relabel indices are selected",
    )


def _check_split_order() -> dict:
    src = _read("scripts/run_relabeling_viability_sweep.py")
    split = src.find("train_test_split(")
    noise = src.find("inject_noise(")
    imbalance = src.find("induce_imbalance(")
    return _result(
        "split before imbalance/noise",
        -1 not in {split, noise, imbalance} and split < imbalance < noise,
        "train/test split must happen before training-only imbalance and noise injection",
    )


def _check_relabeler_signature() -> dict:
    src = _read("pipeline/augmentation/relabeling.py")
    signature = src[src.find("def relabel_typeA("):src.find("def random_relabeling(")]
    forbidden = ["y_clean", "y_true", "gold", "test"]
    return _result(
        "relabeler has no clean-label inputs",
        not any(token in signature for token in forbidden),
        "relabeler must use y_noisy and scores only",
    )


def _check_active_runner_selection_inputs() -> dict:
    src = _read("scripts/run_relabeling_viability_sweep.py")
    start = src.find("y_rel, idx = relabel_typeA(")
    end = src.find("return _eval_relabel", start)
    block = src[start:end]
    forbidden = ["y_tr", "y_te", "y_clean", "gold"]
    ok = start != -1 and end != -1 and not any(token in block for token in forbidden)
    return _result(
        "active runner balanced relabel selection inputs",
        ok,
        "balanced relabel selection must use y_noisy, OOF scores, budget, and labels only",
    )


def _check_baseline_fairness_surface() -> dict:
    src = _read("scripts/run_relabeling_viability_sweep.py")
    required = [
        "budget_count = max(1, int(round(cfg.cleaning_budget * len(y_noisy))))",
        "train_test_split(",
        "METHODS = [",
        "\"random_relabel\"",
        "\"unbalanced_oof_relabel\"",
        "\"class_proportional\"",
    ]
    return _result(
        "baseline fairness surface",
        all(token in src for token in required),
        "core baselines must share split, noisy labels, and budget",
    )


def _check_poison_pill_declared() -> dict:
    src = _read("scripts/run_relabeling_viability_sweep.py")
    return _result(
        "poison-pill controls present",
        "\"shuffled_score_relabel\"" in src and "\"inverted_score_relabel\"" in src,
        "shuffled/inverted score relabel controls must be present in active METHODS",
    )


def _check_archive_ignored() -> dict:
    path = ROOT / ".gitignore"
    src = path.read_text(encoding="utf-8") if path.exists() else ""
    return _result(
        "stale archive ignored",
        "plans/archive/" in src and "docs/archive/" in src,
        "archive folders should stay out of publication context",
    )


def _result(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _write_report(checks: list[dict]) -> None:
    OUT.parent.mkdir(exist_ok=True)
    lines = ["# Relabeling Protocol Audit\n"]
    verdict = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    lines.append(f"## Verdict\n\n**{verdict}**\n")
    lines.append("## Checks\n")
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{c['name']}`: {c['detail']}")
    lines.append("\n## Unresolved Questions\n")
    lines.append("None.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
