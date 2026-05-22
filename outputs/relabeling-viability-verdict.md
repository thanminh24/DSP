# Relabeling Viability Verdict

## Evidence Summary

- `balanced_oof_relabel` vs `class_proportional`: delta=+0.0077, p=4.543e-11, n=1200(*)
- `balanced_oof_relabel` vs `random_relabel`: delta=+0.0624, p=1.12e-154, n=1200(*)
- `balanced_oof_relabel` vs `unbalanced_oof_relabel`: delta=-0.0005, p=0.2166, n=1200
- `balanced_oof_relabel` vs `global_top_loss`: delta=+0.0548, p=2.565e-96, n=1200(*)
- `balanced_oof_relabel` vs `no_cleaning`: delta=+0.0673, p=2.63e-126, n=1200(*)

## Model Coverage

calibrated_lr, catboost, extra_trees, hgb, lightgbm, lr, random_forest, xgboost

## Per-Model Win Rate vs class_proportional (balanced_accuracy)

- calibrated_lr: 91/150=61% mean_delta=+0.0188 p=5.226e-06
- catboost: 97/150=65% mean_delta=+0.0099 p=3.821e-05
- extra_trees: 98/150=65% mean_delta=+0.0131 p=6.619e-06
- hgb: 85/150=57% mean_delta=+0.0087 p=1.035e-02
- lightgbm: 84/150=56% mean_delta=+0.0018 p=2.485e-01
- lr: 86/150=57% mean_delta=+0.0100 p=2.633e-03
- random_forest: 97/150=65% mean_delta=+0.0115 p=2.891e-04
- xgboost: 41/150=27% mean_delta=-0.0126 p=9.599e-08

## Verdict

**VIABLE** — significant wins on majority of baseline comparisons.

## Unresolved Questions

- Confirm weak-supervision transfer after Phase 5 data is available.
