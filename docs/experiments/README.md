# Experiments Index

## Consolidated Results

| File | Rows | Description |
|------|------|-------------|
| `COINS-all-results.xlsx` | 49,250 | All experiments — 9 tabs + README sheet |
| `COINS-all-results.csv` | 49,250 | Same data as flat CSV with `experiment` tag column |

## Individual Experiment CSVs (`raw/`)

| File | Rows | Purpose |
|------|------|---------|
| `raw/full-benchmark-solution-v2.csv` | 24,750 | Main benchmark — 15 datasets × 9 models × 7 methods × 3 protocols × 10 seeds. Table 1 source. |
| `raw/full-benchmark-ir030-solution.csv` | 8,250 | IR=0.30 sensitivity sweep — same protocol, 5 datasets. Table 6 source. |
| `raw/competitor-headtohead-expanded.csv` | 8,100 | External comparison — LR+SVM+HGB × 6 methods × 3 protocols × 15 datasets. Table 2 source. |
| `raw/cwms-msbs-deep-sweep.csv` | 4,350 | Per-model deep sweep + oracle upper bound, 5 datasets. Appendix reference. |
| `raw/rfet-ablation-sweep.csv` | 1,500 | RF/ET failure-mode ablation — cwms vs msbs vs cwms_msbs. Discussion section. |
| `raw/scorer-agnosticism-sweep.csv` | 1,250 | Self-family vs cross-family OOF scorer ablation. Validates scorer design. |
| `raw/clean-data-ablation.csv` | 400 | Zero-noise ablation — confirms no degradation without label noise. |
| `raw/failure-mode-sweep.csv` | 400 | Symmetric/reverse-asymmetric noise protocols — documents failure regimes. |
| `raw/iw-lamda-sweep.csv` | 250 | IW-SMOTE lambda sensitivity (λ ∈ {10,30,50,100,200}). Gates lambda=30 default. |

## Analysis Reports (`reports/`)

| File | Description |
|------|-------------|
| `reports/table1-15dataset-final.txt` | Table 1 formatted output — mean BA per model/method, Wilcoxon+Stouffer |
| `reports/table6-ir-sensitivity-final.txt` | Table 6 — IR=0.30 sensitivity numbers |
| `reports/competitor-experiment-setup.md` | External comparison setup details and validation notes |
| `reports/scorer-novelty-audit.md` | Novelty audit for the self-family OOF scoring design |

## Superseded Files

Archived in `raw/archive/` — original 3-dataset pilot runs superseded by v2 and expanded runs.
