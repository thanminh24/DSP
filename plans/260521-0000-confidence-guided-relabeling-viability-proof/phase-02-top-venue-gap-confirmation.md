---
phase: 2
title: "Top-Venue Gap Confirmation"
status: pending
priority: P1
effort: "2d"
dependencies: [1]
---

# Phase 2: Top-Venue Gap Confirmation

## Overview

Run a publication-grade novelty audit before claiming the gap. The goal is to prove that
2024+ Q1/Q2 journal papers and A*/A conference papers have not already filled the exact
gap: lightweight tabular post-detection relabeling for hidden minority examples under
imbalanced noisy labels.

## Requirements

- Functional: produce an auditable literature survey table with venue rank, year, method,
  setting, data modality, intervention type, and similarity to our method.
- Non-functional: use primary sources only where possible. Do not rely on blog summaries.

## Architecture

```text
venue/source list
  -> keyword search
  -> candidate paper triage
  -> close-paper matrix
  -> exact-gap filled? yes/no
  -> claim boundary update
```

## Venue Scope

Conferences:

- A*/top ML: NeurIPS, ICML, ICLR.
- A/top AI/data mining: KDD, AAAI, IJCAI, ECML-PKDD.
- Related A/top CV where noisy-label learning is active: CVPR, ICCV, ECCV.

Journals:

- Q1/Q2 or top ML/AI/data journals: JMLR, TMLR, TPAMI, TKDE, TNNLS, Pattern Recognition,
  Information Sciences, Machine Learning, Neural Networks, Expert Systems with Applications.

Year scope:

- Mandatory: 2024, 2025, 2026 to current date.
- Include 2023 only when a 2024+ paper cites it as direct nearest work.

## Search Queries

Use combinations of:

- "imbalanced noisy labels sample selection"
- "long-tailed noisy data label correction"
- "noisy labels class imbalance relabeling"
- "tabular noisy labels relabeling"
- "weak supervision noisy tabular relabeling"
- "confident learning imbalanced data"
- "out-of-fold confidence label correction"
- "budgeted label cleaning imbalanced"

## Related Code Files

- Create: `plans/260521-0000-confidence-guided-relabeling-viability-proof/research/top-venue-gap-survey.md`
- Create: `plans/260521-0000-confidence-guided-relabeling-viability-proof/reports/novelty-verdict.md`
- Modify: `docs/research-foundation.md` after the survey is verified.

## Implementation Steps

1. Build a venue/source checklist with one row per target venue/journal.
2. Search each venue/source for 2024+ papers using the query set.
3. For each candidate, record:
   - year and venue.
   - rank class: A*/A or Q1/Q2.
   - data modality: tabular, image, text, graph, multimodal.
   - handles class imbalance?
   - handles noisy labels?
   - intervention: delete/filter, reweight, relabel, semi-supervise, active human review.
   - uses OOF or out-of-sample confidence?
   - uses class-balanced scorer?
   - targets hidden minority examples in majority-labeled pool?
   - can serve as a baseline?
4. Mark close papers as:
   - Direct blocker: fills the same gap.
   - Partial overlap: must cite and distinguish.
   - Context only: useful related work, not a blocker.
5. Update the working claim:
   - If direct blocker exists, pivot to benchmarking/replication/extension.
   - If partial overlaps exist, narrow claim to the exact differentiator.
6. Create a "reviewer ammunition" section listing likely citations reviewers will mention.

## Success Criteria

- [ ] Survey covers at least 30 candidate papers or every searched venue/source has a
      documented "none found" entry.
- [ ] At least 10 close papers are summarized in the final matrix if they exist.
- [ ] Every direct or partial overlap has a one-sentence distinction from our method.
- [ ] Novelty verdict is one of: CLEAR, NARROW BUT DEFENSIBLE, BLOCKED/PIVOT.
- [ ] No paper draft proceeds unless verdict is CLEAR or NARROW BUT DEFENSIBLE.

## Risk Assessment

If a close method exists, do not hide it. Either make it a baseline, narrow our claim, or
pivot to an empirical tabular benchmark where our method is the simpler alternative.

## Security Considerations

None.

## Next Steps

Proceed to Phase 3 only after recording a novelty verdict.

## Venue Ranking Standards

- **Conferences**: CORE ranking (A*, A, B, C).
- **Journals**: Scimago/JCR ranking (Q1, Q2, Q3, Q4).

Use CORE for all conference-based novelty gating. Use Scimago Q1/Q2 as the threshold for
journal inclusion in the survey.

## Unresolved Questions

None.
