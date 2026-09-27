# Experiment History

This is a condensed record of the project's major milestones and submissions, verified
against the local experiment ledger. Only figures that could be directly verified against
a file, hash, or logged metric are reported; nothing here is estimated or rounded up.

## Milestones

| Phase | Change | Outcome |
|---|---|---|
| Dataset audit | Full raw-integrity audit of all source tables and ground truth | Zero duplicate IDs, zero invalid ID prefixes; singleton rate 5.58%; zero cross-entity target reuse (a per-entity split is leakage-safe) |
| Validation design | Frozen 80/20 entity-level split (`validation_v1`, seed 42), evaluator verified against the official worked example | 17/17 evaluator tests, 29/29 split integrity tests passing |
| Candidate generation (blocking) | Union of exact match, forward/reverse word-TF-IDF, numeric-signature match, character n-gram retrieval | Link recall 0.891 on the initial architecture; superseded by the retrieval pipeline below |
| Retrieval redesign (R1 → R2 → R3) | Hashed combination-key retrieval with per-country document frequency, replacing the token/blocking-union approach | Substantially higher candidate recall and lower candidate volume; became the architecture used for every real submission from #3 onward |
| Stage-1 matcher | LightGBM GBDT on engineered lexical/numeric/retrieval-rank features, hard-negative training | See submission ledger below |
| Stage-2 | Collective entity-level features (owner competition, target/entity probability share) on cross-fitted stage-1 probabilities | Consistent improvement over stage-1 alone at every retrieval generation |
| Normalization v2 | NFKC + casefold, Latin diacritic removal, train-derived script-to-Latin alias map (1,470 tokens, learned only from `matcher_train` links) | Eliminated the large majority of script-mismatch false negatives |
| IDF augmentation | Random half-row IDF-scale jitter during stage-1/stage-2 training | Reduced dependence on corpus-specific IDF scale, improving transfer to the unseen France partition |
| Final decoder | Per-entity adaptive thresholds (target source x count of confident candidates), fitted on `calibration_holdout` only and frozen before evaluation | +0.000174 macro F0.5 on `dev_eval` vs. the global threshold (paired bootstrap 95% CI +0.000041 to +0.000308) |

## Submission ledger

Public scores are macro-averaged F0.5, the official competition metric.

| # | Architecture | Public score | Outcome |
|---|---|---|---|
| 1 | Initial blocking union + GBDT (B004 architecture) | 0.829192 | Superseded |
| 3 | Retrieval redesign (R1) + stage-1 matcher | 0.948318 | Superseded |
| 4 | R1 + stage-1 + stage-2 | not uploaded | — |
| 5 | R2 retrieval stack | 0.967319 | Superseded |
| 6 | R3 (IDF-scope variant) | 0.966812 | Regressed vs. #5, not adopted |
| 7 | R3 with row-augmentation | 0.969719 | Superseded |
| 8 | Stage-2 collective features on R2 base | not uploaded | Not recommended |
| 9 | R3-augmented + stage-2 | 0.970895 | Superseded |
| 10 | R3-augmented + stage-2 + corroboration features | 0.971415 | Superseded |
| 11 | R3 retrieval + stage-1 + stage-2 (final architecture) | 0.973142 | Kept as the immutable fallback |
| 12 | Submission 11 + corroboration-feature stack | 0.972955 | Regressed vs. #11, not adopted |
| Final | Submission 11 + per-entity adaptive-threshold decoder | **0.973462** | **Best verified result** |

Every real submission's exact matching-results and candidate-pairs file hashes are
recorded in the local (non-public) experiment ledger; the final submission's hashes are
in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Techniques evaluated and not adopted

Recorded here so the negative results aren't lost, per the project's own evidence-preservation policy:

- **Character n-gram retrieval at full scale** — a small pilot suggested a large rescue rate for
  hard-to-retrieve links, but at true full corpus scale the benefit was much smaller
  (closing about 12% of the recall gap it appeared to close in the pilot) than a bounded
  target-pool pilot suggested. Kept as a documented caution about blocking-pilot scale effects.
- **Per-source (rather than global) forward retrieval** — no material recall advantage at a
  matched candidate-edge budget.
- **Numeric rare-token retrieval route** — rejected twice independently; incremental value
  was negligible relative to its added candidate volume.
- **Target-to-target corroboration / rival-anchor features (stage-2)** — measured gains were
  not statistically distinguishable from zero in paired evaluation; not adopted into the
  final stage-2 model.
- **Full isotonic-regression calibration** — a simple global-threshold-per-group decoder,
  fitted by exact coordinate ascent, matched or exceeded the more complex alternatives
  tried and was adopted for its simplicity and calibration-only fitting guarantee.
- **Probability blending between stage-1 and stage-2 models** — tested blend weights did not
  improve on the stage-2 model alone.
- **Applying the adaptive-threshold decoder to countries absent from training** — not
  validated on labeled data for the unseen-country partition, so the global threshold is
  kept there instead (see `docs/REPRODUCIBILITY.md`).

## Open-set robustness (France)

The test set includes France, a country with zero training examples (~15% of test
entities). This was treated as a first-class constraint throughout: no country-specific
hand-tuned normalization rules were adopted (the project's normalization and alias-map
policy is trained-data-derived only, never hand-curated per country), and the adaptive
decoder explicitly falls back to the global, training-validated threshold for any country
not observed during training, rather than extrapolating an unvalidated per-entity rule to
it.
