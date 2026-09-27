# Validation Split — `validation_v1`

**Status: FROZEN.** Future experiments should use this split unless a documented reason
(recorded in `DECISIONS.md`) creates `validation_v2`.

## Creation

- **Created (UTC):** 2026-09-25T06:03:21+00:00
- **Seed:** 42
- **Split algorithm:** stratified random split of Source-1 entities, grouping unit = single
  Source-1 entity (see leakage finding below), target validation fraction = 20%, per-stratum
  shuffle via `numpy.random.RandomState(42)`, `round(len(stratum) * 0.20)` sent to
  `validation`, the remainder to `development`.
- **Implementation:** `student_resource/code/business_entity_resolution/src/validation_split.py`
  (`build_strata` + `assign_split`), covered by
  `student_resource/code/business_entity_resolution/tests/test_validation_split.py`.

## Leakage grouping rule

**VERIFIED:** Phase 1's target-reuse audit (`experiments/audit_cache/target_reuse_stats.json`)
found `max_s1_degree_for_any_target == 1` — every Source-2/Source-3 ID that appears as a true
match target is linked to exactly one Source-1 entity in `train_ground_truth.tsv`. There is no
cross-Source-1 target sharing, so **no connected-component/graph grouping was needed**: a plain
per-entity split is leakage-safe. `assign_split` still accepts an optional `component_map` so a
future re-audit that finds reuse (e.g. after a ground-truth update) can force shared-target
entities into the same split without an API change.

## Stratification

Stratum key = `(country, singleton, match_bucket)` where:
- `country` — the Source-1 entity's own `country` field (US / India in train).
- `singleton` — whether the entity has zero true matches.
- `match_bucket` — `"0"` / `"1"` / `"2+"` true-match count (coarse, per spec section 26).

Rationale: country and singleton status are both first-order factors in the official metric
(precision-heavy scoring, singleton correctness) and in later modeling (country as a possible
blocking signal, singleton detection as a separate sub-problem). The match-count bucket
prevents rare high-multiplicity entities from being under-represented in validation. No finer
stratification was used — six strata (2 countries × 3 buckets, singleton is implied by bucket
`"0"`) all have tens of thousands to over a million members, so none is support-starved.

## Split sizes

| | development | validation | total |
|---|---:|---:|---:|
| Source-1 entities | 1,765,456 | 441,365 | 2,206,821 |
| Fraction | 80.00% | 20.00% | 100% |

### Country distribution

| country | development | validation |
|---|---:|---:|
| US | 1,058,906 | 264,727 |
| India | 706,550 | 176,638 |

### Singleton distribution

| | development | validation |
|---|---:|---:|
| non-singleton | 1,666,858 | 416,716 |
| singleton | 98,598 | 24,649 |

### Match-count bucket distribution

| bucket | development | validation |
|---|---:|---:|
| 0 (singleton) | 98,598 | 24,649 |
| 1 | 95,325 | 23,832 |
| 2+ | 1,571,533 | 392,884 |

All per-stratum validation fractions land within ~0.1 percentage points of the 20% target
(verified by the split-assignment code path; see `test_stratum_proportions_close_to_target_fraction`
in the test suite for the property on synthetic data of comparable stratum sizes).

## Reproducibility

- Frozen split file: `experiments/splits/validation_v1.tsv`
  (columns: `source1_entity_id`, `split` ∈ {`development`, `validation`}).
- Manifest SHA-256 (order-independent, over sorted `source1_entity_id\tsplit` lines):
  `f9772dd4fe8b515657c620b0952f933902ae2dd77ab736887162bdea0a2fc7e0`
- Source file hashes at creation time:
  - `train_ground_truth.tsv` SHA-256: `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037`
  - `train_source1.tsv` SHA-256: `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973`
- Regeneration: re-running `assign_split(build_strata(gt, s1), SplitConfig(seed=42))` against
  the same two files reproduces this file exactly (deterministic under fixed seed; verified by
  `test_deterministic_under_fixed_seed` and `test_manifest_hash_reproducible_and_order_independent`).

## Score granularity at this scale

One Source-1 entity flipping from a wrong to a fully-correct prediction changes the macro F0.5
by:
- On `development` (N=1,765,456): ≈ **5.66 × 10⁻⁷** per entity.
- On `validation` (N=441,365): ≈ **2.27 × 10⁻⁶** per entity.

At a competitive target near 0.980473, a genuine **+0.001** validation improvement corresponds
to roughly **441** entities moving from a materially wrong prediction toward a correct one (a
rough equivalent, since real entity scores are continuous F0.5 values, not only 0/1 — see
`docs/DATASET_AUDIT.md` §19 for the full error-budget discussion). A **+0.0001** movement
(~44 entities) is resolvable in principle but is close to the noise floor of any
stochastic step in candidate generation or modeling; treat single-run deltas below this size
as unproven until repeated.

## Optional stress-test splits (not the primary selection metric)

Feasible, not yet built in Phase 1:
- **Cross-country transfer stress test:** develop on India-only train entities, evaluate on
  US-only (and vice versa). Useful because France is present only in test and never in train —
  this stress test is the closest available proxy for "how much does country-specific behavior
  cost us on an unseen country," using only training data.
- These are diagnostic only. Model promotion must not be based on a stress-test score alone
  (see `docs/VALIDATION_DESIGN.md`, model-selection policy).

## Limitations

- The split stratifies on the *Source-1* entity's own attributes only. It does not stratify on
  target-side (Source-2/Source-3) characteristics such as missing-address rate or normalized
  name collision rate; Phase 1's collision/missingness audit (`docs/DATASET_AUDIT.md`) found
  these fairly stable between train and test, so this was judged unnecessary complexity for a
  first split version.
- The split cannot measure transfer to France (test-only country) — see the France section of
  `docs/DATASET_AUDIT.md` and the cross-country stress test above as the closest proxy.
- `round(len(stratum) * 0.20)` on the two smallest strata (bucket `"1"`) introduces a rounding
  choice smaller than 1 entity; this has no visible effect at the observed stratum sizes
  (tens of thousands each).
