# Validation Design — Phase 1

This document records the design and freezing of the local validation benchmark used for all
future model-selection decisions. See `experiments/splits/validation_v1_metadata.md` for the
frozen artifact's exact numbers and hashes, and `docs/DATASET_AUDIT.md` for the underlying
dataset evidence this design is based on.

## Official metric

Macro-averaged F0.5 (β=0.5) per Source-1 entity, precision weighted 2× over recall:

```
T = truth target-ID set, P = predicted target-ID set (per Source-1 entity)

T empty, P empty      -> entity score = 1.0
T empty, P non-empty  -> entity score = 0.0
T non-empty, P empty  -> entity score = 0.0
otherwise:
    TP = |T ∩ P|
    precision = TP / |P|
    recall    = TP / |T|
    F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)  (0.0 if TP == 0)

macro score = mean(entity F0.5 over all Source-1 entities)
```

Implemented exactly in
`student_resource/code/business_entity_resolution/src/evaluation.py::evaluate`, verified against
Amazon's worked example (precision=2/3, recall=1 → F0.5≈0.714285714285714) and 16 other edge
cases in
`student_resource/code/business_entity_resolution/tests/test_evaluation.py` (17/17 passing).

## Exact singleton behavior

A Source-1 entity with an empty truth set scores 1.0 for an empty prediction and 0.0 for *any*
non-empty prediction, regardless of how small — there is no partial credit for a false merge on
a true singleton. 5.58% of training Source-1 entities are true singletons
(`docs/DATASET_AUDIT.md` §5).

## Why a pair-level split is invalid

Splitting at the level of individual (S1, S2/S3) *pairs* — e.g. putting some of an entity's true
matches in "train" and others in "validation" — would let a model observe part of an entity's
true answer during development and be evaluated on the rest of the same entity, which is a
direct label leak for that entity's non-singleton status and partial match set. The correct
grouping unit is therefore the whole Source-1 entity (with all of its true matches) or larger.

## Target-overlap (leakage) findings

Phase 1 measured target reuse across the *entire* training ground truth graph
(`experiments/audit_cache/target_reuse_stats.json`): **zero** Source-2 or Source-3 ID is a true
match target for more than one Source-1 entity (`max_s1_degree_for_any_target == 1`). This rules
out the leakage vector a graph/connected-component grouping exists to prevent, so **no grouping
beyond the individual Source-1 entity was required**. `validation_split.py` still accepts an
optional `component_map` parameter that would force shared-target entities into the same split,
so a future re-audit (e.g. after any ground-truth update) can enable grouping without changing
the split-building call sites — but it is not active in `validation_v1`.

## Primary split design

- **Grouping unit:** individual Source-1 entity (justified above).
- **Fraction:** 80% development / 20% validation.
- **Stratification key:** `(country, singleton, match_bucket)` where `match_bucket` ∈
  {`"0"`, `"1"`, `"2+"`} — see `docs/DATASET_AUDIT.md` §5–§6 for why these three factors
  (country, singleton status, match multiplicity) are the first-order structural axes of the
  task.
- **Seed:** 42.
- **Algorithm:** `numpy.random.RandomState(seed)`, per-stratum shuffle, `round(len(stratum) *
  0.20)` assigned to validation, remainder to development. Implementation:
  `validation_split.py::build_strata` + `assign_split`.

## Split sizes

Development: 1,765,456 entities. Validation: 441,365 entities. Full country/singleton/
match-bucket breakdowns are in `experiments/splits/validation_v1_metadata.md`.

## Reproducibility

Frozen at `experiments/splits/validation_v1.tsv` (columns: `source1_entity_id`, `split`).
Manifest SHA-256 (order-independent): `f9772dd4fe8b515657c620b0952f933902ae2dd77ab736887162bdea0a2fc7e0`.
Deterministic under the fixed seed (verified by
`test_deterministic_under_fixed_seed`); every train Source-1 entity appears exactly once, and
development/validation partition it completely and disjointly (verified against the real
frozen file by `test_frozen_split_integrity.py`, and generically on synthetic data by
`test_validation_split.py` — 8 tests, all passing).

## Score granularity

At N=441,365 validation entities, one entity flipping from wholly wrong to wholly correct moves
macro F0.5 by ≈2.27 × 10⁻⁶. A genuine **+0.001** improvement corresponds to roughly 441 entities'
worth of correction; a **+0.0001** improvement (~44 entities) is resolvable in principle but sits
close to the noise floor of any stochastic step in the pipeline (random tie-breaking, floating
point summation order, etc.) — treat deltas at that scale as unproven until they reproduce
across at least two runs. Full derivation in `experiments/splits/validation_v1_metadata.md`.

## Optional stress tests (secondary, not for model selection)

- **Cross-country transfer:** develop on India-only train entities, evaluate on US-only (and
  vice versa). Feasible with the existing data; not yet built. Purpose: France appears only in
  test with zero training examples, so we have no direct way to measure transfer to an unseen
  country. A cross-country stress test on the *seen* countries is the closest available proxy
  for "how much does a country-specific model or feature cost us on a country it never saw,"
  even though it cannot directly predict France performance (`docs/DATASET_AUDIT.md` §17).
- These stress tests are diagnostic only and must never replace `validation_v1` as the primary
  selection metric (policy below).

## Model-selection policy

Future models/pipeline changes are promoted based primarily on, in this order:

1. Frozen local validation (`validation_v1`) macro F0.5.
2. Singleton behavior (singleton accuracy diagnostic from `evaluation.py`).
3. Non-singleton macro F0.5 (the 94.4% of entities carrying most of the achievable score).
4. Candidate recall (once Phase 2 candidate generation exists).
5. Robustness / stress-test behavior (cross-country transfer, once built).
6. Runtime / resource cost (must stay within a ≤8B-parameter, MIT/Apache-2.0-licensed budget —
   see `PROJECT_RULES.md`).
7. Public leaderboard score, as **secondary evidence only**.

**The public leaderboard score alone must never override a clearly weaker local validation
result without investigation first** — the public leaderboard uses only part of the hidden test
set, includes France (structurally similar per §17 but with unverified transfer), and is a
smaller, noisier signal than the 441,365-entity frozen validation set. A leaderboard gain that
contradicts validation is a signal to check for validation/test drift or overfitting to
leaderboard feedback, not a reason to abandon the frozen split.

## Unseen-France limitation

`validation_v1` is built entirely from training data, which contains only US and India. It
cannot measure performance on France by construction. This is a known, accepted limitation of
any split built from the supplied training set — see `docs/DATASET_AUDIT.md` §17 for the
structural evidence available in its place, and the cross-country stress test above as the
closest indirect proxy.
