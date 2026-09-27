# Dataset Audit — Phase 1

Labels: **VERIFIED** (measured directly from the raw files in this session), **INFERRED**
(a reasonable reading of VERIFIED numbers, not itself directly measured), **UNKNOWN** (not
determined). Raw source files were read-only throughout; nothing in `student_resource\dataset\`,
`_source\`, or `_original_student_resource\` was modified.

Generating scripts (kept as a temporary audit record, not part of the production pipeline):
ad hoc scripts run against `student_resource/code/business_entity_resolution/src/` modules
(`data_io.py`, `evaluation.py`, `normalization.py`, `validation_split.py`). Raw JSON outputs are
cached under `experiments/audit_cache/` (`gt_core_stats.json`, `target_reuse_stats.json`,
`raw_source_stats.json`, `true_link_stats.json`, `collision_drift_stats.json`,
`split_v1_summary.json`).

## 1. Executive Findings

1. **Ground truth and raw sources are clean.** Zero duplicate `entity_id`s, zero invalid ID
   prefixes, zero orphaned ground-truth targets, zero rows with missing `business_name` or
   `country` anywhere, in both train and test, across all six source files — VERIFIED.
2. **Singletons are a minority, not the dominant case.** Only 5.58% of train Source-1 entities
   have zero true matches; 48% have 4+ matches (max observed = 11) — VERIFIED. This is the
   opposite of a singleton-dominated task; multi-match recovery matters as much as singleton
   precision.
3. **Zero cross-entity target reuse.** No Source-2/Source-3 ID is a true match target for more
   than one Source-1 entity in training — VERIFIED. A plain per-entity stratified split is
   leakage-safe; no connected-component grouping was required.
4. **Country is 100% consistent on every true link, in training.** Every true-matched pair has
   identical `country` on both sides, for both S2 and S3 targets — VERIFIED (on train; country
   as a *production* hard blocker is not yet decided — see §21).
5. **Exact-match shortcuts alone cover very little.** Only 0.83% of true links have both
   normalized name AND normalized address exactly equal; 77.6% have *neither* field exactly
   equal — VERIFIED. A solution built mostly on exact/near-exact string matching will miss the
   large majority of true matches.
6. **A numeric-address-token signal is unusually strong.** 63.5% of true links have an
   *exactly* matching set of numeric tokens (house numbers, PIN/ZIP-like codes) extracted from
   the address, even though full normalized-address equality is only 7.4% — VERIFIED. This looks
   like the single most promising cheap, language-agnostic candidate/feature signal found in
   Phase 1.
7. **Name-only blocking has a real blind spot.** 10% of true links have *zero* normalized-name
   token overlap (Jaccard = 0) — VERIFIED. Any candidate-generation strategy that requires
   nonzero name-token overlap will structurally miss roughly a tenth of true matches.
8. **Name collisions are common and must be respected.** Only 88–93% of normalized business
   names are unique within any single source table; the (name, normalized-address) pair is
   99–100% unique — VERIFIED. Name alone is not a safe identifying key; name+address jointly
   almost always is.
9. **Train/test structural drift is minimal.** Length, token-count, and numeric-token-presence
   distributions are nearly identical between train and test for every shared source and
   quantile checked — VERIFIED. The one intentional distribution shift is the addition of
   France in test.
10. **France is structurally similar to US/India on every measured axis** (name-uniqueness rate,
    missingness rate, numeric-token presence, name/address length scale) — VERIFIED as a
    structural statement only; whether France content (language, legal-suffix vocabulary,
    address conventions) transfers is a separate, unresolved question (§17, §21).

## 2. Dataset Scale

VERIFIED row counts (reconfirmed by full scan in this session, matching the Phase 0 estimate):

| File | Rows |
|---|---:|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

Raw file sizes range from ~127 MB (`train_ground_truth.tsv`) to ~510 MB (`train_source3.tsv`,
`test_source2.tsv`); total dataset ≈ 2.5 GB. The full S1 × (S2 ∪ S3) Cartesian product for train
alone is on the order of 2.2M × 10.3M ≈ 2.3 × 10¹³ pairs — confirmed infeasible, as expected;
candidate generation/blocking is mandatory (Phase 2, not attempted here).

## 3. Source Integrity

All six tables, VERIFIED (`experiments/audit_cache/raw_source_stats.json`):

| Table | Duplicate `entity_id` | Invalid ID prefix | Missing name | Missing address | Missing country | Unexpected columns |
|---|---:|---:|---:|---:|---:|---:|
| train_source1 | 0 | 0 | 0 | 0 (0.00%) | 0 | none |
| train_source2 | 0 | 0 | 0 | 168,967 (3.36%) | 0 | none |
| train_source3 | 0 | 0 | 0 | 175,916 (3.33%) | 0 | none |
| test_source1 | 0 | 0 | 0 | 0 (0.00%) | 0 | none |
| test_source2 | 0 | 0 | 0 | 129,408 (2.65%) | 0 | none |
| test_source3 | 0 | 0 | 0 | 136,098 (2.68%) | 0 | none |

No encoding anomalies detected (zero rows containing the Unicode replacement character
`U+FFFD` in `business_name`/`business_address`, any table). No malformed rows encountered by
the strict TSV parser (would have raised on read).

Country distribution (VERIFIED):

| Table | US | India | France |
|---|---:|---:|---:|
| train_source1 | 1,323,633 (60.0%) | 883,188 (40.0%) | — |
| train_source2 | 3,016,817 (59.9%) | 2,017,799 (40.1%) | — |
| train_source3 | 3,170,056 (60.0%) | 2,115,547 (40.0%) | — |
| test_source1 | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) |
| test_source2 | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) |
| test_source3 | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) |

Business name is *never* missing in any table (0 rows). Only `business_address` has
missingness, concentrated in S2/S3 (2.6–3.4%); S1 has zero missing addresses in either split.

## 4. Ground-Truth Integrity

VERIFIED (`experiments/audit_cache/gt_core_stats.json`, `target_reuse_stats.json`):

- 2,206,821 rows, one per unique `source1_entity_id` — no duplicates, no S1 in train missing
  from ground truth and vice versa.
- 0 rows with internal duplicate IDs inside a single `matched_entity_ids` list.
- 0 invalid target-ID prefixes (every listed ID is a well-formed `S2-\d+` or `S3-\d+`).
- 0 ground-truth target IDs absent from `train_source2.tsv`/`train_source3.tsv` — every truth
  link resolves to a real record.
- Total true links: 3,693,619 S1↔S2 links + 3,944,746 S1↔S3 links = 7,638,365 total.

## 5. Singleton Prevalence

VERIFIED:

| | Count | Rate |
|---|---:|---:|
| Total Source-1 (train) | 2,206,821 | 100% |
| Singletons (0 true matches) | 123,247 | 5.58% |
| Non-singletons | 2,083,574 | 94.42% |

**TRIVIAL TRAINING-ONLY REFERENCE — NOT A DEPLOYABLE TEST SCORE, NOT A LEADERBOARD CLAIM:** a
hypothetical system that predicts an empty match set for every training Source-1 entity scores
macro F0.5 = **0.05585** on the training ground truth (exactly the singleton rate, since a
correct empty prediction scores 1.0 and every non-singleton scores 0.0 for an empty
prediction). This quantifies how little of the competitive target (0.980473) is attributable to
singleton handling alone: the overwhelming majority of achievable score comes from correctly
recovering non-singleton matches, not from suppressing false positives on singletons (though
singleton false-positives still matter — see §19 error budget).

## 6. Match Multiplicity

VERIFIED, over all 2,206,821 train Source-1 entities:

| Match count | Entities | Rate |
|---|---:|---:|
| 0 (singleton) | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4+ | 1,058,364 | 47.96% |

Max observed match count: **11**. Mean 3.46 (3.67 restricted to non-singletons), median 3.0
(4.0 restricted to non-singletons). This is a heavily multi-match task: nearly half of all
entities need 4 or more correct targets recovered simultaneously, so **candidate recall at
higher multiplicities matters as much as singleton precision** — an argmax/top-1 matching
strategy would be structurally wrong for the majority of entities.

## 7. S2 vs S3 Match Structure

VERIFIED, per train Source-1 entity's target-source pattern:

| Pattern | Entities | Rate |
|---|---:|---:|
| Matched in both S2 and S3 | 1,776,047 | 80.48% |
| S3 only | 164,498 | 7.45% |
| S2 only | 143,029 | 6.48% |
| Empty (singleton) | 123,247 | 5.58% |

Most non-singleton entities (85.2% of non-singletons) have matches in *both* target sources —
a matching pipeline cannot treat S2 and S3 as separate independent sub-problems and expect to
recover full credit; per-entity aggregation across both sources is required.

## 8. Target Reuse / Graph Structure

VERIFIED (`experiments/audit_cache/target_reuse_stats.json`): **zero** target reuse.
`max_s1_degree_for_any_target == 1` for both S2 and S3 targets — no Source-2 or Source-3 ID is
a true-match target for more than one Source-1 entity anywhere in the training ground truth.
Consequently `n_targets_reused_across_multiple_s1 == 0` and no connected-component grouping was
needed for the validation split (see §18 and `docs/VALIDATION_DESIGN.md`).

## 9. Country Consistency

VERIFIED, computed over all 7,638,365 true links: **100.00% agreement** — every true-matched
pair has identical `country` on both sides. Breakdown:

| | Agreement rate |
|---|---:|
| Overall | 100.00% |
| S1↔S2 links only | 100.00% |
| S1↔S3 links only | 100.00% |

Given S1=US, all 4,578,522 true S1→target links have target country = US; given S1=India, all
3,059,843 true links have target country = India. Zero disagreements. **This does not yet
freeze country as a production hard blocker** (per Phase-1 scope) — France's unseen status in
training means we cannot verify this pattern holds for France, and a hard filter risks
catastrophic recall loss if it doesn't. See §17 and §21 for the open question this leaves for
Phase 2.

## 10. Missingness

VERIFIED, over all 7,638,365 true links:

- **Name:** both sides present in 100% of true links (business_name is never empty anywhere in
  the dataset).
- **Address:** both sides present in 95.59% (7,301,347); target-side-only missing in 4.41%
  (337,018); Source-1-side address is *never* missing on a true link (S1 has 0% address
  missingness overall). No true link has both addresses missing.
- Target address missingness on true links: 4.47% for S2, 4.36% for S3; 4.74% when S1 is US,
  3.92% when S1 is India — all close to the unconditional per-table missingness rates in §3, i.e.
  missingness is not concentrated on hard-to-match entities.
- Because name is always present, every true link has at least the name field usable; a
  matching approach that requires both fields present still covers 95.6% of true links.

## 11. True-Match Name Characteristics

VERIFIED, over 7,638,365 true links:

| Measure | Rate |
|---|---:|
| Raw exact equality | 4.64% |
| Normalized exact equality (non-missing, both sides) | 15.80% |
| Token-set equality (bag-of-words, order-independent) | 27.73% |

Name-token Jaccard quantiles:

| 0 | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.50 | 0.667 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**10% of true links have zero name-token overlap** — a hard tail that any name-token-overlap
blocking rule will structurally miss. The median link has 2-of-3 name tokens shared.

Name edit-ratio (`difflib.SequenceMatcher`, **approximated on a fixed random sample of 300,000
true links** — the full 7,638,365-row computation was measured too memory/time-heavy in an
initial attempt and killed before it caused system-wide memory pressure; this sample is the
documented cheaper alternative, not a full-population statistic):

| 0 | 1% | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.082 | 0.105 | 0.471 | 0.718 | 0.866 | 0.949 | 1.0 | 1.0 | 1.0 | 1.0 |

Consistent with the Jaccard tail: ~5–10% of true links have severely dissimilar names by any
lexical measure — these will need address-driven or other non-name signals to recover.

## 12. True-Match Address Characteristics

VERIFIED, over 7,638,365 true links:

| Measure | Rate |
|---|---:|
| Raw exact equality | 2.23% |
| Normalized exact equality (non-missing, both sides) | 7.44% |
| Numeric-token exact set equality | **63.49%** |

Address-token Jaccard quantiles:

| 0 | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.111 | 0.25 | 0.429 | 0.625 | 0.786 | 1.0 | 1.0 | 1.0 | 1.0 |

Address-length ratio (shorter/longer, normalized text) quantiles:

| 0 | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.248 | 0.531 | 0.789 | 0.892 | 0.963 | 1.0 | 1.0 | 1.0 | 1.0 |

The numeric-token exact-match rate (63.5%) is dramatically higher than full normalized-address
equality (7.4%) — the surrounding address text (street/city names, abbreviations, transliteration)
varies heavily even when the house number / PIN-like code is identical. This is the single
strongest cheap signal found in Phase 1 (see §21).

## 13. Joint Signal Difficulty

VERIFIED, all 7,638,365 true links bucketed (name/address "exact" = normalized-equal AND
non-missing on both sides; "strong overlap" = token Jaccard ≥ 0.5):

| Bucket | Description | Count | % |
|---|---|---:|---:|
| A | Name exact AND address exact | 63,349 | 0.83% |
| B | Name exact, address not exact | 1,143,756 | 14.97% |
| C | Name not exact, address exact | 504,830 | 6.61% |
| D | Both non-exact, both ≥0.5 token overlap | 2,782,950 | 36.43% |
| E | One field missing, other field strong (≥0.5 overlap) | 243,028 | 3.18% |
| F | Both present, but weak overlap on at least one (<0.5) | 2,869,716 | 37.57% |
| G | Difficult (both fields missing/unusable, or missing+weak) | 30,736 | 0.40% |

Interpretation: only 0.83% of the task is "trivially easy" (bucket A). A further 21.6%
(B+C) is solvable by an exact match on one field alone. The largest two buckets — D (36.4%) and
F (37.6%), together 74% of all true links — require genuine fuzzy/learned matching: D has
workable lexical overlap on both sides, but F does not clear the 0.5-Jaccard bar on *either*
field despite both being present, meaning naive lexical-similarity thresholds will not cleanly
separate these from non-matches without a stronger signal (candidate: numeric address tokens,
§12/§21). This 74% "hard middle" is the core of the matching problem.

## 14. Collision / Ambiguity Analysis

VERIFIED (`experiments/audit_cache/collision_drift_stats.json`), per source table:

| Table | Unique normalized name | Unique normalized address | Unique (name, address) pair |
|---|---:|---:|---:|
| train_source1 | 88.43% | 98.12% | 100.00% |
| train_source2 | 92.18% | 89.57% | 99.08% |
| train_source3 | 92.69% | 91.70% | 99.39% |
| test_source1 | 89.47% | 97.99% | 100.00% |
| test_source2 | 92.64% | 88.59% | 99.19% |
| test_source3 | 93.08% | 90.80% | 99.45% |

Name alone is a weak identifying key (7–12% of names collide within a source); joint
(name, address) is a near-perfect key (99–100% unique). Largest observed name-collision groups
are generic-name businesses (train examples: "primary care group" ×253, "physical therapy"
×458–460, "ear nose & throat group" ×251) and, in the test set, some very short/likely
abbreviated names in S2/S3 ("cc" ×302 in test_source2, "cc"/"sc"/"ac" ×297–387 in test_source3),
plus a France-specific pattern in test_source1 ("<city> club sarl" recurring ×147–205) — see §17.
This confirms candidate generation cannot rely on name matching alone without an address-based
(or other) disambiguation step for the ~10% collision-prone tail of names, and that source
tables are not internally deduplicated the way Source-1 is (S1's (name,address) pair is exactly
100% unique in both train and test; S2/S3 are 99.1–99.5%, i.e. a small but nonzero number of
near/exact duplicate records exist within S2/S3 themselves).

## 15. Truth-Conditioned Exact-Signal Coverage

**TRUTH-CONDITIONED TRAINING DIAGNOSTICS — NOT model validation scores, NOT deployable oracle
features.** VERIFIED, over 7,638,365 true links:

| Condition | Fraction of true links |
|---|---:|
| Name exact (only) | 15.80% |
| Address exact (only) | 7.44% |
| Both exact | 0.83% |
| Either exact | 22.41% |
| Numeric address tokens agree exactly | 63.49% |
| Neither name nor address exact | 77.59% |

A deterministic rule "flag as a match if name is exactly normalized-equal" would, on its own,
recover at most 15.8% of true matches before any consideration of whether that exact name is
unique enough to be safe (§14 shows normalized names are only 88–93% unique within a source, so
an exact-name-only rule would also need a disambiguation step even within that 15.8%). The
numeric-address-token-agreement signal (63.5%) covers far more of the true-link population than
either exact-text condition and is comparatively insensitive to language/transliteration
differences — the most promising deterministic-adjacent signal Phase 1 found.

## 16. Train/Test Drift

VERIFIED (`experiments/audit_cache/collision_drift_stats.json`): length, token-count, and
numeric-token-presence distributions are close between train and test for every shared source
table, at every quantile checked. Representative comparison (Source-1):

| Quantile | Name length: train / test | Address length: train / test |
|---|---|---|
| 5% | 12 / 12 | 27 / 28 |
| 25% | 18 / 18 | 33 / 36 |
| 50% (median) | 24 / 24 | 41 / 50 |
| 75% | 30 / 29 | 70 / 74 |
| 95% | 37 / 36 | 103 / 105 |

Address length drifts slightly upward in test (median 41→50 for S1), consistent with France's
test-only presence and its own address-length profile (§17) pulling the pooled test
distribution, not with a change in the US/India distributions themselves. Address-numeric-token
presence is materially unchanged (96.5%→95.9% for S1; ~90–93% throughout S2/S3, train and test
alike). No material unexplained drift was found beyond the expected France effect.

## 17. France Structural Analysis

**France exists only in test.** No external lookup, geocoding, or business-identity inference
was used — all figures are purely structural (lengths, token counts, uniqueness rates) computed
the same way as for US/India. VERIFIED counts:

| Table | France rows | % of test table |
|---|---:|---:|
| test_source1 | 259,452 | 14.98% |
| test_source2 | 703,378 | 14.39% |
| test_source3 | 731,615 | 14.40% |

Structural comparison (France vs US vs India, test set):

| Metric | France | India | US |
|---|---:|---:|---:|
| Name length (median, S1) | 19 | 27 | 22 |
| Address length (median, S1) | 48 | 76 | 34 |
| Name token count (median, S1) | 3 | 4 | 3 |
| % address has numeric token (S1) | 99.58% | 91.27% | 99.9998% |
| Missing-address rate (S2) | 3.06% | 2.28% | 2.94% |
| % normalized name unique within source (S1) | 88.65% | 88.10% | 91.15% |

**VERIFIED:** France's basic structural profile (name-uniqueness rate, missingness rate,
numeric-address-token presence) falls within the same range as US/India — no metric is a
wild outlier. **INFERRED:** France names run shorter than India's and closer to (slightly
shorter than) the US's, and France addresses are shorter than India's but noticeably longer
than the US's — a distinct but not extreme profile. **INFERRED (from collision examples in
§14):** France business names appear to include country-specific legal-form vocabulary
analogous to "Ltd"/"Pvt Ltd" in the other two countries (e.g. "SARL" recurring in a repeated
`<city> club sarl` collision group) — the same *class* of legal-suffix noise the task already
has to handle for US/India, not a structurally new noise type, though the specific vocabulary is
unseen in training. **UNKNOWN:** whether France's real name/address noise patterns (accented
characters, department/postal-code conventions, street-numbering conventions) transfer as well
as the coarse structural statistics suggest — Phase 1 cannot measure this without France ground
truth, which does not exist. This is the single largest unresolved generalization risk carried
into Phase 2.

## 18. Validation Leakage Findings

VERIFIED: zero target reuse across Source-1 entities (§8) means a per-entity split has no
leakage from shared match targets. No other leakage vector was identified in Phase 1 (no
duplicate Source-1 rows, no ground-truth anomalies that would let a validation entity's answer
be inferable from a development-set record). See `docs/VALIDATION_DESIGN.md` for the full
design and the frozen split.

## 19. Competitive Error Budget at 0.980473

The validation split (`validation_v1`, N=441,365) has a per-entity score granularity of
**1/441,365 ≈ 2.27 × 10⁻⁶** (see `experiments/splits/validation_v1_metadata.md`). Treating a
macro F0.5 of 0.980473 as if every entity's score were binary (0 or 1) — which real F0.5 scores
are not, but this gives an intuitive equivalent — a macro score of 0.980473 corresponds to
roughly **1.95% of entities being completely wrong** (1 − 0.980473 ≈ 0.0195). At N=441,365 that
is about **8,610 entities**. Real errors are typically partial-credit F0.5 values rather than
strict 0s, so the *actual* number of entities with some error is larger than 8,610 but each
error is individually smaller.

Given §13's finding that only 0.83% of true links are "trivially easy" and 74% require genuine
fuzzy/learned matching, a competitive solution in the 0.98+ region cannot rely on volume from
easy cases to absorb sloppy handling of the hard majority. Implications for later phases:
- **Singleton precision is high-leverage but not the dominant lever** (§5): singletons are only
  5.6% of entities, so perfect singleton handling alone caps achievable score around 0.944 if
  every non-singleton scored zero; the score budget is won or lost mostly on non-singletons.
- **False merges are the most expensive failure mode.** The metric weights precision 2× over
  recall (F0.5); combined with 88–93% name-collision rates (§14), a matcher that trusts name
  similarity without address/numeric-token corroboration risks systematic false merges across
  the many generic-name collision groups (§14 examples).
- **Blocking misses on the 10% zero-name-overlap tail (§11) are unrecoverable by any matcher
  downstream** — candidate generation must not rely solely on name-token overlap.
- **Country-specific brittleness on France (§17) is a live risk** given it is 15% of every test
  source table — a country-conditioned model or hard country filter that only works for
  US/India threatens roughly one in seven test entities.

## 20. Phase-2 Implications

Candidate generation (blocking) is the next hard requirement: the full cross product is
infeasible (§2), name-token blocking alone would miss ~10% of true links (§11), and country
cannot yet be safely used as a hard filter (§9, §17 — 100% consistent on train but unverified,
by construction, for the unseen France test country). Phase 2 should measure candidate recall
under blocking strategies that combine name-token overlap, address numeric-token overlap (§12,
the strongest single signal found), and soft (not hard) country agreement, before any modeling
decision is made.

## 21. Path to >0.980473 — Evidence-Based Requirements

Every statement below is tied to a measured Phase-1 finding; none is a guarantee.

- **Candidate recall requirement:** blocking must recover close to 100% of true links across
  the full match-multiplicity distribution (§6 — nearly half of entities need 4+ correct
  targets), not just top-1 nearest neighbor. A name-only blocking key will structurally cap
  recall around 90% (§11's zero-overlap tail); the address numeric-token signal (§12, 63.5%
  exact agreement on true links) is evidence-backed as a second, largely orthogonal blocking key
  worth combining with name-token blocking.
- **Singleton precision requirement:** given the 2×-precision-weighted metric and 5.58%
  singleton rate (§5), the matcher needs a confident "no match" decision path, not just a
  similarity threshold — the joint-bucket analysis (§13) shows a clean similarity cutoff would
  have to separate bucket D/F (74% of true links, often only moderate similarity) from
  non-matches without also swallowing singletons that happen to resemble something in D/F.
- **Easy-case deterministic-matching potential is small but real:** buckets A+B+C (§13, 22.4%
  of true links) are recoverable by high-confidence exact-field rules *if* combined with the
  uniqueness check from §14 (an exact name match is only safe when that normalized name is
  unique enough in the target source, which holds for ~88–93% of names but not all).
- **Difficult-tail treatment is unavoidable:** 74% of true links (buckets D+F, §13) will not be
  solved by lexical thresholds alone; some combination of the numeric-address-token signal
  (§12), possibly weighted token similarity, and a learned decision boundary appears necessary
  based on the data's actual difficulty distribution — this is INFERRED from the difficulty
  distribution, not a proven requirement.
- **Whether heavy ML is justified:** the size and shape of the "hard middle" (D+F buckets) is
  consistent with needing *some* learned combination of several weak-to-moderate signals (name
  similarity, address similarity, numeric-token overlap, and possibly length/structure
  features) rather than one dominant handcrafted rule — but nothing in Phase 1 indicates that a
  large neural/semantic model is *necessary* rather than a calibrated combination of lexical
  features. Per the operating principle against "fanciest model = best," Phase 2's candidate
  generation and a lexical-feature baseline should be evaluated before considering embedding
  models.
- **Source-specific behavior:** S2 and S3 have nearly identical structural statistics
  throughout this audit (missingness, collision rates, length distributions all within ~1–2
  percentage points of each other); no evidence was found that S2 and S3 need materially
  different treatment, beyond tracking them as separate ID namespaces.
- **Country handling:** country is a perfect signal on training true links (§9) but its
  reliability on France is UNKNOWN (§17) since France has no training examples at all. Using
  country as a hard filter is a real risk for the 14–15% of every test table that is French;
  Phase 2 should measure candidate recall both with and without a country filter before deciding.
- **France transfer risk is the single largest open question** this audit could not resolve
  (§17): structural statistics transfer cleanly, but nothing here can verify whether learned
  name/address similarity behavior transfers, since no France ground truth exists anywhere.
