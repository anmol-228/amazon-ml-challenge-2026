# Phase 2 — Blocking / Candidate Generation

STATUS: **COMPLETE**, with a Phase 2.5 competitive-ceiling addendum (see
section 25) that supersedes sections 1/12/13/18/19/20/23 for architecture
selection purposes — the five-route architecture in section 25, not the
four-route architecture in section 19, is the final one. All numbers below
are read directly from the JSON result files under `experiments/blocking/`,
computed on the `development` partition of `validation_v1`. The frozen
`validation` partition was never touched during Phase 2 or Phase 2.5.

## 1. Executive Result

The selected candidate architecture is a four-route union — deterministic
exact-match sanity route, forward word-TF-IDF top-k=50, reverse word-TF-IDF
top-r=5, and exact numeric-address-token-signature matching — combined via
a disk-backed, bounded-memory partitioned union (never a single in-memory
`concat`+`groupby` over the full candidate set).

On `development` (1,765,456 Source-1 entities, 6,110,820 true links):

- **Link recall: 0.8914**
- **Complete true-link coverage (non-singleton): 0.7465**
- **Singleton candidate exposure rate: 0.9944**
- **Total candidate edges: 228,440,608** (global reduction ratio 0.9999875)
- **Overall candidate load:** mean 129.4, median 56, p95 493, p99 657, max 3361

A conditional targeted-rescue experiment (EXP-B005, a re-tuned numeric
rare-token route run in isolation) was executed and **rejected on measured
evidence**: it added only 4,146 true links (0.07% of total truth) despite
generating 80.0M additional candidate rows. Character n-gram retrieval
(EXP-B002) showed a genuinely strong pilot signal (74.4% rescue of
EXP-B001's own residual) but is **deferred, not adopted**, because its
true full-`development`-scale cost is judged substantially higher than a
naive linear extrapolation suggests and was not independently verified at
that scale within this phase's time budget — see section 17 and the
Phase-3 handoff.

## 2. Entry State / Constraints

Phase 0/1/1.5 complete per `PROJECT_STATE.md`. Phase 2 executed the
EXP-B000→B005 sequence from `docs/research/RESEARCH_TO_EXPERIMENT_PLAN.md`,
with two review checkpoints that added a per-source forward retrieval
ablation, a verified (not assumed) address-rescue diagnostic, and a
scalable metrics/union rewrite once several routes' candidate tables
reached tens to hundreds of millions of rows. No `validation_v1`
`validation`-partition touch occurred anywhere in Phase 2. No test data
was read. No commit, push, or leaderboard submission occurred.

## 3. Environment / Resource Budget

- Machine: 12 logical cores, 33,946,222,592 bytes (~31.6 GB) total physical
  RAM, no GPU used. W: drive had 129 GB free at the start of Phase 2.
- Python 3.12.10, virtual environment at `.venv/`.
- Dependencies added this phase, all permissive-licensed tooling (not the
  final matching model itself): `scikit-learn` 1.9.1 (BSD-3, TF-IDF
  vectorization), `scipy` 1.18.1 (BSD-3, sparse matrix operations),
  `psutil` 7.2.2 (BSD-3, OS-level RSS/CPU monitoring), plus their own
  transitive dependencies (`joblib`, `threadpoolctl`, `narwhals`,
  `cloudpickle`, all BSD/MIT). `pandas` 3.0.6 and `pyarrow` 25.0.1 were
  already present from Phase 1.
- Final disk footprint of all Phase-2 experiment artifacts:
  **9.7 GB** under `experiments/blocking/` (largest: B004's union table at
  ~4.4 GB, B003's numeric candidate tables at ~1.3 GB).

## 4. TF-IDF Fitting Policy

Frozen for this Phase-2 round (see
`student_resource/code/business_entity_resolution/src/blocking_vectorization.py`
module docstring for the authoritative version):

1. Vocabulary/IDF corpus = normalized joint (name + address) text of ALL
   train Source-2 rows, ALL train Source-3 rows, and train Source-1 rows
   restricted to the `development` split only.
2. S1/S2/S3 fit **jointly** into one shared word-unigram vectorizer (not
   three separate per-source vectorizers); S2 and S3 share the same fitted
   statistics.
3. No country partitioning at the fitting stage — one global fit; this
   also sidesteps the open-set France categorical problem entirely at this
   stage (Phase 2 never touches test data, France or otherwise).
4. EXP-B001 (forward) and EXP-B001R (reverse) use the **same** fitted
   vectorizer/matrices, so direction is isolated from representation.
5. EXP-B002 (character n-gram) used a separately-fit vectorizer with its
   own provenance record (`char_provenance` in `B002/b002_results.json`),
   fit only on its own pilot corpus, never silently compared against B001
   as if sharing statistics.
6. No test-derived, validation-derived, or hidden-label-derived statistic
   is used anywhere in the fitting pipeline.
7. **Tuning correction (measured, not guessed):** the vectorizer was
   initially fit with `max_df=0.02`, which B000's own retrieval pilot
   showed made full-`development`-scale forward retrieval infeasible
   (~654 minutes estimated). Retuned to `max_df=0.0008, min_df=3` based on
   the measured document-frequency distribution, which reduced the
   estimated forward retrieval time to ~11.5 minutes — a 57x measured
   speedup at this specific corpus scale, with negligible recall cost
   (verified: only 0.81% of dev-S1 rows and 2.35% of target rows became
   empty vectors under the stricter threshold).
8. **Rationale for reading S2/S3 in full:** S2 and S3 have no `split`
   property — only Source-1 entities are split by `validation_v1`.
   Restricting the target universe to a subset would not reflect
   production reality (there is no "which split" label on a target row at
   inference time) and would require consulting ground truth to decide
   which rows to exclude — worse than reading all of it unlabeled. What
   *is* withheld from every fitting step is `validation`-split Source-1
   text.

## 5. Candidate Identity / Provenance Contract

Candidate identity = `(s1_entity_id, target_source, target_entity_id)` per
`src/blocking_identity.py`. `target_source` is derived from the
entity_id's own `S2-`/`S3-` prefix, never trusted from route metadata;
`assert_no_cross_source_collision` makes "S2 and S3 IDs never collide" an
enforced, tested invariant (`tests/test_blocking_infrastructure.py`).
Route provenance (`route_*` boolean columns, `*_rank`/`*_score` numeric
columns) is preserved through every union operation via a fixed,
tested aggregation convention (OR / min / max respectively) shared by both
the simple in-memory union and the disk-backed partitioned union
(`_aggregate_union_frame` in `src/blocking_union.py`), verified identical
on synthetic data (`test_union_candidates_partitioned_matches_simple`).

## 6. EXP-B000 Preflight

Two rounds. **Round 1** (`max_df=0.02`): word vectorizer fit succeeded
(vocabulary 773,377) but a 3,000-query retrieval pilot took 66.6s
(22ms/query), extrapolating to **654 minutes** for the full forward pass
and **3,821 minutes** for reverse — both marked **BLOCKED**. **Round 2**
(`max_df=0.0008, min_df=3`): the same pilot took 1.18s (0.4ms/query),
extrapolating to **11.5 minutes** forward (**GO**) and **67.5 minutes**
reverse (**GO WITH MODIFICATIONS**). Character n-gram and numeric routes
were marked **GO** based on their own pilot-scale measurements (numeric
postings measured directly and cheaply at full scale: 90,438 distinct
tokens, p99 collision-group size 1,824, max 763,372).

## 7. Deterministic Sanity Route

Normalized (name, address) exact equality: **50,437** candidate rows,
link recall **0.83%** — matching the Phase-1 audit's independently
measured "0.83% of true links exact on both fields" figure exactly, a
correctness cross-check that passed. Normalized-name-only exact equality
was also measured for its own collision-size sanity check: **12,677,375**
rows (essentially every dev-S1 row got at least one name-only match) —
confirmed far too coarse to use as a production candidate source and
excluded from the final union; only the (name, address) relationship
contributes to `candidate_pairs.tsv`.

## 8. EXP-B001 Forward Word Retrieval

Full k-grid on `development`, one retrieval pass at k=50 with smaller grid
points derived by rank-filtering (no redundant re-retrieval):

| k | link_recall | macro_candidate_recall | complete_coverage | singleton_exposure | edges |
|---|---|---|---|---|---|
| 5 | 0.6990 | — | 0.4244 | 0.9908 | 8,752,184 |
| 10 | 0.7774 | — | 0.5609 | 0.9908 | 17,490,031 |
| 20 | 0.8113 | — | 0.6170 | 0.9908 | 34,942,041 |
| 50 | **0.8467** | 0.8911 (union value) | **0.6752** | 0.9908 | 87,161,478 |

Resource (k=50 pass): 482.3s wall, peak RSS 19,736 MB.

**Zero-name-overlap diagnostic (verified, review-checkpoint item 5):** at
k=50, nonzero-name-token-overlap true links reach recall **0.8728**
(close to the ~90% structural ceiling implied by the Phase-1 audit for
that population, i.e. tokenization/normalization is not silently losing
recall there). Zero-name-token-overlap true links (878,738 links across
594,586 S1 entities, ≈10% tail) reach recall **0.6910**. Of the 607,231
zero-name-overlap links recovered at k=50, **all 607,231 (100%) were
directly verified** to have a genuine nonempty normalized-ADDRESS-token
intersection — 0 cases of the theoretically-possible cross-field rescue
mechanism (S1 address matching target NAME or vice versa) were found. The
original diagnostic label ("address_rescued") was numerically correct;
this is now proven, not assumed.

## 9. EXP-B001R Reverse Retrieval

Full r-grid, forward(k=50) ∪ reverse(r), computed via the scalable
join/groupby metrics path (not a materialized union table):

| r | reverse_only_recall | union_recall | union_coverage | new_links_rescued | newly_complete |
|---|---|---|---|---|---|
| 1 | 0.7677 | 0.8634 | 0.7024 | 102,201 | 45,427 |
| 2 | 0.8005 | 0.8685 | 0.7106 | 133,052 | 59,146 |
| 3 | 0.8162 | 0.8721 | 0.7165 | 154,979 | 68,840 |
| 5 | **0.8344** | **0.8775** | **0.7252** | **188,431** | **83,490** |

Resource: 353.6s wall, peak RSS 16,670 MB, 49,433,565 candidate rows.
r=5 selected (best measured point, largest grid value tested).

## 10. EXP-B002 Character Retrieval

**Pilot-first, per the mission's own gate.** Query population: the
207,130 dev-S1 entities with ≥1 residual true link after EXP-B001 (zero
name-token overlap, not address-rescued) — an evidence-driven pilot
selection, not an arbitrary sample. Target pool: 660,733 ids (271,507
guaranteed-positive residual targets + a 400,000-id random background
sample).

- Residual true links: 271,507.
- **Rescued by char-trigram retrieval: 202,061 (74.42%).**
- Pilot link recall with char: 0.7567 vs. without: 0.5129 (within this
  bounded pilot population).
- Resource: fit 47.5s / peak 20,716.7 MB; retrieval 1,413.7s / peak
  21,340.0 MB (both memory-safe after a fix — see section 22).
- Naive full-`development`-scale linear extrapolation: **200.8 minutes**.

**Status: SELECTED_CONDITIONAL_FULL_SCALE_RECOMMENDED** by the pilot's own
threshold rule, but **not promoted to the canonical B004 union in this
Phase-2 round.** The 200.8-minute estimate extrapolates only the
per-query retrieval cost from a 660,733-target pilot pool; a genuine
full-`development` run would need (a) a character-trigram vectorizer
refit over the full ~12M-document corpus (not the 867,863-document pilot
corpus) and (b) a full ~10.32M-row target matrix (not the 660,733-row
pilot pool) — both materially more expensive than this pilot's own
measurement accounts for, and neither was independently verified at true
scale within this phase. Recommended instead as a Phase-3 candidate for a
targeted, **observable**-trigger-gated rescue (e.g. applied only to S1
entities whose best current word-route similarity score is low), which
would keep its cost close to what this pilot itself measured rather than
the unverified full-scale cost. This is not ground-truth gating — see
section 20 for why "residual" cannot be the production trigger.

## 11. EXP-B003 Numeric / Address Retrieval

Two routes, each with independent provenance:

- **Route A (exact numeric-set signature):** 117,706,510 candidate rows
  (pruned at a measured, unconditional signature-group-size cutoff of 650,
  informed by measured p99=65/max=59,180 — the max, not the tame p99, was
  the actual risk). Standalone link recall **0.3320**.
- **Route B (rare individual numeric token):** 18,030,637 candidate rows
  at a memory-pressure-forced `max_posting_size=200` (target: 1,000;
  measured token collision p50=4, p99=1,824, max=763,372). Standalone
  link recall **0.0862**.
- **A ∪ B:** link recall **0.3453**; A contributes 1,583,243 unique links
  over B, B contributes 81,253 unique links over A.
- **Incremental value beyond the word forward(k=50)∪reverse(r=5) union:**
  A adds 84,519 links / 35,405 newly-complete entities; B adds only 5,000
  links / 2,739 newly-complete; A∪B together add 85,619 links / 36,102
  newly-complete (≈1.4% of total truth).

**Discrepancy noted, not silently reconciled:** Route A's measured
standalone recall (33.2%) is notably lower than the Phase-1 audit's
"63.49% of true links have exact numeric-token-set agreement" figure. This
project's `numeric_signature()` (sorted, deduplicated numeric-token tuple)
and the Phase-1 audit's own numeric-token-set-agreement measurement are
not necessarily computed identically, and this was not re-verified
against the Phase-1 audit's exact method within this phase — flagged as
an open discrepancy for a future pass, not resolved here.

**Selection:** Route A included in the final B004 union (positive,
if modest, contribution). Route B **excluded** — see section 17 for the
subsequent EXP-B005 experiment that re-tested a better-resourced version
of Route B and confirmed rejection on stronger evidence.

## 12. EXP-B004 Union

Final selected architecture: exact(name+address) ∪ forward-word(k=50) ∪
reverse-word(r=5) ∪ numeric-exact-set(Route A), computed via
`union_candidates_partitioned` (128 hash buckets on `s1_entity_id`,
disk-backed, never a full in-memory concat+groupby).

- **Total input rows across 4 routes (pre-dedup): 254,351,990.**
- **Final unioned candidate rows (post-dedup): 228,440,608.**
- Union construction: 601.0s wall, peak RSS 25,380 MB.

**Authoritative metrics:**

| Metric | Value |
|---|---|
| Link recall | 0.8914 |
| Macro per-entity candidate recall (non-singleton) | 0.8911 |
| Complete true-link coverage (non-singleton) | 0.7465 |
| Singleton candidate exposure rate | 0.9944 |
| Singleton candidate load (mean/median/p95/p99/max) | 130.6 / 56 / 493 / 660 / 2,663 |
| Overall candidate load (mean/median/p95/p99/max) | 129.4 / 56 / 493 / 657 / 3,361 |
| Total candidate edges | 228,440,608 |
| Global reduction ratio | 0.9999875 |
| Routing-aware reduction ratio | Not computed (no hard country routing used) |
| n_true_singletons / n_true_non_singletons | 98,598 / 1,666,858 |

## 13. Route Incremental Value

Processing order exact → forward → reverse → numeric (each row shows what
that route added over everything already unioned before it):

| Route | Standalone recall | Incremental new links | Incremental newly-complete | Edges in route |
|---|---|---|---|---|
| exact (name+address) | 0.0083 | 50,437 | 807 | 50,437 |
| forward_word (k=50) | 0.8467 | 5,124,278 | 1,124,626 | 87,161,478 |
| reverse_word (r=5) | 0.8344 | 188,292 | 83,479 | 49,433,565 |
| numeric_exact_set | 0.3320 | 84,391 | 35,399 | 117,706,510 |

(Sum of incremental new links = 5,447,398 ≈ 0.8914 × 6,110,820, consistent
with the reported union recall.) Forward word retrieval is overwhelmingly
the dominant contributor; reverse and numeric each add a real but much
smaller increment, at proportionally higher edge cost per link rescued —
the numeric route in particular costs 117.7M edges for 84,391 links
(≈1,395 edges per rescued link) versus forward's 87.2M edges for 5.12M
links (≈17 edges per rescued link).

## 14. Candidate Load / Reduction Ratios

See section 12's table. Median load (56) is comfortable for a downstream
GBDT-scale matcher; the tail (p99=657, max=3,361) is driven by generic,
high-frequency names/addresses and numeric signatures shared across many
records (the same collision-group phenomenon measured in EXP-B000/B003).
Global reduction ratio 0.9999875 reflects the astronomically large
unrestricted comparison space (|S1| × (|S2|+|S3|) ≈ 1.82 × 10¹³); no hard
country-based routing was used in the final architecture (DEC-017 remains
open — country was not adopted as a hard filter, only implicitly present
in the text the joint vectorizer already sees), so no separate
routing-aware ratio applies.

## 15. Diagnostic Slices

| Slice | n_entities | link_recall | complete_coverage | singleton_exposure |
|---|---|---|---|---|
| country=India | 706,550 | 0.8738 | 0.7120 | 0.9921 |
| country=US | 1,058,906 | 0.9032 | 0.7695 | 0.9960 |
| match_bucket=0 (singletons) | 98,598 | n/a | n/a | 0.9944 |
| match_bucket=1 | 95,325 | 0.8861 | 0.8861 | n/a |
| match_bucket=2+ | 1,571,533 | 0.8915 | 0.7380 | n/a |
| address_present | 1,765,456 (all) | 0.8914 | 0.7465 | 0.9944 |

`address_missing` is empty by construction — the Phase-1 audit's finding
that Source-1 addresses are never missing (only S2/S3 have missingness)
holds exactly. **US outperforms India** on both recall (+2.94pp) and
coverage (+5.75pp) — a real, measured gap worth Phase-3 attention (see
section 16's country breakdown of the residual).

## 16. Residual Blocking Miss Analysis

663,422 / 6,110,820 true links missed (**10.86%**), computed via the
scalable join-based `link_found_mask` against the full 228.4M-row union
(never a per-entity Python set over that table). Per-pair observable
features computed only for this bounded residual set:

| Category | Count | % of missed |
|---|---|---|
| numeric_agreement_but_still_missed | 368,300 | 55.5% |
| near_miss_nonzero_word_similarity | 381,346 | 57.5% |
| wildly_dissimilar_zero_word_similarity | 282,076 | 42.5% |
| zero_name_overlap | 187,897 | 28.3% |
| zero_address_overlap | 139,292 | 21.0% |
| target_address_missing | 138,596 | 20.9% |
| numeric_disagreement | 102,851 | 15.5% |
| short_s1_name_lte_2_tokens | 100,087 | 15.1% |
| no_numeric_evidence_either_side | 7,829 | 1.2% |
| s1_address_missing | 0 | 0.0% |

Word-cosine-similarity stats over the residual: mean 0.2596, p50 0.2722,
p90 0.5976, max 1.0 — the majority of misses are not lexically
"invisible," they simply rank outside the tested k=50/r=5 windows or lose
to competing candidates.

- **41,976 S1 entities are fully missed** (0 of their true targets found);
  **380,571 are only partially missed** — 90% of affected entities already
  have *some* correct candidates, meaning the practical problem is mostly
  "a few more siblings," not "unreachable from scratch."
- By target source: S3 (362,450) misses more than S2 (300,972).
- By country: US (354,495) and India (308,927) are closer in absolute
  missed-link count than the recall-slice gap in section 15 suggests,
  because US has more total dev entities.
- By match multiplicity: 98.4% of missed links (652,565 / 663,422) belong
  to multi-match (2+) entities — structurally expected, since those
  entities need every one of several targets simultaneously for complete
  coverage.

**No embedding/semantic escalation was pursued.** Per the mission's own
gate, an observable, lexical-signal-based rescue was tested first (section
17) before any consideration of a heavier method.

## 17. EXP-B005

**Run, not skipped** — the residual showed a large (55.5%), fully
observable candidate category ("numeric agreement but missed") with an
obvious, low-complexity candidate mechanism already built (Route B,
excluded from B004). Re-ran Route B **in isolation** (not immediately
after Route A's large in-memory generation) with a properly
measurement-informed target cutoff (2,000, just above the measured
p99=1,824).

- Generation still hit `MemoryError` at cutoff=2,000 even in isolation and
  fell back to **500** — confirming the constraint is a genuine scale
  property of this route (the single most common numeric token has
  763,372 postings), not merely a memory-pressure artifact of running
  after Route A.
- 80,049,759 candidate rows generated. Standalone link recall 0.1634.
- **Incremental value over the current B004 union: only 4,146 new true
  links (0.068% of total truth) and 2,510 newly-complete entities**, for
  80.0M additional candidate edges.
- Of the specific 368,300-link "numeric agreement but missed" residual
  category, this route rescues only **4,146 (1.13%)**.

**Decision: REJECTED_INSUFFICIENT_VALUE.** The "numeric agreement" widely
observed in the residual is dominated by generic, high-frequency numbers
(unit numbers, common house numbers) whose posting lists exceed any
responsible pruning threshold — a rare-token-based blocking route
structurally cannot reach this population; it is better addressed, if at
all, as a downstream matcher **feature** (numeric agreement/conflict as a
signal, not a blocking gate), consistent with the mission's own §19.1
guidance that numeric disagreement/agreement is primarily matcher-stage
evidence. No second EXP-B005 mechanism was attempted — the character
n-gram alternative was deliberately deferred (section 10) rather than
rushed into an under-verified full-scale B005 run within this phase's
remaining time budget.

## 18. Final Pareto Selection

Non-dominated single-route configurations measured: exact (cheapest,
negligible recall alone), forward-k=50 (0.8467 recall / 87.2M edges),
forward∪reverse (0.8775 / 118.99M — wait, forward∪reverse edges from
B001R's r=5 row: 89,537,048 at r=1 up to 118,999,722 at r=5), and the full
B004 union (0.8914 / 228.4M edges). Each additional route strictly
increased both recall/coverage and edge cost — no route was found to hurt
recall, and none was excluded from the final union for that reason. The
exclusions (numeric Route B, char n-gram, per-source retrieval) were each
excluded specifically because their **measured incremental value did not
justify their measured cost**, not because of a blanket policy:

- Per-source retrieval: at matched candidate-edge budget (not matched
  nominal k), no material advantage over global top-k (e.g. global k=10 at
  17.49M edges: recall 0.7774; per-source k=5, also 17.49M edges: recall
  0.7751 — global very slightly ahead). **Rejected.**
- Numeric Route B: ≤5,000-link incremental value beyond word routes in
  EXP-B003, confirmed via a second, better-resourced attempt in EXP-B005
  (4,146 links against the fuller B004 baseline) at a cost of tens of
  millions of edges each time. **Rejected twice, independently.**
- Char n-gram: strong pilot signal, but promotion deferred pending
  verified full-scale cost, not rejected on value. **Deferred to Phase 3.**

The selected B004 architecture (exact + forward-k50 + reverse-r5 +
numeric-Route-A) is the practical frontier this phase reached: every
component route earns its place with positive, individually-measured
incremental value, and the two components with genuinely poor value
(numeric Route B, confirmed twice) are excluded.

## 19. Selected Candidate Architecture

**Routes (all union, never intersection):**
1. Deterministic exact-match: normalized (business_name, business_address)
   equality. No parameters.
2. Forward word-TF-IDF: top-k=50 per S1 entity over the combined S2+S3
   target universe, shared fitted vectorizer per section 4.
3. Reverse word-TF-IDF: top-r=5 per S2/S3 target over the S1 development
   universe, same fitted vectorizer, inverted to S1-indexed edges.
4. Numeric exact-signature: candidates for every (S1, target) pair sharing
   an identical non-empty sorted numeric-token signature, pruned at a
   measured group-size cutoff of 650.

**Union mechanism:** `union_candidates_partitioned` (128 s1-hash buckets,
disk-backed scatter/gather, `_aggregate_union_frame` aggregation: route
flags → logical OR, rank fields → minimum, score fields → maximum).

**Candidate schema:** `(s1_entity_id, target_source, target_entity_id)`
plus every contributing route's own provenance columns
(`route_exact_name_address`, `route_forward_word`/`forward_rank`/
`forward_score`, `route_reverse_word`/`reverse_rank`/`reverse_score`,
`route_numeric_exact_set`).

**Artifact:** `experiments/blocking/B004/union_candidates.parquet`
(228,440,608 rows, ~4.4 GB).

## 20. Selected / Rejected / Deferred Route Status

| Route | Status |
|---|---|
| Exact (name+address) | **SELECTED** |
| Exact (name-only) | REJECTED (too coarse — 12.68M rows, no specificity) |
| Forward word (global, k=50) | **SELECTED** |
| Forward word (per-source) | REJECTED (no material advantage at matched budget) |
| Reverse word (r=5) | **SELECTED** |
| Numeric Route A (exact-set signature) | **SELECTED** |
| Numeric Route B (rare token, v1 and v2/EXP-B005) | REJECTED (confirmed twice: negligible incremental value at any responsibly-resourced cutoff) |
| Character n-gram (EXP-B002) | DEFERRED (strong pilot evidence, unverified full-scale cost — Phase-3 candidate) |

## 21. Scale / Runtime / RAM / Disk

| Step | Wall time | Peak RSS | Output size |
|---|---|---|---|
| EXP-B000 preflight (both rounds) | ~7 min each | ~9.5–11.9 GB | cache: 526 MB |
| Exact sanity route | 108.6s | 14.4 GB | 130 MB |
| EXP-B001 forward (k=50) | 482.3s | 19.7 GB | 1,018 MB (B001 dir) |
| EXP-B001R reverse (r=5) + scalable r-grid metrics | 353.6s retrieval + separate metrics pass | 16.7 GB (retrieval); 20.5 GB (metrics) | 767 MB |
| Per-source ablation (S1→S2, S1→S3, k=20 each) | 210.1s + 200.1s | 12.8 / 14.7 GB | 843 MB |
| EXP-B002 char pilot | fit 47.5s + retrieval 1,413.7s | 21.3 GB | 154 MB |
| EXP-B003 (both routes + cross-route metrics) | ~250s generation | 20.5 GB | 1.3 GB |
| EXP-B004 union (254.4M→228.4M rows) | 601.0s | 25.4 GB | 4.4 GB |
| Residual analysis | 208.4s (mask) + missed-set analysis | 27.5 GB | 14 MB |
| EXP-B005 | 90.8s generation + metrics | 28.5 GB | 727 MB |
| **Total experiment artifact disk** | — | — | **9.7 GB** |

Peak RSS across the entire phase never exceeded ~28.5 GB against a ~31.6 GB
machine — closest margins occurred in EXP-B004/B005/residual analysis,
each involving the full 228M-row union table; all completed successfully
after the memory-safety fixes in section 22.

## 22. Known Limitations

- **Numeric Route A recall discrepancy** (33.2% measured vs. Phase-1's
  63.49% figure) is flagged, not resolved (section 11) — the two
  measurements may use subtly different numeric-token-set-agreement
  definitions.
- **Character n-gram's true full-scale cost is unverified** — the 200.8-
  minute pilot extrapolation almost certainly understates a genuine
  full-corpus-refit, full-target-pool run.
- **No hard country routing** was implemented or evaluated as a candidate
  architecture change in Phase 2 (DEC-017 remains open); country only
  enters implicitly through the shared joint-text vectorizer.
- **France is entirely untested** by construction (zero training examples,
  per DEC/Phase-1 findings) — nothing in Phase 2 changes this; the
  selected architecture (lexical + numeric union, no hard country filter)
  does not structurally exclude France, but its behavior there is unknown
  until test-time evaluation (never performed in Phase 2).
- **B005's numeric rescue attempt hit `MemoryError` even at a
  measurement-informed cutoff** (2,000, just above measured p99) — the
  practical safe ceiling for this specific route's naive
  posting-list-based generation is closer to 500 on this machine. A more
  memory-efficient algorithm (e.g. an inverted-index approach with
  streaming candidate emission) could likely push this higher, but given
  the route's poor measured value, this was not pursued further.
- **`routing_aware_reduction_ratio` is not computed** (no hard routing
  partition exists in the selected architecture).

## 23. Phase-3 Handoff

**Candidate artifact:** `experiments/blocking/B004/union_candidates.parquet`
(228,440,608 rows). Schema: `s1_entity_id`, `target_source`,
`target_entity_id`, `route_exact_name_address`, `route_forward_word`,
`forward_rank`, `forward_score`, `route_reverse_word`, `reverse_rank`,
`reverse_score`, `route_numeric_exact_set` (booleans default False where a
route did not contribute; rank/score columns are null where their route
did not contribute).

**Blocking recall ceiling (measured, `development`):** link recall 0.8914,
complete true-link coverage (non-singleton) 0.7465. This is the hard
upper bound any Phase-3 matcher can reach on this candidate set — the
mission's own principle ("if a true link does not enter the candidate
set, no later matcher can recover it") applies directly: **10.86% of true
links are structurally unreachable by Phase 3 unless the candidate set is
revisited** (e.g. by implementing the deferred char n-gram route as a
targeted, observable-trigger-gated addition).

**Recommended Phase-3 feature starting set**, supported by this phase's
own evidence:
- `forward_score` / `reverse_score` (already-computed cosine similarities)
  as direct matcher features, not just retrieval ranks.
- A numeric-agreement/conflict flag (not a filter) per candidate pair —
  section 17's finding that numeric agreement is common but not
  discriminative for *blocking* strongly suggests it is more useful as a
  *matching*-stage feature.
- Country-agreement (S1 country == target country), not raw country
  identity, per the existing France open-set policy (DEC-017 and
  `docs/research/RESEARCH_TO_EXPERIMENT_PLAN.md`'s cross-cutting France
  policy) — not evaluated in Phase 2, carried forward as a design
  constraint.
- Given the residual's own composition (55.5% "numeric agreement but
  missed", 57.5% "near-miss nonzero word similarity"), a Phase-3 matcher
  trained on the current candidate set already has strong lexical/numeric
  signal to work with among the candidates it *does* see; the residual
  problem is primarily one of blocking *recall*, not blocking *precision*.

**Remaining miss categories** (for Phase-3 awareness, not Phase-3 action):
see section 16's full table. The single largest, most actionable
follow-up is the deferred char n-gram route (section 10), recommended as
the first Phase-3-adjacent blocking investigation if additional recall is
required before matcher-stage work begins.

## 24. Exact Reproduction Commands

```bash
# from student_resource/code/business_entity_resolution/
.venv/Scripts/python.exe scripts/run_b000_preflight.py
.venv/Scripts/python.exe scripts/run_exact_sanity_route.py
.venv/Scripts/python.exe scripts/run_b001_b001r_word.py
.venv/Scripts/python.exe scripts/fix_b001r_metrics.py
.venv/Scripts/python.exe scripts/run_b001_per_source_ablation.py
.venv/Scripts/python.exe scripts/fix_b001_address_rescue_label.py
.venv/Scripts/python.exe scripts/run_b002_char_pilot.py
.venv/Scripts/python.exe scripts/run_b003_numeric.py
.venv/Scripts/python.exe scripts/run_b004_union.py
.venv/Scripts/python.exe scripts/run_residual_analysis.py
.venv/Scripts/python.exe scripts/run_b005_numeric_rescue.py
```

All scripts are deterministic given the frozen `validation_v1` split and
the fixed seeds recorded in each script; re-running reproduces the same
candidate artifacts and metrics. Test suite:
`cd student_resource/code/business_entity_resolution && .venv/Scripts/python.exe -m pytest tests/ -q`
(51/51 passing at the time of this report).

---

## 25. PHASE 2.5 ADDENDUM — Competitive-Ceiling Gate and Character-Retrieval Rescue

**Trigger:** before starting Phase 3, a narrow extension checked whether
the Phase-2 B004 blocker structurally caps the achievable Amazon macro
F0.5 below a competitive reference score (~0.980473, used as an
engineering reference, never an official threshold). This section
supersedes sections 1, 12, 13, 18, 19, 20, and 23 above for architecture
selection: **the five-route union in §25.7, not the four-route union in
§19, is the final selected Phase-2 architecture.** B004 itself
(`experiments/blocking/B004/union_candidates.parquet`) was never modified
or deleted and remains available as the prior canonical checkpoint.

### 25.1 Exact B004 Oracle F0.5

For every dev S1 entity, oracle prediction = truth ∩ B004 candidates (zero
false positives by construction: singletons always score 1.0 under an
oracle since the oracle prediction is empty whenever truth is empty; for
non-singletons, oracle precision is always 1.0, so
F0.5 = 1.25·recall/(0.25+recall)). Run through the actual frozen evaluator
(`src/evaluation.py::evaluate`, DEC-010), not a reimplementation.

- **ORACLE_B004_MACRO_F0.5 = 0.950159.**
- Independently verified two ways: (1) a closed-form arithmetic identity
  derived from the properties above, matching the evaluator's output to
  full float precision (`closed_form_matches_evaluator: true`); (2) a tiny
  synthetic test (3 entities, hand-computed expected score), which also
  matched exactly.
- Breakdown: singleton_accuracy=1.0; non_singleton_macro_f_beta=0.9472;
  by country: India=0.9410, US=0.9563; by match_bucket: singletons=1.0,
  single-match=0.8861, multi-match=0.9509; by source involvement:
  S2-only=0.9208, S3-only=0.9148, both=0.9523, singleton=1.0; by complete
  coverage (non-singleton): incomplete=0.7918, complete=1.0 (tautological
  under the oracle definition).

### 25.2 Competitive Gate

0.950159 is below the competitive reference (0.980473) and well below the
0.985 engineering headroom target — a ~3.03-point gap. **Gate result:
CONTINUE the extension** (not a stop-and-document case).

### 25.3 Character Targeting Analysis (Part D)

Tested whether S1 entities needing character-n-gram rescue (measurement-
only label: ≥1 missed link with word-cosine-similarity == 0, the "wildly
dissimilar" residual bucket, 191,142 entities / 10.83% of dev S1) can be
identified from observable signals alone (best forward score, best
reverse score, current B004 candidate count) — never from ground truth.

**Result: no useful rule found.** The best efficiency-ranked rule
(best_word_score ≤ 0) flagged only 0.81% of entities and captured just
6.18% of the target population. Even a very permissive rule (flag bottom
20% by score) captured only 22.74%. Candidate-count-based rules performed
worse. Root cause: char-rescue need is a **per-link**, not per-entity,
property — an entity with several true targets can have strong scores
from easy siblings while still missing one lexically dissimilar sibling,
so entity-level aggregate signals cannot distinguish it from a fully-
satisfied entity. **No observable targeting rule is adopted.**

### 25.4 Character Retrieval Resource Calibration (Part E)

EXP-B002's original pilot used `max_df=0.05` for the char-trigram
vectorizer and a bounded 660,733-row target pool, extrapolating to a
naive 200.8-minute full-scale estimate. Following the exact methodology
that fixed the word route's own analogous mistake (§4, item 7), the char
vectorizer was retuned and calibrated at **true full scale**:

- Fit on the real full corpus (12,085,675 documents) at `max_df=0.0008`:
  vocabulary 73,448 (12.7x smaller than the word route's 603,738), fit
  time 329.5s, peak RSS 18.2 GB.
- Transformed the **real full target matrix** (10,320,219 rows, not a
  bounded pool): nnz 38,520,424, transform time 278.1s, peak RSS 7.6 GB.
- A 3,000-query retrieval pilot against this real full target matrix
  extrapolated to **13.8 minutes** for full-scale retrieval — verdict
  **GO** on the very first (most aggressive, safest) `max_df` tried; no
  fallback to a more permissive value was needed.

### 25.5 Full-Scale Character Retrieval Execution

Ran forward char-trigram retrieval (k=50) for **all** 1,765,456 dev S1
entities against the real full 10,320,219-row target universe (no
targeting — Part D found none, so this applies broadly, exactly like
every other route, with no oracle gating anywhere in the decision to run
it):

- Corpus build: 70.6s. Fit: 329.5s. Transform: 278.1s. **Retrieval: 582.8s
  (9.7 min)** — faster than the 13.8-minute calibration estimate.
- Peak RSS: 19.15 GB (retrieval step). 65,924,065 candidate rows.
- Total wall time: ~21 minutes end-to-end.
- Artifact: `experiments/blocking/PHASE_2_5/char_full_candidates.parquet`.

### 25.6 Char Rescue Evaluation — Before / After

Unioned exact + forward-word(k=50) + reverse-word(r=5) + numeric-exact-set
+ char-full into a **new** artifact
(`experiments/blocking/PHASE_2_5/union_candidates_with_char.parquet`,
via `union_candidates_partitioned`, 160 buckets, 320,276,055 raw input rows
→ 287,735,705 deduped rows, union time 866.3s, peak RSS 23.9 GB), without
ever touching B004's own artifact.

| Metric | B004 (before) | B004 + char (after) | Δ |
|---|---|---|---|
| Link recall | 0.8914 | 0.8977 | +0.0063 |
| Macro candidate recall (non-singleton) | 0.8911 | 0.8974 | +0.0063 |
| Complete true-link coverage (non-singleton) | 0.7465 | 0.7573 | +0.0108 |
| Singleton candidate exposure | 0.9944 | 0.9967 | +0.0023 |
| Total candidate edges | 228,440,608 | 287,735,705 | +59,295,097 (+26.0%) |
| Overall candidate load (mean/median/p95/p99/max) | 129.4/56/493/657/3,361 | 163.0/100/524/691/3,361 | higher across the board |
| **New true links rescued by char (beyond B004)** | — | **38,148** | (0.62% of total truth) |
| **Newly-complete S1 entities** | — | **18,020** | — |
| **ORACLE_B004_MACRO_F0.5** | **0.950159** | **0.953901** | **+0.003741** |

**The pilot's own 74.4%-rescue-rate signal did not transfer to true full
scale**, and this discrepancy is itself the phase's most important
finding: EXP-B002's pilot retrieved top-50 candidates from a **bounded**
660,733-row target pool, where a true target competes against far fewer
rivals for a top-50 slot than it does against the real 10,320,219-row
universe. At true scale, the same true targets rank far lower on average
and mostly fall outside k=50 — the underlying lexical similarity scores
did not change, only the competition they face for a limited number of
slots. This is recorded as a concrete, quantified caution against trusting
a bounded-target-pool pilot's rescue-rate as a full-scale estimate,
independent of resource-cost extrapolation (which, by contrast, the
`max_df` retuning calibration in §25.4 handled correctly).

### 25.7 Final Decision

**Gap closed: 0.003741 / (0.980473 − 0.950159) = 12.35%** of the distance
from B004's oracle ceiling to the competitive reference. The char route is
adopted — it is strictly positive, cheap relative to its own measured cost
(21 minutes, memory-safe throughout), and required no targeting complexity
— but it does **not** close the competitive gap.

**Selected: Option 3 (B004 + broader/full character route), adopted as the
new candidate architecture, WITH THE EXPLICIT FINDING that this does not
achieve adequate competitive headroom.** Per the mission's own option
definitions, this is simultaneously the closest fit to "cost acceptable,
materially raises the ceiling" (in absolute terms: real, positive,
measured, essentially free) and an honest instance of "BLOCKED because no
Phase-2.5 approach produces adequate ceiling at practical cost" (in
relative terms: the ~2.66-point remaining gap to the competitive reference
is not addressable by any candidate-generation-stage mechanism identified
in this phase — numeric rare-token rescue was rejected twice on
independent evidence, §17; per-source retrieval showed no advantage at
matched budget, §18; char targeting found no viable rule, §25.3; and even
unrestricted full-scale char closes only ~12% of the gap). **No embeddings
were introduced, no matcher was trained, `validation_v1` was never
touched, and test data was never read**, per the mission's explicit
constraints. Closing the remaining gap is assessed as a Phase-3-level
(matcher quality/calibration) question, or a fundamentally different
blocking technique not investigated here, not a further blocking-stage
tuning question.

**Final selected candidate artifact:**
`experiments/blocking/PHASE_2_5/union_candidates_with_char.parquet`
(287,735,705 rows). Schema: all of B004's provenance columns (§19) plus
`route_char`, `char_rank`, `char_score`. B004's own artifact remains on
disk, unmodified, as the prior checkpoint.

### 25.8 Files Created (Phase 2.5)

`scripts/run_b004_oracle_f05.py`, `scripts/run_char_targeting_analysis.py`,
`scripts/run_char_scale_calibration.py`, `scripts/run_char_full_scale.py`,
`scripts/run_phase25_union_and_oracle.py`; all artifacts under
`experiments/blocking/PHASE_2_5/`.

### 25.9 Phase-3 Handoff Update

**Use `experiments/blocking/PHASE_2_5/union_candidates_with_char.parquet`
(287,735,705 rows), not B004's own artifact, as the Phase-3 candidate
input** — it strictly dominates B004 on every measured quality metric at a
26% candidate-load cost. Blocking ceiling for Phase-3 planning purposes:
oracle macro F0.5 = 0.953901; this is the hard upper bound any Phase-3
matcher can reach on this candidate set, and it is the number to compare
against the competitive reference, not the raw link-recall/coverage
figures. `char_score`/`char_rank` are recommended additions to the
Phase-3 feature starting set alongside `forward_score`/`reverse_score`
(§23).
