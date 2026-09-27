# Research-to-Experiment Plan — Phase 1.5 → Phase 2+

Ordered by expected information gain × competitive value ÷ (time + compute + engineering
cost), not by "simple → fancy." All experiments use the `development` partition of
`validation_v1` for iteration; `validation_v1`'s `validation` partition is reserved for
stage-level comparison only, per `docs/VALIDATION_DESIGN.md`'s model-selection policy —
**no experiment below tunes directly against `validation`.**

Every experiment ID is a placeholder for `experiments/EXPERIMENT_LOG.md` — none has been
run. "B" = blocking-stage experiment, "M" = matching-stage, "F" = feature/normalization.

**Revision note (post-independent-audit, round 1):** an independent read-only technical audit
(recorded in `RESEARCH_DOSSIER.md` §20) found two CRITICAL and twelve
MATERIAL issues in the first draft of this plan. That round's repairs added the
cross-cutting policies below (frozen-validation reuse, candidate-set provenance,
scale-budget requirement) and are recorded in `RESEARCH_DOSSIER.md` §20.

**Revision note (round 2 — methodology repair, no new research):** a second, human-directed
pass found the round-1 candidate-recall metric definition still conflated true-target
recall with singleton false-candidate exposure, the candidate-set provenance contract did
not draw a clear boundary at the decisive matching stage, EXP-M001/M002 lacked an explicit
out-of-sample development evaluation partition (distinct from whatever subset the matcher
trains on), and there was no lightweight scale/collision preflight before materializing full
blocking routes. This round repairs exactly those four things — see each affected section
below, each marked "(round 2)".

**Revision note (round 4 — teammate-research reconciliation, 2026-09-25):** an independent
teammate research package (see `TEAMMATE_RESEARCH_RECONCILIATION.md`) was reviewed and
verified. Four changes were made: a new EXP-B001R (reverse doc→S1 retrieval, promoted P0), a
new EXP-B001I (train-derived Indic-script normalization ablation, REPRODUCE FIRST/conditional
P1), an implementation note on EXP-B001 (word-token joint TF-IDF via sparse top-k cosine,
directly costed against character n-grams), and new numeric-conflict/entity-frequency/
co-duplicate-support features added to EXP-F001's inventory. No existing experiment's question,
metric, or acceptance criterion was weakened; no canonical Phase-1 fact was overridden; no
teammate exploratory score was adopted as a canonical benchmark. See
`TEAMMATE_RESEARCH_RECONCILIATION.md` §11/§12 for the full mapping.

**Revision note (round 3 — EXP-B001 semantic repair, no new research):** a third,
human-directed pass found EXP-B001's own hypothesis internally inconsistent with its own
implementation: EXP-B001 indexes and retrieves via *both* name tokens and address tokens,
but its round-1/round-2 wording treated the §11 "~10% zero-name-token-overlap" structural
ceiling as if it applied to EXP-B001 as a whole. That ceiling only applies to a blocker that
*requires* name-token overlap; EXP-B001 can independently rescue some zero-name-overlap
links via address tokens, and how much it rescues is an empirical question, not something to
assume either direction of. EXP-B001 and its immediate downstream reference in EXP-B002 are
corrected accordingly below, each marked "(round 3)"/"(corrected round 3)". No other
experiment, metric definition, or conclusion changed in this round.

---

## Cross-cutting policies (apply to every experiment below)

### Formal blocking-metric definitions (closes MATERIAL #6; revised round 2)

**Round-2 correction:** the round-1 definition of "macro per-entity candidate recall" made
a true singleton contribute 1.0 to that average only when its candidate set was also
empty — this silently mixed two different things: *recall of true targets* (undefined and
meaningless for an entity with zero true targets) and *false-candidate exposure risk* (a
real, separate risk for singletons). The definitions below separate them cleanly, per
explicit instruction. This is a blocking-diagnostics repair only — **the official Amazon
macro F0.5 evaluator (`student_resource/code/business_entity_resolution/src/evaluation.py`,
frozen per DEC-010) is not touched by this repair and its T-empty/P-empty scoring convention
is unchanged.** These are blocking-stage diagnostics computed *before* any matcher/threshold
decision exists, not a redefinition of the scored metric.

- **Link recall** = (number of individual true S1↔S2/S3 links whose target ID appears in
  that S1's candidate set) / (total true S1↔S2/S3 links), counted separately over the S2
  and S3 ground-truth link sets combined (not S1-count-weighted — every individual true
  link counts once). True singletons contribute no true links to either the numerator or
  denominator here (they have none), so this metric is unaffected by singleton behavior
  either way — unchanged from round 1.
- **Macro per-entity candidate recall (round 2: restricted to non-singleton entities)** =
  macro-average, taken **only over S1 entities that have at least one true target**, of
  `|recovered true targets for this S1| / |all true targets for this S1|`. True singletons
  are excluded from the population this average is computed over entirely — not scored as
  1.0, not scored as 0.0, simply not included, because "recall of true targets" has no
  meaningful value for an entity with zero true targets. Duplicate IDs within a candidate
  list are deduplicated via `set()` before scoring, matching DEC-010's evaluator contract
  for consistency, though this diagnostic is independent of the evaluator itself.
- **Complete true-link coverage (round 2: renamed from "complete-coverage rate," restricted
  to non-singleton entities)** = fraction of **non-singleton** S1 entities for which
  candidate recall (above) is exactly 1.0 — i.e. every true S2 and S3 target for that entity
  is present in the candidate set. This is the metric most directly threatened by the ~48%
  of entities needing 4+ correct targets (`docs/DATASET_AUDIT.md` §6) and must be reported
  separately from link recall in every blocking experiment, not inferred from it.
- **Singleton candidate exposure rate (round 2: new metric)** = fraction of true singletons
  (`docs/DATASET_AUDIT.md` §5 — 123,247 training entities, 5.58%) that receive **one or more
  candidates at all**. This is the blocking-stage analog of the false-merge risk the metric's
  2× precision weighting punishes downstream: every true singleton exposed to even one
  candidate is a live risk of a false match once a matcher/threshold stage runs, independent
  of what that later stage eventually decides. Must be reported separately in every blocking
  experiment — it is the metric that actually captures the singleton-specific risk the
  round-1 conflated definition was trying (incorrectly) to express through candidate recall.
- **Singleton candidate load (round 2: split out from the general candidate-load metric)** =
  candidate count **per true singleton only**, reported as mean/median/p95/p99/max. A route
  that behaves well on non-singletons could still expose singletons disproportionately (e.g.
  via a very common generic name or numeric token), so this must be tracked as its own
  distribution, not folded into the overall candidate-load number below.
- **Overall candidate load** = candidate count per S1 entity, across **all** entities
  (singleton and non-singleton together), reported as mean/median/p95/p99/max.
- **Reduction ratio** = candidate-pair count relative to the exhaustive S1×target Cartesian
  product, as before.
- **Scale budget** (closes MATERIAL #10): every blocking experiment reports, not just
  estimates, peak RAM (measured via OS-level monitoring during the run, not inferred from
  Phase-1 audit precedent), wall-clock runtime, disk usage for the candidate output file,
  and candidate-row count — before that route or the union is declared "feasible." See also
  EXP-B000 below, which estimates these *before* any route is run at full scale.

Every experiment below that previously said "all six blocking metrics" or "the six blocking
metrics" now means: link recall, macro per-entity candidate recall (non-singleton), complete
true-link coverage (non-singleton), singleton candidate exposure rate, singleton candidate
load, overall candidate load, and reduction ratio — seven named metrics plus the scale-budget
diagnostics, referred to below as "the full blocking-metric set."

### Frozen-validation reuse policy (closes CRITICAL-adjacent MATERIAL #2)

The first draft of this plan scheduled a `validation` touch at the end of nearly every
experiment (B004, M001, M002, M003, M004), which — even individually labeled "a single
confirmatory run" — amounts to adaptively selecting the next architecture step from each
result, i.e. exactly the repeated-tuning risk `docs/VALIDATION_DESIGN.md` warns against.
**Revised policy:** every blocking experiment (B001-B005) and the first two matcher
iterations (M001 random-negative baseline, M002 hard-negative iteration) run on
`development` only — no `validation` touch. The **first and only `validation` touch in this
phase of experiments** happens once, at EXP-M003 (calibrated, hard-negative-trained,
threshold-tuned matcher) — this is the first point the pipeline is "mature" in the sense the
original validation design intended a stage-level comparison for. EXP-M004 (country-filter
ablation) and EXP-M008 (conflict-resolution ablation, added below) are diagnostic/ablation
studies and stay on `development` only; if either changes the pipeline in a way that would
be shipped, that changed pipeline gets its own single, later `validation` confirmation
rather than an additional mid-stream one. Any future experiment beyond this document that
wants a `validation` read must state explicitly why the pipeline under test is a genuinely
new, complete candidate for selection — not a running tally of confirmations.

### Supervised matcher development split (closes CRITICAL #1; expanded round 2)

The round-1 fix here only split `development` into two parts and only required it starting
at the calibration step (EXP-M003). A second, human-directed review found this still left
EXP-M001/M002 with **no explicit out-of-sample evaluation partition** — "train and evaluate
on `development` only" does not by itself prevent in-sample evaluation, and the round-1 text
for EXP-M001 still said "validation macro F0.5" / "frozen-validation score," stale wording
left over from before `validation` touches were removed from that experiment. This round
replaces the two-way split with an explicit three-way one, required from EXP-M001 onward,
not just from EXP-M003.

**Required structure — a deterministic, entity-level subdivision inside `validation_v1`'s
existing `development` partition (never touching `development`'s sibling `validation`
partition):**

- **`matcher_train`** — the only subset any supervised matcher (GBDT in EXP-M001/M002, or
  any later model) is fit on.
- **`dev_eval`** — held out from all matcher training; EXP-M001 and EXP-M002 are evaluated
  out-of-sample on this same frozen population, so their results are comparable to each
  other and are not in-sample artifacts of `matcher_train`.
- **`calibration_holdout`** — held out from `matcher_train` (and distinct from `dev_eval`);
  isotonic regression / threshold search (EXP-M003) is fit only against the matcher's
  out-of-sample predictions on this subset, never against `matcher_train`'s own predictions.

An equivalent leakage-safe k-fold cross-fitting scheme (train on k−1 folds, evaluate/
calibrate on the held-out fold, rotate) is an acceptable alternative to fixed subsets if a
simple three-way split leaves too little data in a thin stratum — either way, **no matcher
evaluation number, calibration, or threshold value may be computed from predictions the
underlying matcher was trained on.**

**Requirements on the split itself, once it is built:**
- split by S1 entity (never by pair — consistent with why `validation_v1` itself is an
  entity-level split, `docs/VALIDATION_DESIGN.md`);
- preserve country/singleton/match-multiplicity structure sensibly across the three subsets
  (mirroring `validation_v1`'s own stratification factors is a reasonable default, not a
  requirement re-derived from scratch here);
- no S1 entity appears in more than one of `matcher_train`/`dev_eval`/`calibration_holdout`;
- `validation_v1`'s `validation` partition remains completely untouched by any of this —
  it is still reserved exclusively for EXP-M003's single confirmatory read, per the
  frozen-validation reuse policy above.

**This split is not created during this research phase.** Building the actual manifest
(entity IDs assigned to each of the three subsets, with a fixed seed and a recorded hash,
the same discipline `validation_v1` itself used) is a **precondition for EXP-M001**, to be
created as part of whichever future phase actually implements the matcher experiments
(Phase 3, per the phase sequence in this project) — not invented speculatively here where it
cannot be tested against real data.

**Stale wording corrected (round 2):** EXP-M001 previously asked "what *validation* macro
F0.5 does the simplest reasonable matcher... achieve?" and described its acceptance
criterion as "a reproducible *frozen-validation* score" — both are corrected below to refer
to the out-of-sample `dev_eval` result, since EXP-M001 does not touch `validation_v1`'s
`validation` partition at all (that touch is reserved for EXP-M003 only).

### `candidate_pairs.tsv` provenance contract (closes CRITICAL #2; tightened round 2)

`docs/OFFICIAL_REQUIREMENTS.md` requires `candidate_pairs.tsv` to be "the exact, final
candidate set fed into the matching model at inference time," not an earlier raw blocking
pass later filtered further, and every ID in `matching_results.tsv` must appear in the
corresponding `candidate_pairs.tsv` row.

**Round-2 correction:** the round-1 wording ("the exact point candidates stop changing...
after any matcher-side re-filtering") was imprecise about *when* that boundary falls — it
could be misread as placing `candidate_pairs.tsv` *after* the decisive matcher has already
scored and filtered candidates, which is wrong. The correct boundary, precisely stated:

**`candidate_pairs.tsv` = the candidate set as it exists immediately after all upstream
blocking, retrieval, and candidate-pruning stages have finished, and immediately *before*
the decisive matching/scoring stage runs.** It is the *input* to the decisive matcher, not
a byproduct of the matcher's output. Everything the decisive matcher does after that point —
scoring, calibration, threshold application, acceptance/rejection logic — happens strictly
downstream of the set this file records and must never cause candidates to be added to or
silently removed from what gets written as `candidate_pairs.tsv`; it may only ever be a
*subset* relationship going the other way (`matching_results.tsv` IDs ⊆ that row's
`candidate_pairs.tsv` IDs).

**If the eventual pipeline uses a learned/cheap pruning stage (e.g. EXP-B005) in addition to
a separate decisive matcher (e.g. the GBDT from EXP-M001 onward), the implementation must
explicitly name, in its own documentation:**
1. the **upstream candidate-pruning stage(s)** (blocking routes, union, any recall-preserving
   pruning such as EXP-B005) — candidates may still be added or removed here;
2. the **decisive matching stage** (the GBDT/calibration/threshold pipeline that produces
   `matching_results.tsv`) — candidates must **not** be added or removed here, only scored
   and accepted/rejected;
3. the **exact boundary** between (1) and (2) — the single versioned artifact that both
   `candidate_pairs.tsv` is built from *and* that stage (2) reads as its entire input, with
   no other candidate source feeding stage (2).

This is a pipeline-engineering requirement to carry into Phase 2/3's implementation, not
something this research phase resolves by itself (no pipeline exists yet to apply it to); it
is recorded here so the boundary is not lost or blurred once implementation starts. A
submission-time validator check (every `matching_results.tsv` ID present in the same-row
`candidate_pairs.tsv` entry) should be added to whatever wraps `utils/validate_submission.py`.

### Country handling for the unseen France category (closes MATERIAL #8)

Per DEC-017, whether to hard-filter on country is still an open, to-be-measured question
(EXP-M004 below). Independent of that: any feature table (EXP-F001) should default to a
**country-agreement feature** (do S1 and candidate country strings match?) rather than a raw
one-hot/categorical encoding of `country` itself as a GBDT input — a raw categorical
encoding trained only on `{US, India}` has no learned behavior for the `France` category at
all, which is a distinct risk from the hard-filter question EXP-M004 addresses. If a raw
country identity feature is tried at all, it must be an explicit ablation compared against
the country-agreement-only feature set, never the unqualified default.

---

## EXP-B000 — Blocking scale / collision preflight (round 2 addition)

- **Question:** before materializing any of EXP-B001/B002/B003 at full `development` scale,
  what do cheap, small-pilot estimates say about the resource and collision risk of each
  route — specifically, is there a realistic risk of repeating Phase 1's own documented
  memory-pressure incident (`experiments/EXPERIMENT_LOG.md` EXP-000: one computation dropped
  free RAM from ~17GB to ~3GB and had to be rewritten as a streaming pass)?
- **Why this exists — scope boundary (must be read before running this experiment):** this
  is a short engineering/scale gate, not a new research question. It must **not** become
  research into new blocking algorithms, new literature, or new techniques — it only
  estimates resource and collision risk for the three routes already selected in
  `TECHNIQUE_EVIDENCE_MATRIX.md` §1.1-1.3, using a small pilot sample, so that EXP-B001-B004
  are run with a known-safe budget rather than discovering a memory or runtime problem
  mid-run at full scale.
- **Implementation scope (smallest version — a pilot, not a full run):** on a small,
  deterministically-sampled pilot subset of `development` (e.g., a fixed few thousand S1
  entities and their candidate-generating S2/S3 rows — exact size is an implementation
  detail for whoever runs this, not fixed here), estimate:
  - token document-frequency / posting-list-size distribution (how many records share the
    single most common name/address token — the direct predictor of EXP-B001's worst-case
    candidate blocks);
  - very-high-frequency-token risk specifically (is there a token analogous to §14's
    ~460-way name-collision groups, or worse, in the address/token vocabulary?);
  - numeric-signature collision distribution estimate (a cheap pilot-scale precursor to
    EXP-B003's full collision-group analysis — this does not replace that experiment, only
    gives an early warning before full-scale materialization);
  - character-n-gram vocabulary size and sparse-matrix footprint estimate (bytes) for
    EXP-B002's TF-IDF construction, extrapolated from the pilot to full `development` scale;
  - estimated full-scale candidate-row count (extrapolated from the pilot's observed
    candidates-per-entity rate);
  - estimated full-scale candidate-output disk usage (bytes), extrapolated the same way;
  - estimated full-scale peak RAM / index size, extrapolated from the pilot's measured
    memory use;
  - representative small-pilot wall-clock runtime, extrapolated to a full-scale runtime
    order-of-magnitude estimate.
- **Data scope:** a small pilot subset of `development` only — explicitly not the full
  `development` partition (that materialization is EXP-B001-B004's job), and never
  `validation`.
- **Metrics:** the estimates listed above, each labeled ESTIMATED (extrapolated from a
  pilot) vs. MEASURED (directly observed in the pilot itself) per this project's own
  evidence-labeling discipline (`docs/DATASET_AUDIT.md`'s VERIFIED/INFERRED/UNKNOWN
  convention, adapted here to MEASURED/ESTIMATED for resource figures).
- **Acceptance criterion:** extrapolated peak RAM, disk, and runtime for all three routes
  fall within a budget that leaves reasonable headroom under the measured 31.6GB machine
  total (`docs/DATASET_AUDIT.md` §15 hardware reference) — informing, not blocking, EXP-B001
  onward; this is a planning input, not a pass/fail gate on the routes themselves.
- **Fail/stop condition:** if any route's extrapolated resource use appears to approach or
  exceed the machine's measured RAM, that route's EXP-B001/B002/B003 implementation should
  plan a streaming/chunked approach from the start (the same discipline Phase 1's EXP-000
  had to retrofit after the fact) rather than attempting a full in-memory materialization
  first and discovering the problem mid-run.
- **Compute cost:** ESTIMATED LOW (pilot-scale only, by design).
- **Time cost:** LOW.
- **Dependencies:** none.
- **Order: 0th — runs before EXP-B001.**

## EXP-B001 — Token-blocking recall baseline

**Round-3 correction (semantic repair, EXP-B001 only):** the round-1/round-2 text below
treated business-name tokens and business-address tokens as forming one merged retrieval
key, then separately assumed the ~10% zero-name-token-overlap population (§11) would be
"near-total missed" by this experiment — but EXP-B001 also indexes and retrieves via
address tokens. A true pair can have zero name-token overlap while still sharing address
tokens, so that pair may still be recovered through the address-token route even though it
is invisible to the name-token route specifically. **The ~90%/~10% split from §11 is a
structural ceiling only for a blocker that *requires* name-token overlap — it does not
automatically apply to EXP-B001, which can retrieve independently via address tokens.**
Whether, and how much of, the zero-name-overlap tail address tokens actually rescue is an
empirical question this experiment must measure, not assume either way. This correction is
scoped to EXP-B001's question/hypothesis/metrics/acceptance/stop logic only — it does not
change EXP-B001's role as a simple token-based baseline, and it does not pull character
n-gram retrieval (EXP-B002) or numeric-signature retrieval (EXP-B003) into this experiment.

- **Question:** how much true-link recall can simple token blocking achieve when candidate
  generation uses business-name tokens and business-address tokens as **complementary**
  token-derived retrieval signals (not a single merged condition)? In particular: how well
  does it recover links with nonzero name-token overlap (§11)? How much of the ~10%
  zero-name-token-overlap tail can address-token overlap recover? What candidate-load cost
  does this simple token baseline incur?
- **Hypothesis:** name-token retrieval should recover a substantial fraction of links with
  nonzero name-token overlap. Address-token retrieval may additionally recover some true
  links that have zero name-token overlap. **The amount of this complementary recovery must
  be measured — do not assume either that the entire zero-name-overlap tail will be missed,
  or that address tokens will recover most of it.** A candidate-load long tail driven by
  §14's collision groups (up to ~460-way) is still expected on the name-token side.
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §1.1.
- **Project basis:** DATASET_AUDIT.md §11 (Jaccard quantiles), §14 (collision structure).
- **Implementation scope:** smallest version — build an inverted index over normalized
  name tokens and, separately, over normalized address tokens, on the `development`
  S1/S2/S3 subset; generate candidates per S1 as the **union** of both routes' token
  co-occurrence results (each S1's candidate set can therefore contain IDs found only via
  name tokens, only via address tokens, or both), no filtering/pruning yet. This is still a
  single simple baseline experiment, not a merge with EXP-B002/B003.
- **Data scope:** `development` only.
- **Metrics:** the full blocking-metric set as formally defined above (link recall, macro
  per-entity candidate recall restricted to non-singleton entities, complete true-link
  coverage restricted to non-singleton entities, singleton candidate exposure rate,
  singleton candidate load, overall candidate load, reduction ratio) plus scale-budget
  diagnostics (peak RAM, runtime, disk, candidate-row count), **plus the following
  diagnostic breakdowns specific to this experiment:**
  - **(A) Overall link recall** — across all true S1↔S2/S3 links.
  - **(B) Nonzero-name-token-overlap link recall** — recall restricted to true links where
    S1 and the true target share ≥1 normalized business-name token.
  - **(C) Zero-name-token-overlap link recall** — recall restricted to true links where the
    business-name token intersection is empty (the §11 ~10% tail).
  - **(D) Address-rescued zero-name-overlap links** — among the true links counted in (C),
    how many are nevertheless present in the candidate set because the address-token route
    found them (i.e., recovered via address-token overlap despite zero name-token overlap)?
    Report both count and percentage of the (C) population.
- **Acceptance criterion:** link recall on the nonzero-name-token-overlap population (B) is
  close to the theoretical ~90%-of-links ceiling implied by §11 for that population
  specifically (i.e., tokenization/normalization isn't silently losing recall there); **the
  zero-name-token-overlap population (C) has no assumed target — its measured recall, and
  specifically the address-rescue fraction (D), is itself the finding**, not a pass/fail
  threshold to be judged against.
- **Fail/stop condition:** if recall on the nonzero-name-token-overlap population (B) falls
  well short of its ~90% ceiling, the bug is in normalization/tokenization, not the
  blocking concept — fix normalization before concluding anything about the technique. A
  low address-rescue fraction (D) is not a failure of this experiment; it is a valid
  finding that sizes how much work is left for EXP-B002/B003 on the remaining
  zero-overlap, non-address-rescued population.
- **Expected value:** establishes the baseline every other blocking route is measured
  against; cheapest possible experiment to run first; also produces the first real measured
  answer to how much complementary value simple address-token retrieval already provides,
  which directly sizes the urgency of EXP-B002 (character n-gram) specifically for the
  *residual* zero-overlap population, not the full ~10%.
- **Compute cost:** ESTIMATED LOW (two inverted-index passes — name, address — CPU-only).
- **Time cost:** LOW (implementation + one run).
- **Dependencies:** EXP-B000 (preflight; informs the implementation's memory/runtime
  approach but is not a hard blocking gate on starting this experiment).
- **Order: 1st (after EXP-B000).**
- **Information gain even if it "loses":** quantifies exactly how large the *true* residual
  zero-overlap gap is (i.e., zero name-token overlap **and** not address-rescued) on our
  real data, which directly sizes the urgency of EXP-B002/B003 more precisely than the raw
  §11 figure alone.
- **Implementation note (added after teammate-research reconciliation — see
  `TEAMMATE_RESEARCH_RECONCILIATION.md` §11):** **TEAMMATE EXPLORATORY EVIDENCE** (independent
  single-machine measurement, not a canonical `development` result) found a word-token joint
  name+address TF-IDF (sparse top-k cosine retrieval via a hashed vectorizer, IDF-weighted,
  df-pruned) at 2–8 ms/query, versus a character 4-gram equivalent at ~60 ms/query
  for comparable recall (a ~27× cost ratio) and a character 3-gram variant that failed to
  complete at scale unpruned. This does not change EXP-B001's question, hypothesis, or metrics
  — it is guidance on the concrete scoring mechanism to implement for the token-blocking route
  (sparse top-k cosine over hashed/TF-IDF word-unigram vectors, not raw inverted-index
  intersection), since it is directly comparable in engineering cost to the inverted-index
  approach and has now been measured, not merely assumed.

## EXP-B001R — Reverse (doc→S1) retrieval (new, added after teammate-research reconciliation)

- **Question:** for each S2/S3 document, what is the recall of retrieving its true S1 within
  the document's own top-r nearest S1 entities (the same word-token joint TF-IDF vectors as
  EXP-B001, queried in the opposite direction — S1 is the smaller, already-deduplicated index)?
  Does this reverse pass recover true links that EXP-B001's forward (S1→doc) pass misses, and
  does the union of forward-per-source-top-k and reverse-top-r materially raise recall over
  either alone?
- **Why this exists:** **TEAMMATE EXPLORATORY EVIDENCE** (`TEAMMATE_RESEARCH_RECONCILIATION.md`
  §4 Finding #1, directly verified from raw script output, but on a non-canonical train sample,
  not `development`) measured reverse@1 recall of 92.4–96.1% and reverse@2 of 94.0–97.1% —
  higher than forward-per-source@10 alone (86.9–93.8%) — and a forward-top-k ∪ reverse-top-r
  union reaching 94.8–98.4% pair recall on that sample. This is not assumed to transfer at that
  exact magnitude to `development` at full
  scale (§6/§8 of the reconciliation document name the scale/split differences explicitly) —
  it is strong enough directional evidence to justify a real experiment, run at this project's
  own canonical scale and split discipline.
- **Why this is expected to work, structurally:** `docs/DATASET_AUDIT.md` §8 (VERIFIED) already
  establishes that every true-matched S2/S3 id belongs to exactly one S1 entity
  (`max_s1_degree_for_any_target == 1`) — i.e., the S1→target relationship is a partition from
  the target side. A document therefore has at most one "correct" S1 to be retrieved by, which
  is exactly the structural precondition reverse nearest-neighbor retrieval exploits; this is
  the same target-reuse evidence EXP-M008's capacity-one conflict check already relies on,
  applied here at the blocking stage instead of the matching stage.
- **Research basis:** `TEAMMATE_RESEARCH_RECONCILIATION.md` §4/§9; structurally motivated by
  `docs/DATASET_AUDIT.md` §8 (already canonical).
- **Implementation scope:** smallest version — reuse EXP-B001's word-token joint TF-IDF
  vectors; for each S2/S3 document in `development`, retrieve its top-r (start r=2–3) nearest
  S1 entities by the same joint cosine score; union with EXP-B001's forward-per-source-top-k
  candidates per S1.
- **Data scope:** `development` only.
- **Metrics:** the full blocking-metric set (as EXP-B001), computed for the reverse route
  alone and for the forward∪reverse union; additionally report the reverse route's own
  candidate load per document (mean/median/p95/p99/max) and runtime per document, since cost
  characteristics differ from a per-S1 forward query (teammate evidence: reverse cost is
  sensitive to memory pressure — budget accordingly, see the scale-budget cross-cutting policy
  above).
- **Acceptance criterion:** the forward∪reverse union's complete true-link coverage and
  candidate load materially improve over EXP-B001 (forward) alone, without a candidate-load
  blowup.
- **Fail/stop condition:** if reverse retrieval's own candidate load or runtime is
  disproportionate (e.g., driven by a small number of S1 entities acting as a "reverse
  attractor" for many unrelated documents), investigate before abandoning — this is exactly
  the kind of collision behavior EXP-B000's preflight is meant to catch early for the routes it
  already covers, and should be added to that preflight's scope for future reruns.
- **Dependencies:** EXP-B001 (shares its vectorization); can run in parallel with EXP-B001's
  own measurement once vectors exist.
- **Order: alongside EXP-B001, both feeding EXP-B004.**

## EXP-B001I — Train-derived Indic-script normalization ablation (new, added after teammate-research reconciliation; REPRODUCE FIRST)

- **Question:** does a token-level Devanagari/Indic-script → Latin-script transliteration
  dictionary, learned **only** from `train_ground_truth.tsv` pairs restricted to
  `matcher_train` (never `dev_eval`, `calibration_holdout`, or `validation`), measurably
  improve India-side retrieval recall on an entity-disjoint held-out set, without evidence of
  leakage?
- **Why this exists:** **TEAMMATE EXPLORATORY EVIDENCE** (`TEAMMATE_RESEARCH_RECONCILIATION.md`
  §4 Finding #3, on a non-canonical train sample) measured a +6.04pt forward@10 recall
  improvement on India from a 1,312-token name dictionary and 17-token address dictionary,
  built by positional/co-occurrence alignment
  on train pairs, using a leakage-conscious methodology (S1-entity-level 80/20 held-out split,
  dictionary quality measured only on the held-out 20%, test-side coverage measured with no
  labels). This is credible, directly-verified evidence — but the **specific artifact shipped**
  in the teammate's package was built on an 80%-of-entities split that does not correspond to
  this project's own `matcher_train` manifest, so it must be rebuilt, not reused as-is
  (`TEAMMATE_RESEARCH_RECONCILIATION.md` §6/§9).
- **Research basis:** `TEAMMATE_RESEARCH_RECONCILIATION.md` §4/§6/§9; no external
  transliteration dictionary or API — learned entirely from supplied training data, consistent
  with the fair-play rule.
- **Compliance note:** CLEAR on fair-play grounds (no external data/lookup). This is **not**
  the same category as a hand-authored, standards-sourced country-specific abbreviation list
  (which the reconciliation document's §7/§10 explicitly declines to adopt for France, citing
  the existing `DEC-014`/§23 anti-pattern policy) — a script-transliteration table learned
  purely from supplied training pairs is a data-derived normalization step, not external domain
  knowledge, and is judged separately on that basis.
- **Implementation scope:** rebuild the dictionary using only `matcher_train` entities (once
  that manifest exists, per the supervised-matcher development split policy above); evaluate
  coverage/purity on `dev_eval` (never `matcher_train` itself, never `validation`); measure the
  resulting forward/reverse retrieval recall delta on India-country `dev_eval` entities with
  vs. without the dictionary applied during normalization.
- **Data scope:** `matcher_train` for dictionary construction, `dev_eval` for evaluation — both
  subsets of `development`; never `validation`.
- **Metrics:** dictionary coverage/purity on `dev_eval` (mirroring `deva_exp.py`'s own
  held-out methodology); India-country forward@k/reverse@r recall delta with vs. without the
  dictionary, as part of the full blocking-metric set.
- **Acceptance criterion:** a measurable, reproducible recall improvement on India `dev_eval`
  entities that were not used to build the dictionary, at a magnitude in the same direction as
  (not necessarily identical to) the teammate's sample-scale finding.
- **Fail/stop condition:** if the entity-disjoint held-out improvement is materially smaller
  than the teammate's sample-scale result, treat the difference as informative about sample-
  scale effects, not as a implementation bug, unless the held-out split itself is found to leak.
- **Dependencies:** the `matcher_train`/`dev_eval`/`calibration_holdout` manifest (same
  precondition as EXP-M001 onward — this experiment cannot run before that manifest exists);
  EXP-B001 (shares normalization pipeline).
- **Order: conditional, runs once the Phase-3 development-split manifest exists — not blocking
  EXP-B001–B005's own Phase-2 sequence, which can proceed without it using the existing
  Devanagari-only or no-dictionary normalization as an interim baseline.**

## EXP-B002 — Character n-gram / TF-IDF blocking recall

- **Question (corrected round 3):** does character-n-gram TF-IDF top-k retrieval recover
  the population EXP-B001 misses **entirely** — i.e., true links with zero name-token
  overlap that EXP-B001's address-token route also failed to rescue (EXP-B001 diagnostic
  (D)) — per Sparkly's precedent? (Not the full raw ~10% zero-name-token-overlap population
  from §11: EXP-B001 may have already recovered part of that population through address
  tokens, per its own round-3 correction above — EXP-B002's actual target is whatever
  residual remains after EXP-B001's measured address-rescue fraction.)
- **Hypothesis:** high recall on that residual zero-overlap, non-address-rescued tail;
  comparable or better overall recall than EXP-B001 at a plausibly larger but bounded
  candidate load (top-k controls this, per Sparkly's design).
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §1.2 (Sparkly, Cohen et al.).
- **Project basis:** DATASET_AUDIT.md §11.
- **Implementation scope:** smallest version — single-node sparse TF-IDF over character
  3-grams. **Both name and address fields are in scope, run as two explicit ablations (name-
  only, address-only, and the union of both), not address-as-optional** (revised after audit
  finding: treating address n-grams as "if time allows" would leave the zero-name-overlap
  tail's actual recovery rate — the specific population this route is supposed to protect —
  unquantified). Top-k retrieval per S1 entity (k as a tunable start point, e.g. k=50, adjust
  based on candidate-load results).
- **Data scope:** `development` only.
- **Metrics:** the full blocking-metric set as in EXP-B001, **plus a specific breakdown on
  the EXP-B001 residual-miss subset** (zero name-token overlap and not address-rescued per
  EXP-B001 diagnostic (D)) — does this route recover exactly that residual population, not
  the raw §11 ~10% figure which EXP-B001 has already partially addressed?
- **Acceptance criterion:** materially recovers true links that EXP-B001's combined name+
  address token routes missed entirely, without a candidate-load blowup that makes
  downstream matching infeasible.
- **Fail/stop condition:** if candidate load explodes (e.g., p99/max far exceeds what a
  GBDT-scale matcher can process per entity) without a proportional recall gain, reduce k
  or add a minimum-similarity cutoff before abandoning the technique.
- **Expected value:** HIGH — Sparkly is the strongest single piece of blocking evidence
  found in this research pass, at a directly comparable scale.
- **Compute cost:** ESTIMATED MEDIUM (sparse matrix construction over ~10M+ rows; single-
  node, not Sparkly's distributed setup — time this explicitly since it is not directly
  measured for our hardware).
- **Time cost:** MEDIUM.
- **Dependencies:** EXP-B001 (for the comparison baseline and the specific "recovered the
  missed population" check).
- **Order: 2nd.**

## EXP-B003 — Numeric-address-token blocking recall

- **Question:** how much recall does the numeric-token signal (§12, 63.49% of true links)
  contribute as a standalone blocking route, and what is its collision-group structure
  (not yet measured in Phase 1 — §14 only analyzed name/address-pair collisions, not
  numeric-token collisions specifically)?
- **Hypothesis:** high recall on the numeric-overlap population, largely orthogonal to
  EXP-B001/B002's misses (since surrounding address text varies even when numbers match,
  per §12), but with its own collision risk on common/generic numeric tokens.
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §1.3.
- **Project basis:** DATASET_AUDIT.md §12, §21.
- **Implementation scope:** extract numeric tokens from addresses (already done once for
  the Phase-1 audit — reuse/extend that extraction code), build an exact-numeric-set (and
  optionally partial-overlap) index, generate candidates.
- **Data scope:** `development` only.
- **Metrics:** the full blocking-metric set as in EXP-B001, **plus a new numeric-token
  collision-group-size analysis** analogous to DATASET_AUDIT.md §14's name-collision table
  (not previously computed — this experiment is also the place to close that specific
  evidence gap).
- **Acceptance criterion:** recall on the numeric-overlap true-link population is high;
  collision groups are not dominated by a small number of catastrophically common tokens
  that would need special handling (e.g., a very frequent unit number).
- **Fail/stop condition:** if numeric-token collision groups are as large as or larger than
  the worst name collisions (§14's ~460-way example), a raw exact-numeric-set blocking key
  needs a rarity/IDF weighting refinement before being trusted as a P0 route.
- **Expected value:** HIGH — this is our own strongest measured signal (§12), not an
  imported one; also fills a genuine gap in the Phase-1 audit (numeric-token collision
  structure was never measured).
- **Compute cost:** ESTIMATED LOW.
- **Time cost:** LOW-MEDIUM.
- **Dependencies:** none (can run in parallel with EXP-B001/B002).
- **Order: parallel with EXP-B002, both after EXP-B001's tokenization sanity-check.**

## EXP-B004 — Multi-route union benchmark (the Phase-2 headline experiment)

- **Question:** what link recall, complete true-link coverage, singleton candidate exposure,
  and candidate load does the union of EXP-B001+B002+B003 achieve, and is it a defensible
  `candidate_pairs.tsv` strategy at full scale?
- **Hypothesis:** materially higher complete true-link coverage (over non-singleton
  entities) than any single route alone (critical given ~48% of entities need 4+ correct
  targets simultaneously, §6), at a candidate load that is larger than any single route but
  not combinatorially explosive. **Unioning three routes could also increase singleton
  candidate exposure relative to any single route (more routes = more chances a true
  singleton picks up a false candidate) — this is a real, separate risk the union must be
  checked against, not just its recall benefit.**
  **This experiment is the actual test of route complementarity — the routes are only
  presumed complementary by design (each targets a different Phase-1-measured population,
  §1.1-1.3); Phase 1 did not measure their overlap or joint recall directly, so the "sub-
  additive, largely orthogonal" characterization must be treated as a hypothesis this
  experiment confirms or refutes, not an established fact** (revised after audit finding
  that the first draft stated this more strongly than the evidence supported).
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §1.4 (anti-anchoring rule, §6 of the
  research brief — union, not intersection).
- **Project basis:** DATASET_AUDIT.md §6 (multiplicity), §13 (no single signal covers the
  hard middle).
- **Implementation scope:** deduplicate unioned candidate IDs per S1 across the three
  routes; no learned pruning yet (that is EXP-B005, conditional).
- **Data scope:** `development` only — no `validation` touch at this stage (revised per the
  frozen-validation reuse policy above; the first draft's per-experiment "single
  confirmatory run" pattern was itself the adaptive-reuse risk the audit flagged).
- **Metrics:** the full blocking-metric set, broken out by country, target source, match
  multiplicity, missing-address status, and (if feasible) the §13 difficulty bucket.
- **Acceptance criterion:** complete true-link coverage (non-singleton) materially higher
  than any single route; singleton candidate exposure rate not disproportionately worse than
  the best individual route; candidate load p95/p99 within a range the planned matcher
  (GBDT, §3.2) can process within the competition's compute/time budget.
- **Fail/stop condition:** if candidate load is unworkable, prioritize per-route caps
  (top-k per route) over abandoning any route entirely, since each route targets a
  genuinely different recall population (§1.1-1.3).
- **Expected value:** this experiment's output *is* the Phase-2 deliverable — the
  candidate-generation stage the rest of the pipeline depends on.
- **Compute cost:** ESTIMATED MEDIUM-HIGH (dominated by whichever component route is most
  expensive, likely EXP-B002's TF-IDF construction).
- **Time cost:** MEDIUM.
- **Scope note (added after teammate-research reconciliation):** the union now includes
  EXP-B001R's reverse (doc→S1) route as a fourth input alongside token/char-n-gram/
  numeric-address routes — it is cheap (shares EXP-B001's vectors) and independently evidenced
  (`TEAMMATE_RESEARCH_RECONCILIATION.md` §4/§9). EXP-B001I's Indic dictionary, if and when it
  passes its own reproduction gate, is a normalization-pipeline input to all routes, not a
  separate union member.
- **Dependencies:** EXP-B001, EXP-B001R, B002, B003 all complete.
- **Order: 4th (after all three-plus-reverse component routes are individually measured).**
- **Rollback:** if the union proves unworkable at full scale, fall back to the two
  cheapest, highest-individual-recall routes (numeric-token + token blocking) and treat
  character-n-gram retrieval as a P1 follow-up restricted to the residual zero-recall
  population only, rather than a full-population route.

## EXP-B005 — Candidate-load pruning (conditional)

- **Question:** if EXP-B004's candidate load is too high for economical matching, does a
  cheap pruning step (e.g., per-route top-k caps, or a lightweight learned edge-pruning
  pass per TECHNIQUE_EVIDENCE_MATRIX.md §1.8) reduce load without meaningfully hurting
  recall?
- **Dependencies:** only runs if EXP-B004's fail/stop condition triggers.
- **Order: conditional, after EXP-B004.**

---

## EXP-F001 — Feature-set construction from Phase-1-validated signals

- **Question:** do the specific features Phase 1 already validated as informative (name
  Jaccard, address Jaccard, numeric-token exact/partial agreement, length ratios, source
  indicator, country agreement) produce a usable feature table at candidate-set scale?
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §2.1-2.4.
- **Project basis:** DATASET_AUDIT.md §11, §12 (the exact statistics these features
  reproduce, now computed per-candidate-pair rather than per-true-link).
- **Implementation scope:** smallest version — reuse the Phase-1 audit's normalization
  code (`normalization.py`, explicitly NOT production-frozen per PROJECT_STATE.md — treat
  this experiment as the point where a production normalization version gets decided and
  frozen), applied to EXP-B004's candidate output. **Explicit feature inventory (revised
  after audit finding that "RapidFuzz-based scoring" was too vague to guarantee the P0
  feature set from TECHNIQUE_EVIDENCE_MATRIX.md §2.1-2.4 was actually implemented):**
  character-level similarity (Levenshtein ratio, Jaro-Winkler — both via RapidFuzz) on
  name and on address; token-level similarity (Jaccard, Dice/overlap coefficient) on name
  and on address; TF-IDF cosine similarity on name and on address (character 3-gram, reusing
  EXP-B002's vectorizer where possible); numeric-address-token features — exact-set
  agreement, partial-overlap ratio, **and an explicit conflict flag (numeric tokens present
  on both sides but disjoint) as a separate feature from "no agreement,"** since a
  present-but-conflicting signal is qualitatively different contradictory evidence than
  simple absence (§7/§2.4 of the dossier); length-ratio features (name, address); source
  indicator (S2 vs. S3); **country-agreement feature (not raw country identity — see the
  unseen-France-category policy above)**; a rarity/uniqueness feature for the candidate's
  normalized name (proxying §14's collision-group-size finding). **Added after
  teammate-research reconciliation (`TEAMMATE_RESEARCH_RECONCILIATION.md` §4 Finding #6, §12 —
  P0; TEAMMATE EXPLORATORY EVIDENCE showed a directly-verified sample-scale gain, not yet a
  canonical `development` measurement):** numeric-conflict sub-features refining the conflict flag
  above — absolute/log-difference of the first extracted numeric token on each side
  (distinguishes a small typo-sized difference from a random-looking distractor mismatch), an
  all-numeric-tokens-equal flag, and a core-name equality/token-diff pair computed after
  stripping a fixed legal-suffix/stopword list (`pvt, ltd, llc, inc, corp, co, the, and, …` —
  paired with, never substituting for, the raw/full name features above, per §2.3's own
  over-normalization caution). **P1, Phase-3 backlog (Finding #7):** label-free,
  country-agnostic entity-frequency features — counts of how many other S1 entities share this
  candidate's normalized name/address, and how many other candidate documents share the same,
  computed per country partition on the inputs alone (no labels) — motivated by France's
  measured "crowded candidate list" structural profile. **P1, Phase-3 backlog (Finding #8):**
  a two-stage co-duplicate "graph-lite" support feature — for each candidate, a
  similarity-weighted aggregate (max/sum/count-above-threshold) of an out-of-fold first-stage
  matcher score across the *other* candidates of the same S1 entity, exploiting the fact that a
  true document usually has sibling duplicates in the same candidate list while a distractor is
  typically a lone near-copy; ordinary feature engineering over independently-scored pairs, not
  a graph algorithm, and not a reversal of the collective-ER REJECT
  (`TECHNIQUE_EVIDENCE_MATRIX.md` §10).
- **Data scope:** `development`.
- **Metrics:** feature coverage (missingness per feature), basic separation check (feature
  distributions on true-match vs. non-match candidates within `development`) — a
  diagnostic, not a model evaluation.
- **Acceptance criterion:** features show visible separation between true and false
  candidates on at least the easier buckets (§13 A/B/C); full separation is not expected
  here (that's the matcher's job).
- **Dependencies:** EXP-B004.
- **Order: 5th.**

## EXP-M001 — GBDT baseline matcher (random-negative training)

- **Question (corrected round 2 — stale "validation" wording removed):** what
  out-of-sample, `dev_eval` macro F0.5 does the simplest reasonable matcher (GBDT on
  EXP-F001's features, trained only on `matcher_train` with random negatives from the
  candidate set) achieve?
- **Hypothesis:** a real, measurable baseline score, expected to be limited by the false-
  merge risk random-negative training does not address (§14's collision structure) —
  this experiment exists specifically to *demonstrate* that gap empirically, not just
  assume it from literature.
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §3.2.
- **Data scope:** train on `matcher_train`, evaluate on `dev_eval` — both subsets of
  `development` per the supervised-matcher development split above (**precondition:** the
  `matcher_train`/`dev_eval`/`calibration_holdout` manifest must exist before this
  experiment can run; it is not created in this research phase — see that policy for why).
  **No `validation` touch at this stage** (per the frozen-validation reuse policy: this
  experiment is expected, by its own hypothesis, to underperform on collision cases and
  exists to *demonstrate* that gap, which is a `development`-only diagnostic, not a
  stage-level comparison worth spending the one early validation read on).
- **Metrics:** macro F0.5, precision, recall, singleton accuracy, non-singleton F0.5,
  candidate recall (inherited from EXP-B004), runtime — all computed on `dev_eval`, never on
  `matcher_train` (in-sample) or `validation`.
- **Acceptance criterion (corrected round 2):** establishes a nonzero, reproducible
  out-of-sample development baseline (on `dev_eval`); no arbitrary numerical target — the
  actual number is the finding.
- **Fail/stop condition:** none — this experiment cannot "fail," only inform.
- **Dependencies:** EXP-F001; the `matcher_train`/`dev_eval`/`calibration_holdout` manifest
  (precondition, not yet built — see the supervised-matcher development split policy above).
- **Order: 6th.**

## EXP-M002 — Hard-negative-augmented GBDT matcher

- **Question:** does training with hard negatives constructed from §14's collision
  structure (same/similar name, conflicting numeric address token) measurably improve
  precision (and macro F0.5) over EXP-M001, consistent with the metric's 2× precision
  weighting?
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §6.
- **Data scope (corrected round 2 — same split as EXP-M001):** hard negatives constructed
  from `matcher_train` only; matcher trained on `matcher_train`; evaluated out-of-sample on
  the same frozen `dev_eval` population EXP-M001 used, so the two are directly comparable.
  **No `validation` touch at this stage** (per the frozen-validation reuse policy; the
  single early `validation` read is reserved for EXP-M003).
- **Ground-truth-completeness caveat (added after audit finding):** Phase 1's referential-
  integrity audit (`docs/DATASET_AUDIT.md` §4) confirms every listed ground-truth target ID
  resolves to a real record — it does **not** independently prove that no unlisted true link
  exists. Hard negatives must therefore be mined only from blocking candidates that are
  confidently *not* ground-truth matches (i.e., excluded from `train_ground_truth.tsv` for
  that S1), treated as a dataset-contract assumption rather than a proven fact, and mining
  should avoid the highest-similarity-score non-matches specifically (the population most
  likely to contain an unlabeled true link if the assumption is ever wrong) in favor of
  collision-structure-driven negatives (same/similar name, conflicting numeric token) —
  which are informative by construction regardless of this caveat.
- **Metrics:** same as EXP-M001, plus explicit false-merge-rate comparison against EXP-M001
  on collision-group-heavy entities specifically.
- **Acceptance criterion:** measurable precision improvement (or at minimum, no regression)
  over EXP-M001 at comparable or better recall.
- **Fail/stop condition:** if hard-negative training degrades recall disproportionately
  (over-conservative matcher), rebalance the negative-mining ratio rather than abandoning
  the approach — the underlying collision evidence (§14) is strong.
- **Dependencies:** EXP-M001 (as the comparison point).
- **Order: 7th.**

## EXP-M003 — Calibration + entity-level threshold policy comparison

- **Question:** does isotonic-regression calibration (per TECHNIQUE_EVIDENCE_MATRIX.md §9),
  fit on `calibration_holdout` per the supervised-matcher development split policy above,
  improve on an uncalibrated raw-score cutoff — and **which threshold policy performs best
  at the
  entity level**: a single global pairwise threshold, or a policy that adapts per entity
  (e.g., by candidate count or by source pattern)? **Revised scope (closes MATERIAL #1):**
  the dossier's architecture hypotheses describe a "per-entity acceptance threshold" while
  the first draft of this experiment only tested a single global shared threshold — these
  are not the same thing, and macro F0.5 is scored per S1 entity, so a threshold that is
  well-calibrated in aggregate can still be poorly suited to, e.g., high-collision entities
  with many similar-scoring candidates versus low-ambiguity entities with one clear
  candidate. This experiment must compare **at least two policies** — (a) a single global
  threshold, and (b) a threshold that varies by a simple candidate-count or source-pattern
  stratum — on the same calibrated scores, not assume (a) is sufficient.
- **Data scope (corrected round 2 — named subsets):** matcher trained on `matcher_train`
  (same fitted matcher as EXP-M002, or a refit under identical conditions); calibration and
  both threshold policies fit on `calibration_holdout` (out-of-sample relative to
  `matcher_train`, and distinct from `dev_eval`, per the supervised-matcher development
  split above); policy comparison itself evaluated on `dev_eval` for the `development`-only
  iteration. **This is the first and only `validation` touch in this experiment sequence** —
  a single confirmatory run comparing the best `dev_eval`/`calibration_holdout`-selected
  policy against EXP-M002's raw-score threshold, per the frozen-validation reuse policy
  above.
- **Metrics:** macro F0.5 (primary), singleton accuracy, non-singleton F0.5, calibration
  curve diagnostic, macro F0.5 broken out by match-multiplicity bucket (to surface whether
  either threshold policy specifically helps/hurts high-multiplicity entities).
- **Acceptance criterion:** measurable macro F0.5 improvement (or singleton-accuracy
  improvement without F0.5 regression) over EXP-M002's raw-score threshold; a clear
  determination of whether the entity-level-adaptive policy beats the global policy by
  enough to justify its added complexity.
- **Dependencies:** EXP-M002.
- **Order: 8th.**

## EXP-M004 — Country-filter ablation (hard vs. soft country signal)

- **Question:** does adding country as a hard blocking/matching filter change candidate
  recall and matcher precision, measured explicitly with and without, per
  DATASET_AUDIT.md §20/§21's own instruction?
- **Research basis:** TECHNIQUE_EVIDENCE_MATRIX.md §12 (country row in the rejected-
  techniques table) — this experiment is what actually resolves that open question rather
  than assuming an answer.
- **Data scope:** `development` (train has no France, so this cannot directly measure
  France-specific risk — it measures the *cost* of a hard filter on US/India data, as a
  proxy per VALIDATION_DESIGN.md's cross-country stress-test note).
- **Metrics:** candidate recall/load with vs. without hard country filtering; matcher
  precision/recall with vs. without country as a hard vs. soft (feature-only) signal.
- **Acceptance criterion:** informs, rather than presupposes, whether country should ever
  be a hard filter — if soft (feature) treatment costs negligible recall/precision versus
  hard filtering on US/India, prefer soft treatment as the France-safer default.
- **Dependencies:** EXP-B004, EXP-M001 (needs both blocking and matching infrastructure).
- **Order: can run in parallel with EXP-M002/M003 once EXP-B004/M001 exist.**

## EXP-M004b — Leave-one-country-out transfer proxy (closes MATERIAL #7)

- **Question:** how much does matcher precision/recall degrade when the matcher/normalizer
  is developed on one seen country and evaluated on the other, as the closest available
  proxy for France transfer risk?
- **Why added:** the first draft of this plan proposed France-related country breakdowns
  *within* `development`, which contains zero France rows by construction — that breakdown
  cannot measure transfer risk at all, only describe US/India separately.
  `docs/VALIDATION_DESIGN.md` already names leave-one-country-out (develop on India-only,
  evaluate on US-only, and vice versa) as the closest available proxy for country-transfer
  cost; the first draft's experiment sequence never actually scheduled it. This experiment
  closes that gap.
- **Research basis:** `docs/VALIDATION_DESIGN.md`'s own "optional stress tests" section.
- **Data scope:** `development` only, partitioned by country for this stress test
  specifically (not `validation`) — explicitly diagnostic, never a substitute for the
  primary `validation_v1` comparison, per `docs/VALIDATION_DESIGN.md`'s own policy.
- **Metrics:** macro F0.5 (and its precision/recall components) when developed on India-only
  and evaluated on US-only, and vice versa, compared against the same-country baseline from
  EXP-M003.
- **Acceptance criterion:** informs, rather than resolves, France risk — a large cross-
  country degradation is a warning sign (motivating more conservative thresholding or
  France-specific monitoring once real France feedback exists, e.g. via the public
  leaderboard as secondary evidence only); a small degradation is reassuring but still not
  proof of France-specific transfer, since France is not merely "a third country" in the
  same distribution as US/India (§17 of the audit).
- **Dependencies:** EXP-M003 (uses the same calibrated pipeline).
- **Order: after EXP-M003, alongside EXP-M004.**

## EXP-M008 — Capacity-one conflict-resolution ablation (closes MATERIAL #9)

- **Question:** does a cheap post-hoc precision control — if two different S1 entities'
  predicted match sets both claim the same S2 or S3 ID, keep only the higher-confidence
  assignment and drop the other — measurably improve precision, consistent with the
  measured training invariant that no S2/S3 target is a true match for more than one S1
  entity (`docs/DATASET_AUDIT.md` §8, `max_s1_degree_for_any_target == 1`)?
- **Why this is distinct from graph/collective ER (and not a reversal of that REJECT):**
  this is a single deterministic conflict check applied after independent per-(S1,
  candidate) scoring — it does not propagate matches transitively, cluster records, or
  require any graph algorithm. The audit's finding was that Phase 1.5's original REJECT of
  graph/collective ER was correct for *transitive-closure/correlation-clustering* methods
  but too broad in also implicitly ruling out this much cheaper, structurally-justified
  check.
- **Research basis:** the measured target-reuse structure itself (`docs/DATASET_AUDIT.md`
  §8) — this is a project-specific inference from Phase-1 evidence, not an imported
  technique from the graph/collective-ER literature reviewed in
  `TECHNIQUE_EVIDENCE_MATRIX.md` §10.
- **Data scope:** `development` only.
- **Metrics:** precision/macro F0.5 with vs. without the conflict-resolution step; frequency
  of actual conflicts observed (if conflicts are rare, the step is low-risk and low-reward;
  if frequent, it is a meaningful precision lever).
- **Acceptance criterion:** no regression in recall (conflicts should be rare enough that
  resolving them costs little); any precision gain is a bonus, not a requirement — this is a
  cheap ablation, not a load-bearing pipeline component.
- **Dependencies:** EXP-M001 (needs a scored candidate set to apply the conflict check to);
  can run any time after that, independent of the M002/M003 calibration work.
- **Order: low-cost, can run in parallel with M002/M003 once EXP-M001 exists.**

## EXP-M005 — Error taxonomy / diagnostic pass (gates any P1 escalation)

- **Question:** decomposing EXP-M003's residual errors into the taxonomy in
  TECHNIQUE_EVIDENCE_MATRIX.md / the research brief §35 (blocking false negative, matcher
  false negative, matcher false positive, singleton false positive, multi-match partial
  miss, ambiguous-collision error, calibration/threshold error) — where is the remaining
  error budget concentrated?
- **Why this gates further work:** this is the evidence that would justify (or rule out)
  promoting any P1 technique (embedding retrieval, transformer reranking, dedicated
  singleton classifier) to active development — per the research brief's own instruction,
  "only if a specific, sizable residual failure mode" is found.
- **Data scope:** `development` for the breakdown; do not re-tune against `validation`
  based on this analysis beyond the confirmatory runs already scheduled above.
- **Dependencies:** EXP-M003.
- **Order: 9th — the decision point for everything after this in the plan.**

---

## Conditional / P1 follow-ups (only triggered by EXP-M005's findings)

- **EXP-B006 (conditional):** character-n-gram/embedding retrieval restricted to the
  specific residual-error population EXP-M005 identifies (e.g., France entities, or the
  zero-lexical-overlap tail specifically) — not a full-population route. Only run if
  EXP-M005 shows a sizable, well-characterized gap of this shape.
- **EXP-M006 (conditional):** narrow transformer/cross-encoder reranker restricted to the
  hard tail only (TECHNIQUE_EVIDENCE_MATRIX.md §4.1) — only run if EXP-M005 shows the
  GBDT+hard-negatives+calibration pipeline has a specific, sizable hard-tail residual that
  a reranker is plausibly positioned to close, AND the pretrained-model compliance
  question (`LICENSE_AND_TOOLING_MATRIX.md`) has been resolved by then.
- **EXP-M007 (conditional):** dedicated singleton classifier — only if EXP-M005 shows
  singleton-specific errors the shared threshold cannot fix.

---

## Explicit non-goals for this plan

- No *transitive-closure/correlation-clustering* graph-ER experiment is scheduled
  (TECHNIQUE_EVIDENCE_MATRIX.md §10 — REJECT, no structural basis in the measured data).
  The one narrow exception is EXP-M008's deterministic capacity-one conflict check, which is
  not a graph algorithm and does not reverse this REJECT.
- `validation_v1`'s `validation` partition is touched **exactly once** in this experiment
  sequence, at EXP-M003 — not "once per stage-level comparison" as the first draft stated
  (which, read literally, permitted a validation touch at every one of B004/M001/M002/M003/
  M004 and was itself the adaptive-reuse risk an independent audit flagged; see the frozen-
  validation reuse policy near the top of this document for the corrected rule).
- No experiment in this plan generates `matching_results.tsv`/`candidate_pairs.tsv` for
  test data, runs test inference, or touches the leaderboard — all of this plan operates on
  `development`/`validation` only, consistent with the §55 no-implementation gate for
  Phase 1.5 and the fact that none of these experiments have been run yet.
- No calibration or threshold value is fit on the same predictions used to train the scoring
  model that produced them — see the supervised-matcher development split policy near the
  top of this document.
- No matcher evaluation number (EXP-M001 onward) is computed on the same subset the matcher
  was trained on — `matcher_train` and `dev_eval` are disjoint by construction, per the same
  policy.
- EXP-B000 is a resource/collision preflight only, run at pilot scale — it must not expand
  into new blocking-technique research; the three routes it estimates resource use for are
  already fixed by `TECHNIQUE_EVIDENCE_MATRIX.md` §1.1-1.3.
- The `matcher_train`/`dev_eval`/`calibration_holdout` manifest referenced throughout the
  matcher experiments (EXP-M001 onward) does not exist yet — it is a precondition for those
  experiments to be built in a future implementation phase, not something this research
  document creates.
