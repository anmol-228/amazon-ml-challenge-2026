# Research Dossier — Phase 1.5

Companion documents: `TECHNIQUE_EVIDENCE_MATRIX.md` (per-technique P0/P1/P2/REJECT detail),
`LICENSE_AND_TOOLING_MATRIX.md` (compliance), `RESEARCH_TO_EXPERIMENT_PLAN.md` (ordered
experiments), `SOURCES.md` (full source list). This document is the narrative synthesis;
it does not repeat every citation inline — see the companions for full sourcing.

## 1. Executive Synthesis

Phase 1 established that this is fundamentally a **multi-route candidate-generation
problem first, and a moderate-difficulty pairwise-classification problem second** — not a
problem where the matching *model* is the primary lever. No single cheap signal (exact
name, exact address, name-token overlap) covers more than a minority of true links; the
combination of (a) token blocking, (b) character n-gram/TF-IDF blocking, and (c) the
numeric-address-token signal Phase 1 discovered as the single strongest cheap signal in the
dataset are complementary, evidence-backed, and should be unioned rather than intersected.
On the matching side, the difficulty distribution (74% of true links in the "hard middle")
is exactly the regime where a GBDT trained on engineered lexical/numeric features with
deliberately constructed hard negatives is expected to perform well, and nothing in the
literature or the data justifies reaching for a large neural/embedding model before that
baseline exists and is measured. The single largest open risk the research could not
resolve is France transfer (§17 of the audit) — no experiment, paper, or technique
eliminates that uncertainty in advance; it can only be reduced by choosing language-robust
building blocks (which are already P0 for other reasons) and measured indirectly once
Phase 2/3 produce a pipeline to test.

## 2. Phase-1 Evidence That Drives Research

The single most decision-relevant facts, restated from `docs/DATASET_AUDIT.md` (all
VERIFIED there): the full Cartesian product is 2.3×10¹³ pairs (§2, blocking is mandatory);
nearly half of entities need 4+ correct targets recovered simultaneously (§6, rules out
top-1/argmax matching); zero cross-entity target reuse (§8, rules in a simple validation
split and rules out any need for graph/collective ER); only 0.83% of true links are
"trivially easy" on both name and address (§13); 10% of true links share zero name tokens
(§11); a numeric-address-token exact-match signal covers 63.49% of true links, far more
than full address equality at 7.44% (§12); normalized names are only 88-93% unique within a
source while (name, address) pairs are 99-100% unique (§14, joint disambiguation is
mandatory); country is 100% consistent on every training true link but has zero training
examples for France, which is ~15% of every test table (§9, §17).

## 3. Tier-A / Tier-B / Tier-C Research Questions

**Tier A (researched deeply):** candidate generation/blocking strategy and union design;
name/address similarity and normalization risk; classical matcher family (GBDT) and
feature design; hard-negative construction; multi-match decision logic; F0.5-aware
calibration/thresholding; the pretrained-model compliance question.

**Tier B (researched moderately):** transformer/cross-encoder matching as a hard-tail
reranker; dense/embedding retrieval as a blocking route; multilingual/France transfer
mechanisms; MinHash/LSH and PPJoin as alternate set-similarity retrieval; singleton/
abstention as a formalized selective-prediction problem.

**Tier C (checked lightly, then deprioritized or rejected):** graph/collective ER; sorted
neighborhood/canopy clustering; DeepMatcher-era RNN/attention matchers; `dedupe`/
`recordlinkage` as production backbones. See `TECHNIQUE_EVIDENCE_MATRIX.md` §12 for the
full rejected-technique table with reasons.

## 4. Entity Resolution Foundations

The Fellegi-Sunter (1969) probabilistic framework — model an agreement vector across
fields as a mixture of a match distribution and a non-match distribution, estimate
parameters (classically via EM), and threshold the resulting likelihood ratio — remains the
conceptual backbone of the field and underlies mature tooling like Splink. Modern practice
generally replaces the hand-derived mixture model with a discriminatively trained
classifier (GBDT or neural) over engineered or learned pair features, which is more
flexible for nonlinear feature interactions but loses some of Fellegi-Sunter's built-in
missing-data handling elegance. For this project, both are viable and not mutually
exclusive: a GBDT is prioritized as the primary matcher (Tier A, strong data fit — see §20
below), with a Fellegi-Sunter-style model (via Splink) as a credible, well-evidenced
alternative baseline for cross-checking rather than a rejected option.

## 5. Candidate Generation / Blocking

This is the deepest and most decision-critical research area, detailed fully in
`TECHNIQUE_EVIDENCE_MATRIX.md` §1. Summary of the P0 conclusion: **three routes, each
individually evidenced and designed to be unioned** — token blocking (§1.1), character
n-gram/TF-IDF blocking (§1.2,
directly evidenced by Sparkly's VLDB 2023 result that simple TF-IDF blocking beat 8 SOTA
blockers including DL-based ones at a comparable 10-26M-row scale), and numeric-address-
token blocking (§1.3, our own strongest measured signal) — **each individually targets a
different Phase-1-measured population, but their actual joint recall and overlap has not
been measured and should be treated as this plan's central open hypothesis, not an
established fact, until `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-B004 measures it directly**
(correction added after independent audit review flagged the first draft's "complementary,
evidence-backed" framing as overstating what Phase 1 actually established about the union
specifically, as opposed to each route individually). Dense/embedding retrieval (§1.7)
and learned/supervised blocking (§1.8) are demoted to P1/P2: the literature is genuinely
split on whether embeddings beat a strong lexical baseline (Zeakis et al. 2023's systematic
study vs. Sparkly's contradictory result), and Phase 1's own difficulty distribution (§13)
shows the "hard middle" is dominated by moderate-lexical-overlap cases (bucket D, ≥0.5
Jaccard on both fields), not zero-overlap cases — exactly the population lexical methods
already handle. MinHash/LSH and PPJoin-style set-similarity joins (§1.6) are well-evidenced
in general but not shown to outperform the simpler TF-IDF route already prioritized, so
they sit at P1 as a benchmark comparison, not a P0 requirement.

## 6. Name Normalization and Similarity

Character-level (Levenshtein, Jaro-Winkler) and token-level (Jaccard, TF-IDF cosine)
similarity are both P0 feature-engineering baselines — cheap, well-understood, directly
reproducing the statistics Phase 1 already validated at full population scale (§11), and
implementable via RapidFuzz (MIT-licensed, C++-backed, fast enough for per-candidate
scoring after blocking). The main *risk* area is normalization policy for legal suffixes/
abbreviations: practitioner literature and Phase 1's own collision evidence (§14, normalized
names only 88-93% unique) agree that aggressive suffix-stripping can *increase* the
collision problem it's meant to solve unless paired with address/numeric disambiguation.
France introduces a specific instance of this risk (§17's "SARL" collision pattern) with no
training data to validate a France-specific normalization rule against — the safe default
is to treat legal-form tokens as a general class (present in all three countries, different
vocabulary) rather than hard-coding a country-specific abbreviation dictionary, which the
research brief explicitly warns against as an anti-pattern.

## 7. Address Matching

The dominant, project-specific finding is the numeric-address-token signal (§12): 63.49% of
true links share an exact numeric-token set even though full normalized-address equality is
only 7.44% — the surrounding text (street/city names, transliteration, abbreviations)
varies far more than the numbers do. General address-matching literature and practitioner
sources (Babel Street, Robin Linacre) corroborate the pattern of matching numeric/
alphanumeric fields (house number, unit, postal-like codes) separately from named fields
rather than folding everything into one text-similarity score — consistent with, not
contradicting, what Phase 1 measured directly. This signal is P0 both as a blocking route
(§1.3) and as a matching-stage feature, including its *negative* form: a name-similar
candidate pair with **conflicting** numeric tokens is a specific, interpretable, high-value
feature for rejecting false merges, directly serving the metric's 2× precision weighting.
No geocoding, external postal-validation service, or address-parsing API is used or needed
for any of this — it is pure text-token extraction, fully compliant with the external-
lookup ban.

## 8. Classical Matching Models

GBDT (LightGBM/XGBoost/CatBoost, all MIT/Apache-2.0, all trivially within the size
constraint) is the P0 matcher family: strong, reproduced fit for exactly the setting Phase
1 measured (74% of true links needing a learned combination of several moderate-strength
engineered features, §13), native missingness handling (relevant given ~4.4% target-address
missingness on true links, §10), and native categorical support (relevant for `country`/
source features). Splink (Fellegi-Sunter + EM, MIT-licensed, laptop-to-100M+-record scale
documented) is a credible P1 second baseline for cross-checking rather than a competing
final choice — the two paradigms are complementary evidence, not mutually exclusive
architecture options. `dedupe` and `recordlinkage` are real, permissively licensed tools but
workflow-mismatched (active-learning/smaller-batch design center) to this project's already-
labeled, tens-of-millions-of-candidate-pairs scale, so they are retained only as reference/
comparison tools (P2).

## 9. Deep / Transformer Entity Matching

Ditto and related transformer cross-encoder matchers (Li et al., VLDB 2020; VLDB Journal
2023 extension) report strong benchmark results (up to 29% F1 improvement over prior SOTA
on Magellan-family benchmarks), but those benchmarks are far smaller (hundreds of thousands
of pairs) than the tens of millions of candidate pairs this project's blocking stage is
likely to produce, and cross-encoder inference cost scales linearly with candidate-pair
count unlike a GBDT's near-negligible per-pair cost. The evidence does not establish that a
transformer matcher is *necessary* — only that it is a plausible, evidence-backed hard-tail
reranker if the GBDT baseline's own error analysis (EXP-M005 in the experiment plan) shows
a specific, sizable residual gap it is positioned to close, and only after the pretrained-
model compliance question is resolved. DeepMatcher-era RNN/attention architectures are
superseded by both the GBDT baseline and Ditto-style transformers on the available evidence
and are rejected outright.

## 10. Hard-Negative Strategy

Direct, low-risk application of two pieces of evidence together: Phase 1's own collision
structure (§14 — normalized-name collision groups up to ~460-way, near-perfect joint
(name, address) uniqueness) identifies *exactly* which negatives are hard and informative
for this data (same/similar name, different business, distinguishable mainly by address/
numeric-token disagreement); the general contrastive-learning literature (Kalantidis et al.
2020; NV-Retriever 2024) warns that naive "hardest similarity-ranked" mining can
mislabel true matches as negatives if the candidate source is noisy. **Correction (post-
audit):** the first draft of this dossier overstated the mitigation here — Phase 1's
ground-truth integrity audit (§4) verified that every *listed* target ID resolves to a real
record (referential integrity), which is not the same claim as "no unlisted true link
exists" (label completeness). The two are easy to conflate but distinct: referential
integrity is fully VERIFIED; label completeness is an assumption this project relies on but
did not and cannot independently prove from the data alone. The practical mitigation stands
regardless — mine hard negatives from collision structure (same/similar name, conflicting
numeric token) rather than from raw highest-similarity-score non-matches specifically,
since the latter is the population most exposed if the completeness assumption is ever
wrong — and negatives are mined only from `development`, never `validation`. Priority: P0,
directly targeting the metric's precision weighting; see
`RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-M002 for the revised, more cautious mining procedure.

## 11. Multi-Match Decision Logic

No single paper "solves" macro-F0.5-optimal set-valued prediction for this exact problem
shape; the research-backed direction is independent calibrated pairwise scoring per
candidate followed by a *tuned* acceptance threshold (all candidates clearing threshold are
accepted, not top-1/argmax and not a hard one-match-per-source rule — both explicitly
unsupported by the measured data: 47.96% of entities need 4+ matches, §6; 80.48% of non-
singleton entities need matches in *both* S2 and S3 simultaneously, §7). **Correction (post-
audit):** the phrase "per-entity acceptance threshold" in the first draft of this dossier
was ambiguous with, and in the experiment plan's first draft inconsistent with, a *single
global* shared threshold — these are materially different policies (a global threshold vs.
one that adapts by candidate count or source pattern), and macro F0.5's per-entity scoring
means the choice matters. This is now treated explicitly as an open comparison, not a
foregone conclusion either way: `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-M003 compares a global
threshold against at least one entity-adaptive policy on calibrated `development` scores
before either is adopted. This is necessarily an empirical, project-specific question, not
one the literature resolves in advance — the literature contributes the calibration/
thresholding *toolkit* (isotonic regression, Platt scaling), not the specific policy.

## 12. Singleton / Abstention

Proportional to the measured 5.58% singleton rate (§5) — real but not dominant (a perfect-
singleton-only strategy caps achievable score around 0.944, §19). The research-backed
direction treats singleton prediction as the natural zero-candidates-clear-threshold case
of the same multi-match decision logic (§11 above, Chow's 1970 reject-option framework and
modern selective-prediction literature), not as a structurally separate classifier that
must be built proactively. A dedicated singleton classifier is demoted to a conditional P1/
P2 follow-up, triggered only if error analysis (EXP-M005) shows singleton-specific failures
the shared threshold cannot fix.

## 13. F0.5 / Calibration

The metric's 2× precision weighting is the throughline connecting hard-negative mining
(§10), numeric-token contradictory evidence (§7), and calibration together: GBDTs are
well-documented to produce uncalibrated probabilities (Niculescu-Mizil & Caruana), and an
uncalibrated raw score makes threshold tuning for the multi-match/singleton decision (§11/
§12) unreliable across strata (country, source, difficulty bucket). Isotonic regression is
preferred over Platt scaling by default given `development`'s scale (far larger than the
~2000-case regime where Platt's variance advantage matters), with Platt as a fallback for
any thin stratum where isotonic regression risks overfitting.

## 14. France / Multilingual Transfer

What appears structurally robust (per §17's own careful labeling): France's basic
statistical profile (name-uniqueness rate, missingness, numeric-token presence, length
scale) falls within the same range as US/India — no wild outlier. What remains genuinely
unknown: whether France's actual noise patterns (accented characters, department/postal-
code conventions, French legal-form vocabulary beyond the one observed "SARL" pattern,
street-numbering conventions) behave the same way under our normalization/similarity
pipeline, since zero France ground truth exists anywhere to check against. The research-
backed mitigation is indirect: prioritize techniques that are inherently script/vocabulary-
agnostic (character n-grams, numeric-token extraction, Unicode-safe normalization) — which
are already P0 for other, data-driven reasons — rather than building any France-specific
rule or dictionary. Multilingual entity-matching literature (Polish product-matching
benchmark, cross-lingual entity-linking transfer studies) confirms transfer is *possible*
but imperfect and pair-dependent; France's use of the Latin alphabet and a legal-form
vocabulary pattern structurally similar to (if lexically different from) US/India's makes
this an easier transfer case than the cross-script examples common in that literature, but
"easier" is not "verified."

**Added after audit review:** the first draft of this research proposed measuring France-
related risk via country breakdowns computed *within* `development`, which by construction
contains zero France rows and therefore cannot measure transfer risk at all — only describe
US and India separately. `docs/VALIDATION_DESIGN.md` already names the correct (if still
indirect) proxy: a leave-one-country-out stress test, developing on one seen country and
evaluating on the other. That proxy existed in the source validation design but had not
actually been scheduled in the Phase 1.5 experiment plan; it now is
(`RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-M004b), explicitly as a diagnostic, never a substitute
for `validation_v1`.

## 15. Scalability

What can plausibly run on the measured hardware (Ryzen 5 7500F, 31.6GB RAM, RTX 5060 Ti
with VRAM unverified): all P0 techniques (inverted-index token blocking, single-node
sparse TF-IDF top-k retrieval, numeric-token exact-set indexing, GBDT training/inference on
an engineered feature table) are CPU-bound operations of the same general shape as what
Phase 1 already ran successfully at full ~17M-row scale for audit purposes (with one
documented memory-pressure incident that was caught and rewritten as a streaming pass — the
same discipline applies to Phase 2). **Correction (post-audit):** the first draft of this
dossier characterized these techniques as "memory-moderate" more confidently than the
evidence supports — Phase 1's streaming *audit* scans are read-then-discard passes; a TF-IDF
index, a multi-million-row candidate file, and a pair-feature table are all *retained*
in-memory or on-disk artifacts with materially different footprints that have not been
measured for this specific workload. Feasibility here is a plausible expectation, not an
established fact, and each blocking experiment now carries an explicit scale-budget
requirement (peak RAM, disk, candidate-row count — see the cross-cutting policy in
`RESEARCH_TO_EXPERIMENT_PLAN.md`) specifically so this gets measured rather than assumed.
What is currently unverified and should not be assumed: any GPU-dependent route (embedding retrieval at scale, transformer reranking)
requires an actual `nvidia-smi` VRAM reading before being scoped, since Phase 1's WMI-based
figure was explicitly flagged as unreliable. FAISS and hnswlib (both permissively licensed)
are documented to handle our ~10M-vector scale comfortably *if* an embedding route is ever
adopted, but that adoption is not yet justified by the evidence (§5/§9 above).

## 16. License / Compliance

Full detail in `LICENSE_AND_TOOLING_MATRIX.md`. Summary: every classical library considered
(LightGBM, XGBoost, CatBoost, RapidFuzz, Splink, scikit-learn, FAISS, hnswlib) is CLEAR —
MIT or Apache-2.0, verified directly from each project's own LICENSE file, not from a
secondary description. The three embedding models surveyed as possible P1/P2 candidates
(paraphrase-multilingual-MiniLM-L12-v2, multilingual-e5-large, LaBSE) are each individually
CLEAR on license (Apache-2.0/MIT) and comfortably within the 8B-parameter ceiling. The one
unresolved item is a **policy-level compliance interpretation, not a license fact**:
whether using *any* pretrained general-purpose model is intended to be permitted at all
under the fair-play rule, versus only fine-tuning-from-scratch on supplied data. The most
natural reading of the two official PDFs (external-lookup/enrichment ban applies to the
*supplied entities specifically*, not to general pretrained representations; the model
restriction is stated as a property of the final model's license/size, not its training
provenance) supports permissibility, but this is Phase 1.5's own INFERRED reading, not a
verified organizer clarification — status is **REVIEW REQUIRED**, and no P0 recommendation
in this dossier depends on it (the P0 techniques are all pure lexical/statistical methods
requiring no pretrained model at all).

## 17. Contradictory Evidence

The most important disagreement found: Zeakis et al. (PVLDB 2023, systematic 12-model
embedding study) treat pretrained embeddings as a serious, sometimes-superior blocking/
matching signal, while Sparkly (PVLDB 2023, same venue, same year) reports a "surprisingly
strong" pure-lexical TF-IDF blocker beating 8 state-of-the-art blockers including DL-based
ones. Both are credible, peer-reviewed, VLDB-published, and neither supersedes the other —
they were evaluated on different benchmark suites with different characteristics. This
dossier treats the disagreement as genuinely unresolved for our specific data (per the
research brief's contradictory-evidence rule) and resolves the *practical* question by
deferring to Phase 1's own difficulty-distribution evidence (§13: the hard middle is
dominated by moderate-lexical-overlap cases, which lexical methods already reach) rather
than assuming either paper's benchmark result transfers directly. A second, smaller
disagreement: NV-Retriever's finding that naive hard-negative mining can mislabel ~70% of
"hardest" candidates as negatives when they're actually unlabeled positives is a real risk
in general, but is substantially mitigated here by Phase 1's ground-truth integrity audit
(§4, zero orphaned targets) — the risk that remains is scoped narrowly (which non-ground-
truth *blocking candidates* get used as negatives), not the ground truth itself.

## 18. Architecture Hypotheses

Presented as coherent families, none selected as final, each with its falsification path
in `RESEARCH_TO_EXPERIMENT_PLAN.md`.

**Hypothesis A — Multi-route lexical blocker → engineered features → calibrated GBDT.**
Pipeline: union of token blocking, character-n-gram/TF-IDF blocking, and numeric-address-
token blocking (deduplicated) → the explicit feature inventory in
`RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-F001 → GBDT trained with hard negatives from collision
structure (mined conservatively, per the ground-truth-completeness caveat in §10 above) →
isotonic calibration fit on a held-out calibration subset (never the GBDT's own training
predictions, per the leakage-prevention policy) → a tuned acceptance threshold, with the
choice between a global and an entity-adaptive policy settled empirically (EXP-M003), not
assumed. *Why plausible:* directly matches every P0 conclusion
above; requires no pretrained model at all (sidesteps the compliance review entirely).
*Supporting evidence:* Sparkly (blocking), GBDT-on-engineered-features literature
(matching), Phase-1's own difficulty distribution. *Contradictory evidence:* none directly
against this family; the open question is whether its ceiling is high enough, which only
EXP-M001-M005 can answer. *Candidate-generation strategy:* §5 above. *Expected recall
characteristics:* high on buckets A-D (§13, ~58% of true links), uncertain on bucket F
without further feature engineering. *Complete-coverage characteristics:* depends on
EXP-B004's measured union recall at high match-multiplicity. *Precision controls:* hard
negatives + calibration + numeric-token contradictory-evidence feature. *Singleton
strategy:* threshold-implied (§12 above). *Multi-match logic:* per-candidate threshold,
no top-k cap. *France strategy:* character-n-gram/numeric-token routes are inherently
script-agnostic. *Scaling profile:* CPU-only, fits measured hardware. *Engineering cost:*
LOW-MEDIUM. *Time cost:* the fastest path to a first real validation score. *License/
compliance:* fully CLEAR, no review-required dependency. *Biggest unknown:* whether bucket
F (37.57% of true links, weak lexical overlap on both fields) is reachable by engineered
lexical/numeric features alone. *Biggest failure mode:* precision loss on high-collision
generic-name entities if the numeric-token feature is weak/missing for a given pair.
*Earliest falsification:* EXP-M001's first `validation` run.

**Hypothesis B — Hypothesis A + hard-tail embedding features (not retrieval).** Same
pipeline as A, plus a cosine-similarity-between-embeddings feature added to the GBDT's
feature table for candidates the lexical features leave ambiguous, without using
embeddings for retrieval/blocking itself. *Why plausible:* isolates embeddings' potential
value (semantic signal on bucket F) from their biggest risk (semantic false positives
gating recall at the blocking stage). *Supporting evidence:* Zeakis et al.'s finding that
embeddings carry real signal. *Contradictory evidence:* Sparkly's result that a strong
lexical baseline may already capture most of the available signal; the compliance-review
status of pretrained models generally. *Biggest unknown:* whether the added feature moves
macro F0.5 enough to justify its compute/compliance-review cost. *Earliest falsification:*
compare against Hypothesis A's EXP-M001 score using the same held-out check.

**Hypothesis C — Hypothesis A + narrow transformer hard-tail reranker.** Same blocking and
GBDT-matcher backbone as A, with a transformer cross-encoder applied *only* to the specific
residual-error population EXP-M005 identifies (e.g., French entities or the lowest-
confidence GBDT decile), not the full candidate set. *Why plausible:* Ditto's benchmark
evidence for pairwise transformer matching is strong; scoping to a narrow tail avoids the
inference-cost problem that would make full-candidate-set transformer matching infeasible
at our scale. *Contradictory evidence:* benchmark scale gap (§9 above); unresolved license/
compliance review for whichever specific checkpoint would be used. *Biggest unknown:*
whether EXP-M005 even identifies a residual gap this technique is positioned to close.
*Earliest falsification:* EXP-M005 itself — if no such gap is found, this hypothesis is not
pursued at all.

**Hypothesis D — Hypothesis A with Splink (Fellegi-Sunter/EM) as the primary matcher
instead of GBDT.** Same blocking backbone, but the matching stage uses Splink's EM-
estimated match weights instead of a discriminatively trained GBDT. *Why plausible:*
mature, well-evidenced, interpretable, and license-clear; a genuinely different modeling
paradigm that could behave differently on the "hard middle" than a GBDT. *Supporting
evidence:* Fellegi-Sunter's long track record; Splink's documented scale (laptop to
100M+ records). *Contradictory evidence:* no head-to-head evidence found comparing
Fellegi-Sunter-style EM models against GBDTs specifically on a difficulty distribution like
ours; GBDT's native handling of nonlinear feature interactions and missingness is a
plausible (not proven) advantage for the 74% hard-middle share. *Biggest unknown:* relative
performance is untested for this data; treated as a **parallel P1 cross-check**, not a
replacement for Hypothesis A, given the frozen validation split makes a direct comparison
cheap to run once both exist. *Earliest falsification:* a `development`-only comparison
run against Hypothesis A's GBDT at the same feature-set stage.

## 19. Council Findings

SKIPPED — evidence did not justify council overhead. The research surfaced one genuine,
consequential disagreement (§17 above, embeddings vs. lexical blocking), but Phase 1's own
difficulty-distribution evidence (§13) was sufficient to resolve the *practical* question
(defer embeddings to P1, not P0) without needing multiple simulated perspectives — the
resolution rests on project-specific data already in hand, not on adjudicating between
external authorities.

## 20. Independent Technical Audit

An independent, read-only, high-effort technical audit was run in an isolated command-line
environment, directed at the four `docs/research/` artifacts against the canonical
project-truth files. A concise, tool-neutral summary of every issue identified and how each
was repaired is preserved at `docs/research/audit/PHASE_1_5_AUDIT_SUMMARY.md`, consistent with
this project's artifact-hygiene policy — kept as evidence rather than discarded, per the
instruction to preserve provenance and not cherry-pick favorable outputs, without retaining a
raw tool-specific session transcript in the project.

**Findings: 2 CRITICAL, 12 MATERIAL, 3 MINOR, 4 NEEDS VERIFICATION.** All CRITICAL and
MATERIAL findings were repaired in this document and its three companions before this
report was finalized — see the "Correction (post-audit)" / "revised" annotations inline
throughout this dossier and `RESEARCH_TO_EXPERIMENT_PLAN.md` for exactly what changed and
why. Summary of what was found and fixed:

- **CRITICAL — calibration/threshold leakage unspecified.** The first draft's EXP-M003 said
  "calibrate and tune threshold on `development`" without requiring out-of-sample
  predictions, risking an optimistic, leaked macro-F0.5 estimate. **Fixed:** a
  matcher-training/calibration-holdout split (or k-fold cross-fitting) is now a named
  cross-cutting policy in `RESEARCH_TO_EXPERIMENT_PLAN.md`, binding on every calibration
  step.
- **CRITICAL — `candidate_pairs.tsv` final-set provenance not operationalized.** The
  official requirement that this file be the *exact* final candidate set, not a pre-pruning
  intermediate, had no explicit tracking mechanism across EXP-B004→B005→matcher stages.
  **Fixed:** an explicit provenance-contract policy was added, to carry into Phase 2's
  actual implementation.
- **MATERIAL findings, fixed:** entity-level threshold policy was under-specified relative
  to the dossier's own "per-entity" language (EXP-M003 now compares global vs.
  candidate-count/source-aware policies explicitly); the frozen `validation` partition was
  scheduled for adaptive reuse across five experiments (now touched exactly once, at
  EXP-M003); the hard-negative-mining section overstated what the ground-truth referential-
  integrity audit proves about label completeness (corrected, with a more conservative
  mining procedure); the P0 blocking routes' complementarity was stated as established
  rather than as EXP-B004's own hypothesis (corrected); EXP-B002 treated address n-grams as
  optional rather than a required ablation protecting the exact population the route exists
  for (fixed); blocking metrics (macro per-entity candidate recall, complete-coverage rate)
  lacked formal definitions (added); the France mitigation plan never actually scheduled the
  leave-one-country-out proxy `docs/VALIDATION_DESIGN.md` itself names (added as
  EXP-M004b); country-feature handling for the unseen France category was ambiguous between
  a raw categorical encoding and an agreement feature (resolved in favor of agreement-by-
  default); the graph/collective-ER REJECT was broader than the evidence supported — a cheap
  capacity-one conflict-resolution check is licensed by the same target-reuse evidence
  without requiring graph modeling (added as EXP-M008, narrow carve-out documented in
  `TECHNIQUE_EVIDENCE_MATRIX.md` §10); scale-feasibility claims ("memory-moderate") were
  more confident than Phase 1's audit-scan precedent actually supports (softened, explicit
  peak-RAM/disk/candidate-row budgets now required per experiment); EXP-F001's feature set
  was named vaguely ("RapidFuzz-based scoring") rather than as the explicit inventory
  `TECHNIQUE_EVIDENCE_MATRIX.md` §2 actually calls for (made explicit); singleton-specific
  candidate-load/max-score diagnostics were folded into post-matching accuracy rather than
  treated as a first-class blocking-stage metric (now a named part of the candidate-load
  metric definition).
- **MINOR findings, addressed:** EXP-M001's validation touch was low-information for its
  cost (removed — EXP-M001 is now `development`-only); occasional over-strong "France
  robust" wording was tightened to "language-agnostic mechanism, transfer unverified"
  throughout; the license matrix's `CLEAR` label risked being read as full submission-
  compliance rather than component-level license/scale compatibility (a scope clarification
  was added to `LICENSE_AND_TOOLING_MATRIX.md`).
- **NEEDS VERIFICATION items, left open (correctly — these are not literature-resolvable):**
  pretrained-model permissibility (already tracked as `REVIEW REQUIRED`, unchanged); exact-
  checkpoint compliance verification at adoption time (already noted as a future step, not
  something to resolve now); numeric-token blocking's actual selectivity (this is precisely
  what EXP-B003's collision-group analysis, already scheduled, exists to measure); the
  frozen-validation reuse *policy* itself needed an explicit decision rather than an implicit
  one (resolved by the "touched exactly once, at EXP-M003" policy above — no longer an open
  item).

No further escalation occurred: none of the audit's findings represented an unresolved
architecture-changing disagreement, a conflict between reputable evidence, or a
validation-methodology question beyond what the findings themselves specified how to fix — the
repair path was clear from the findings alone, so escalation was not warranted per the
research brief's own escalation criteria (§47).

**Round 2 (human-directed methodology repair, no new independent audit run):** a second pass,
requested directly rather than via another independent audit, found the round-1 repairs still had four
residual precision problems: (1) the candidate-recall metric definition still let a true
singleton's "recall" and its false-candidate exposure blur together; (2) the
`candidate_pairs.tsv` provenance contract described the cutoff point ambiguously relative to
the decisive matching stage; (3) EXP-M001/M002 had no explicit out-of-sample evaluation
partition distinct from whatever the matcher trained on, and stale "validation macro F0.5"
wording survived from before `validation` touches were removed from those experiments; (4)
no lightweight scale/collision preflight existed before full-scale route materialization.
All four are fixed in `RESEARCH_TO_EXPERIMENT_PLAN.md` directly (new formal metric
definitions separating link recall / non-singleton candidate recall / non-singleton complete
coverage / singleton exposure rate / singleton candidate load / overall candidate load; a
precisely-bounded provenance contract naming the upstream-pruning vs. decisive-matching
boundary explicitly; a three-way `matcher_train`/`dev_eval`/`calibration_holdout` split
specified as an EXP-M001 precondition, not yet built; and a new EXP-B000 preflight). This
round did not warrant a new independent-audit invocation — the four issues were specific, well-scoped,
and did not surface a new ambiguity requiring independent adjudication.

## 21. Research-to-Experiment Sequence

See `RESEARCH_TO_EXPERIMENT_PLAN.md` in full (revised post-audit — see §20 above; further
revised after teammate-research reconciliation — see §29 below). Order:
EXP-B001 (token-blocking baseline, now with a word-token joint TF-IDF implementation note) and
EXP-B001R (reverse doc→S1 retrieval, new, parallel with B001) → EXP-B002 (character
n-gram/TF-IDF blocking, name AND address both required) and EXP-B003 (numeric-token blocking,
parallel) → EXP-B004 (multi-route union including the reverse route, `development`-only, the
Phase-2 headline deliverable) → [conditional EXP-B005 pruning] → EXP-F001 (explicit
feature-inventory construction, now including numeric-conflict/entity-frequency/co-duplicate
features) → EXP-M001 (GBDT baseline, random negatives, `development`-only) → EXP-M002
(hard-negative-augmented GBDT, conservative mining, `development`-only) and EXP-M008
(capacity-one conflict-resolution ablation, low-cost, parallel) → EXP-M003 (calibration on a
held-out subset + entity-level threshold-policy comparison — **the first and only
`validation` touch in this sequence**) → EXP-M004 (country-filter ablation) and EXP-M004b
(leave-one-country-out France-transfer proxy, both `development`-only, parallel with each
other) → EXP-M005 (error-taxonomy diagnostic, the gate for any P1 escalation) → conditional
EXP-B006/M006/M007, and conditionally EXP-B001I (Indic-normalization ablation, gated on the
Phase-3 `matcher_train` manifest existing), only if EXP-M005 or the manifest's availability
respectively justify them.

## 22. Recommended Phase-2 Starting Mission

Run the lightweight EXP-B000 resource/collision preflight first (round-2 addition — a pilot-
scale scale/collision estimate, not a new research question), then implement and benchmark
the three P0 blocking routes independently (EXP-B001 token blocking, EXP-B002
character-n-gram/TF-IDF blocking, EXP-B003 numeric-address-token blocking) against the
`development` partition of `validation_v1` only. These are independent routes — no
dependency between them, so they can be built and measured in any order (B001 first only
because it doubles as a normalization sanity check for the other two). For each, compute the
full blocking-metric set as formally defined in `RESEARCH_TO_EXPERIMENT_PLAN.md` (round 2:
link recall; macro per-entity candidate recall and complete true-link coverage, both
restricted to non-singleton entities; singleton candidate exposure rate and singleton
candidate load, tracked separately since a true singleton's false-candidate risk is a
distinct concern from non-singleton recall; overall candidate load; reduction ratio),
broken out by country, target source, match multiplicity, and missing-address status where
feasible. Vary primarily: for EXP-B002, the top-k retrieval parameter and n-gram size (start
at 3-grams, per Sparkly's precedent) — and address n-grams are a required ablation, not
optional; for EXP-B003, whether to require exact numeric-set agreement or allow partial
overlap, and how to handle entities with no extractable numeric tokens. Then union the three
into EXP-B004 and measure complete true-link coverage specifically (the metric most directly
threatened by the ~48% of entities needing 4+ matches), alongside singleton candidate
exposure (unioning routes can increase false-candidate risk on true singletons even as it
improves coverage). **Stop/go evidence:** if the union's complete true-link coverage and
candidate load are both workable (recall high, load within what a GBDT-scale matcher can
process) and singleton candidate exposure has not grown disproportionately, proceed directly
to feature engineering and the GBDT baseline (EXP-F001/EXP-M001, both of which now require
the `matcher_train`/`dev_eval`/`calibration_holdout` split manifest to exist first) without
touching any embedding or learned-blocking technique. Only if a
specific, well-characterized recall gap remains after the union — not before — does
investigating a P1 technique (embedding retrieval scoped to the residual, or learned-
blocking pruning) become justified.

## 23. What NOT To Do

- Do not intersect blocking routes (name AND address conditions) — this destroys recall;
  union only (§5, §6 of the research brief).
- Do not use argmax/top-1 or a hard one-match-per-source rule for the matching decision —
  both are directly contradicted by measured match-multiplicity and cross-source structure
  (§6, §7 of the audit).
- Do not hard-filter on country before measuring the cost explicitly (EXP-M004) — France is
  ~15% of every test table and has zero training examples to verify the filter against.
- Do not build a France-specific or India-specific abbreviation dictionary — prioritize
  script/vocabulary-agnostic techniques instead (already P0 for other reasons).
- Do not adopt an embedding-based blocking route as a P0 dependency before the lexical/
  numeric union's recall gap (if any) is actually measured — the literature is genuinely
  split (§17) and Phase 1's own difficulty distribution does not clearly require it.
- Do not commit to any pretrained-model-dependent technique before the compliance-
  interpretation question in `LICENSE_AND_TOOLING_MATRIX.md` is explicitly revisited.
- Do not build graph/collective-ER logic (transitive closure, correlation clustering) — no
  structural basis in the measured target-reuse data (§8 of the audit). The one narrow,
  non-graph exception is EXP-M008's deterministic capacity-one conflict check.
- Do not touch `validation_v1`'s `validation` partition anywhere in this experiment sequence
  except once, at EXP-M003 — not "once per stage" (the first draft's ambiguity here was
  itself an audit finding; see §20).
- Do not fit a calibrator or a decision threshold on the same predictions used to train the
  scoring model that produced them — use the matcher-training/calibration-holdout split
  named in `RESEARCH_TO_EXPERIMENT_PLAN.md`.
- Do not treat the three P0 blocking routes' union as already proven complementary — EXP-B004
  is the experiment that establishes (or refutes) that, not an assumption to build on before
  it runs.

## 24. Genuine Open Questions

- Whether France's real noise patterns transfer as well as its structural statistics
  suggest — unresolvable without France ground truth, which does not exist anywhere.
- Whether the multi-route blocking union's candidate load is actually workable at full
  scale on this hardware — not measured yet, only estimated.
- Whether GBDT-on-engineered-features reaches a competitive macro F0.5 without any neural
  component — the central empirical question the whole P0 plan is designed to answer
  quickly.
- Whether the pretrained-model compliance interpretation in `LICENSE_AND_TOOLING_MATRIX.md`
  is correct — genuinely unresolved absent an organizer clarification.
- Whether numeric-address-token collision groups (never measured in Phase 1) are as
  benign as name-collision groups or worse — direct target of EXP-B003.

## 25. Research Artifacts Created

- `docs/research/RESEARCH_DOSSIER.md` (this file)
- `docs/research/TECHNIQUE_EVIDENCE_MATRIX.md`
- `docs/research/LICENSE_AND_TOOLING_MATRIX.md`
- `docs/research/RESEARCH_TO_EXPERIMENT_PLAN.md`
- `docs/research/SOURCES.md`
- `docs/research/audit/PHASE_1_5_AUDIT_SUMMARY.md` (a concise, tool-neutral summary of the
  independent audit's findings and repairs, preserved as evidence)

## 26. Source Summary

Approximate counts, by category, from `SOURCES.md` (not a quality metric): primary papers
(VLDB/arXiv/ACM, peer-reviewed or peer-venue preprints) ≈ 35; surveys ≈ 6; official docs
(project's own canonical files, already-VERIFIED, re-read not re-counted here) — see §2
above; library/model official documentation and license files (GitHub/PyPI/HuggingFace,
each individually fetched and verified) ≈ 15; practitioner/industrial writeups (Tier 2/3,
used only to corroborate, never as sole evidence) ≈ 6.

## 27. Compliance Audit

- No supplied business name or address was searched externally — every web query used
  generic technical/algorithm/library/model terminology only. VERIFIED (every query logged
  in this session's tool calls used generic terms).
- No external entity/business lookup was performed. VERIFIED.
- No geocoding was performed or proposed as a production technique (address matching is
  researched and recommended as pure text/token extraction, explicitly without geocoding,
  per §7 above). VERIFIED.
- No external business/entity dataset was joined or proposed for joining against supplied
  records. VERIFIED.
- No competitor solution was copied. Two competitor-shaped GitHub repository links
  surfaced incidentally in one search result and were explicitly **not opened, read, or
  used** — recorded in `SOURCES.md`'s exclusion note. VERIFIED.
- No leaked labels were used — all figures cited from this project are the already-
  legitimate, already-audited Phase-1 statistics in `docs/DATASET_AUDIT.md`. VERIFIED.
- No model was downloaded. VERIFIED (license/model-card pages were read; no weights were
  fetched).
- No model was trained. VERIFIED.
- No blocker was implemented. VERIFIED (research and planning only; see
  `RESEARCH_TO_EXPERIMENT_PLAN.md` for what Phase 2 will implement).
- No pair-generation experiment was run. VERIFIED.
- No test prediction was made. VERIFIED.
- No leaderboard attempt was made. VERIFIED.
- No validation split (`validation_v1`) was mutated. VERIFIED (read-only reference to its
  metadata throughout).
- No commit or push was made during this research session (any commit remains the human's
  explicit action per `PROJECT_RULES.md` item 14).

## 28. Next Phase

**PHASE 2 — BLOCKING / CANDIDATE-GENERATION BENCHMARK.** Do NOT begin Phase 2 in this
session. STOP.

## 29. Teammate Research Reconciliation (2026-09-25, after Phase 1.5)

A teammate's independently produced Phase-1.5-style research package was reviewed, verified
against raw script/log output where possible, and reconciled against this dossier and its
companions. Full detail: `TEAMMATE_RESEARCH_RECONCILIATION.md`. Summary of what changed here
and in the companion documents: a reverse (doc→S1) retrieval route was promoted to P0
(§1.9 of `TECHNIQUE_EVIDENCE_MATRIX.md`, new EXP-B001R); the word-token joint name+address
TF-IDF implementation for the existing token-blocking route was given a directly-measured cost
justification (EXP-B001's implementation note); a train-derived Indic-script normalization
technique was added as a conditional, reproduce-first P1 ablation gated on the not-yet-built
Phase-3 `matcher_train` manifest (new EXP-B001I); numeric-conflict, entity-frequency, and
co-duplicate-support features were added to EXP-F001's Phase-3 feature backlog; and a
hand-authored France abbreviation dictionary the teammate flagged as a judgment call was
explicitly **not** adopted, since it conflicts with this dossier's own existing §23 policy
against country-specific hand-curated normalization dictionaries. No canonical Phase-1 fact
was overridden, no teammate exploratory score was promoted to a canonical benchmark, and no
implementation, model training, or candidate generation occurred. Phase 2 remains not started.
