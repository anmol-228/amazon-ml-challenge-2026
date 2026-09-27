# Teammate Research Reconciliation

Companion to `RESEARCH_DOSSIER.md`, `TECHNIQUE_EVIDENCE_MATRIX.md`,
`RESEARCH_TO_EXPERIMENT_PLAN.md`. This document reconciles a teammate's independent
Phase-1.5-style research package against canonical Phase 1 / Phase 1.5 project truth. It does
not implement anything; Phase 2 has not started.

## 1. Scope / Provenance

- **Source package:** `%USERPROFILE%\Downloads\research-20260925T101214Z-1-001.zip`
  (SHA-256 `c55c940805660a4599555c0511aec44d050a86f3764606f5c0c4eaac16f2ecc6`), extracted
  **read-only** into this session's scratchpad directory for inspection. The original ZIP in
  Downloads was never modified, renamed, or deleted. A sibling file,
  `Downloads\deep-research-report.md` (different timestamp, 2026-09-24), was **not** treated
  as part of this package — it does not match the named basename and was not inspected.
- **What was inspected:** all 3 handoff documents (`research/HANDOFF_A.md`, `HANDOFF_B.md`,
  `HANDOFF_C.md`), all 20 scripts in `research/audit_scripts/`, all 8 log files and both JSON
  dictionary artifacts in `research/artifacts/`. Every script that a headline claim in the
  handoffs cites was opened and read; most were cross-checked line-by-line against a preserved
  raw log or artifact (§4/§6 below record exactly which).
- **No teammate script was executed in this session.** Per the no-implementation gate, all
  verification was: reading source code for methodology soundness, reading preserved raw
  `print`/log output, and one independent arithmetic/artifact check (dictionary token counts,
  read directly from the JSON files with `python -c`).
- **Machine/environment note (incidental, not a compliance concern):** the scripts' hardcoded
  local paths are consistent with a real teammate's own machine, not a fabricated package.
  Noted only for authenticity, not acted on further. (Local machine-identifying details have
  been redacted from this document for publication.)
- **Reproducibility limitation (deliberate, per this reconciliation's own governing
  instructions):** the teammate's original package is **not** copied into this repository —
  the mission that produced this document explicitly directs "Do not copy the entire teammate
  archive," and separately requires the original package be left untouched in place. This
  document's SHA-256 hash identifies the exact archive inspected, and every verified claim in
  §4 cites the specific script/log filename and, where a concrete figure was checked, the
  exact field read from it — but a future reviewer without access to the teammate's own
  Downloads folder cannot re-run that verification from this repository alone. This is a known,
  accepted limitation of this reconciliation's scope, not an oversight.

## 2. Package Inventory

| File | Role |
|---|---|
| `HANDOFF_A.md` | Blocking / candidate generation research (backbone recommendation, measured recall, France strategy) |
| `HANDOFF_B.md` | Matcher / model research (feature catalogue, model shortlist, decision-rule theory, LOCO results) |
| `HANDOFF_C.md` | Competitive-optimization findings (distractor features, entity-frequency features, France label-free threshold, graph/stacking features, submission strategy) |
| `audit_scripts/block_exp4.py`, `deva_exp.py`, `block_exp2.py` | Blocking recall measurement (word/char n-gram, forward/reverse, Indic dictionary construction and leakage-safe held-out evaluation) |
| `audit_scripts/keyblock_exp.py` | Composite token-pair hash-key blocking (backup route, not adopted) |
| `audit_scripts/matcher_build.py`, `matcher_eval.py`, `matcher_eval2.py` | Candidate/feature construction and the real per-entity macro-F0.5 scorer (independently reimplemented, not `student_resource`'s `evaluation.py`) |
| `audit_scripts/c_exp.py`, `c_exp_F.py`, `c_exp_F_ablation.py`, `dist_exp.py`, `freq_exp.py` | Handoff-C experiments: learning curve, seed ensemble, FP anatomy, LOCO label-free threshold, self-training, distractor features, entity-frequency features, two-stage graph/stacking features |
| `audit_scripts/distractor_profile.py`, `france_diag.py`, `miss_analysis.py`, `audit1.py`, `audit2.py` | Diagnostics: distractor profile, label-free France test-input diagnostics, blocking-miss characterization, raw dataset audit (independently reproduces canonical Phase-1 numbers) |
| `audit_scripts/fix_labels.py` | A self-caught and fixed bug (see §6) |
| `artifacts/indic_name_dict.json`, `indic_addr_dict.json` | Learned Devanagari/Indic→Latin token dictionaries (1,312 and 17 tokens respectively — **directly verified by reading the files**, matching the handoff's claimed sizes exactly) |
| `artifacts/*.log` | Preserved raw stdout from `exp4`, `exp5` (India/US), `dist_exp`, `freq_exp`, `c_exp_F`, `c_exp_F_ablation` |

No original files were modified during this inspection.

## 3. Methodology Summary

The teammate ran a self-contained, single-machine (i7-13620H, 16GB, Windows) empirical
research pass structured the same way as canonical Phase 1.5: measure first, then recommend.
Distinctive methodological choices, all consistent with sound practice:

- **Evidence tagging** (`[AUDIT]`/`[SRC]`/`[EST]`) mirrors this project's own
  VERIFIED/INFERRED/UNKNOWN discipline.
- **Leakage-safe Indic dictionary construction:** `deva_exp.py` splits truth pairs by **S1
  entity** into an 80/20 train/held-out set *before* building the dictionary, measures
  coverage/purity only on the held-out 20%, and separately measures TEST-token coverage using
  **no labels** (vocabulary overlap only). This is the correct leakage discipline for a
  train-derived transformation.
- **LOCO (leave-one-country-out)** used throughout as the France-transfer proxy — the same
  technique `docs/VALIDATION_DESIGN.md` already names and canonical EXP-M004b schedules but
  had not yet run.
- **A self-caught bug:** `matcher_build.py` line 139's comment ("NOTE: use the qi/di COLUMNS
  here (rows were re-sorted above; the qi/di arrays are stale)") documents a label-alignment
  bug found and fixed in place; `fix_labels.py` is a redundant re-derivation-from-parquet
  safety check for any artifact built before the fix. This is evidence of rigor, not a
  concealed defect — flagged transparently here per the instruction to preserve provenance and
  not cherry-pick favorable outputs.
- **Metric reimplementation, not reuse:** the teammate's `score_sets`/`fbeta` in
  `matcher_eval.py` independently reimplements the official macro F0.5 (T-empty/P-empty→1.0
  etc.), rather than importing `student_resource/code/business_entity_resolution/src/evaluation.py`.
  The formula matches DEC-010 by inspection, but it is **not the same code path** the frozen
  evaluator uses — any number pulled from this package should be understood as scored by a
  parallel, not identical, implementation.
- **Scale:** all measurement is on **samples of TRAIN** (5,000–12,000 S1 entities per
  experiment, full-size un-subsampled document index), not the full `development`/`validation`
  partitions, and not `student_resource`'s own train/test split machinery. This is real
  empirical evidence at meaningful (though not full) scale, not a toy benchmark, but it is a
  **different sample and a different evaluation harness** than `validation_v1`.

## 4. Strong Findings (directly verified from raw script/log output in this session)

| # | Finding | Verification |
|---|---|---|
| 1 | **Reverse (doc→S1) retrieval beats forward (S1→doc) retrieval and is cheap.** Reverse@1 92.4–96.1%, reverse@2 94.0–97.1%, vs forward-per-source@10 86.9–93.8%, across both the earlier (`exp4`, Devanagari-only dict) and later (`exp5`, all-Indic dict) runs. | `exp4_India.log`, `exp5_India.log`, `exp5_US.log` raw JSON output, read directly — numbers match the handoff table exactly. |
| 2 | **Word-token joint TF-IDF (name+address) is far cheaper than character n-grams at equivalent recall.** `c4` (char 4-gram) costs 60.32 ms/query vs `word`'s 2.24 ms/query in the same run — a 26.9× ratio, matching the "27×" claim precisely — for +0.9pt fwd@10 (86.1 vs 85.2%; superseded once the all-Indic dictionary is added to the word route). | `exp4_India.log` line 6 vs line 4 (`ms_per_query` field), direct arithmetic. |
| 3 | **The all-Indic dictionary (1,312 name tokens, 17 address tokens) measurably improves India retrieval and is leakage-safe by construction.** fwd@10 rose from 85.24% (Devanagari-only, dict size 161) to 91.28% (all-Indic, dict size 1,328/1,312) — a +6.04pt gain, matching the claimed "+6.1". | `exp4_India.log` (`fwd.10: 0.8524`, dict size 161) vs `exp5_India.log` (`fwd.10: 0.9128`, dict size 1328); dictionary sizes independently confirmed by reading `indic_name_dict.json` (1,312 keys) and `indic_addr_dict.json` (17 keys) directly. |
| 4 | **Country is a 100% consistent partition key on train** (independently re-derived: `audit1.txt` reports `country agreement S1 vs match: 1.0` over 7,638,365 pairs) — this exactly reproduces `docs/DATASET_AUDIT.md` §9's own VERIFIED figure from an independent script, not a copy. |
| 5 | **Every matched S2/S3 id belongs to exactly one S1** (`audit1.txt`: "matched ids appearing under >1 S1 (should be 0): 0") — independently reproduces `docs/DATASET_AUDIT.md` §8's zero-target-reuse finding, and is the structural basis the reverse-retrieval finding (#1) and the exclusivity/context matcher features exploit. |
| 6 | **Distractor-aware numeric-conflict features give a measured, reproducible matcher gain.** `dist_exp.log`: base macro F0.5 0.94925 (t=0.7) → +dist 0.95789 (t=0.75) → +freq+dist 0.95851 (t=0.65); gain is concentrated in `num_first_logdiff` (6.3–6.7% of total gain share). | `dist_exp.log`, full raw JSON, matches HANDOFF_C §2.2 table exactly. |
| 7 | **Entity-frequency (address/name-sharing count) features give a small in-distribution gain and a larger unseen-country gain at a label-free threshold.** Random split 0.9493→0.9527 (+0.34); LOCO India→US at label-free threshold 0.9095→0.9133 (+0.38). | `freq_exp.log`, matches HANDOFF_C §6.1 table exactly. |
| 8 | **Two-stage stacking + co-duplicate "graph" support features add +0.25pt**, and an ablation shows the two components (stacking alone, graph features alone) each contribute roughly half, non-redundantly. | `c_exp_F.log` and `c_exp_F_ablation.log`, both matching HANDOFF_C §3 exactly, including the paired-comparison design. |
| 9 | **The raw dataset audit in `audit1.txt` independently reproduces essentially every canonical Phase-1 VERIFIED figure** (row counts, singleton rate 5.58%, multiplicity distribution, country splits, name/address exact-match rates, numeric-token digit-sharing rate ~80–92%) from a completely independent script, with no shared code. This is strong corroborating evidence for the canonical Phase-1 audit, not new information. |

**Excluded from this table on review (per the independent technical audit of this document,
§14 below):** self-training on the unseen country was initially listed here alongside the
directly-verified findings above. It does not belong in a "directly verified" table — the
specific −1.0/−1.7pt magnitude has no preserved standalone log in this package (Category C:
teammate-reported, mechanism-plausible, not independently re-derived from raw output in this
session), unlike Findings #1–9 above, each of which was checked against a preserved raw
log or artifact. It is correctly classified and discussed only in §5 below.

## 5. Useful Negative Findings

- **Self-training on the unseen country: kill (Category C — teammate-reported, not
  independently re-derived from a preserved raw log in this session).** HANDOFF_C and
  `c_exp.py`'s `E_selftrain` block report a −1.0 to −1.7pt macro F0.5 drop even at
  98%-precision pseudo-labels; the mechanism (`c_exp.py` lines 74–83: pseudo-positives are
  by construction the *easy*, already-confident pairs, so the retrain learns nothing about the
  hard band while shifting the class balance) is sound and the reported failure direction is
  consistent with well-known self-training pathologies on distribution-shifted pseudo-labels,
  so this is credible — but treat the exact magnitude as reported, not verified, and do not
  cite it as a directly-verified number. Directly relevant to any future France
  strategy — this rules out one otherwise-tempting P1 direction (§18 of the mission's own
  framing) before canonical Phase 3/4 would have had to spend a submission on it.
- **Character n-gram retrieval at full scale on a single machine is not cost-competitive with
  word-token retrieval** for the *primary* blocking pass (27× slower for comparable/marginal
  recall gain, per Finding #2). This does **not** contradict Sparkly's own P0 evidence
  (`TECHNIQUE_EVIDENCE_MATRIX.md` §1.2) — Sparkly benchmarks a distributed, purpose-built
  implementation at cluster scale, not a single-machine `TfidfVectorizer`/`HashingVectorizer`
  pass. It **does** directly substantiate the canonical round-3 framing that already asks "what
  *incremental* coverage does character retrieval add after cheaper routes?" rather than
  treating character n-grams as an assumed P0 backbone. Read together, both are consistent:
  char n-grams remain worth testing as an *incremental/residual* route (canonical EXP-B002,
  already revised this way) and as a **matcher-stage feature** (`n_c3`/`a_c3`, already present
  in the teammate's own feature catalogue and consistent with canonical §2.1's P0 char-level
  similarity features) — just not as the single-machine primary retrieval backbone.
- **Sorted neighborhood, phonetic keys, postal-code blocking, MinHash-LSH, learned/DL blocking,
  libpostal:** all independently arrive at the same REJECT/NOT-WORTH-IT conclusions
  `TECHNIQUE_EVIDENCE_MATRIX.md` §1.5/§1.6/§1.8/§12 already reached, for closely related
  reasons (predictable-output-size risk, English-only mechanisms, low coverage, external-data
  risk). This is convergent, not new, evidence — it raises confidence in the existing REJECTs
  without changing them.

## 6. Methodology Limitations

- **Sampling:** every blocking/matcher number is measured on a 5,000–12,000-S1 sample per
  country, against the **full, un-subsampled document index** — a real, meaningful scale, but
  materially smaller than `validation_v1`'s 441,365-entity `validation` partition or even its
  1,765,456-entity `development` partition. None of these numbers should be read as a
  prediction of what a full-`development` run would show; they are directional.
- **Split/evaluation-harness difference:** the teammate's `matcher_eval.py` reimplements the
  F0.5 formula independently rather than calling `evaluation.py`, and uses a random 50/50 or
  hash-based split of a fresh sample, not `validation_v1`'s frozen, stratified,
  seed-42 entity split. Any specific score (0.948, 0.982 ceiling, 0.900/0.895 LOCO, etc.) is
  **not directly comparable** to a canonical `dev_eval`/`validation` number and must not be
  recorded anywhere as a canonical project benchmark (see §8).
- **Approximated reverse pass in the matcher experiments:** `matcher_build.py` computes the
  reverse blocker only for docs already in each query's forward top-60 (`block_exp4.py`'s
  reverse pass is the only *full*, unrestricted reverse measurement). The teammate's own
  handoffs flag this explicitly ("Compute the reverse pass over all docs in production;
  here it was approximated") — the matcher-side `rev_rank`/`is_doc_best` feature gain-share
  numbers (52.9%/11.4%) are therefore a **lower bound** on the true value of full reverse
  retrieval, not an overstatement.
- **In-sample tuning risk (bounded):** thresholds are chosen "on the validation part of TRAIN"
  within each experiment (`matcher_eval.py`'s `evaluate()` chooses `t_best` on a `va` split,
  scores on a disjoint `test` split) — this is a reasonable honest-eval design at the scale of
  a single experiment, but across the *whole* sequence of experiments (base → +dist → +freq →
  +freq+dist → graph/stacking), each new variant is compared against the *same* held-out
  `test` half repeatedly. This is the same repeated-comparison risk canonical Phase 1.5's own
  audit flagged for `validation_v1` (frozen-validation reuse policy) — here it applies to the
  teammate's `test` half, not to `validation_v1` itself, so it does not touch this project's
  frozen benchmark, but it does mean the exact magnitude of each incremental feature's gain
  (+0.86pt distractor, +0.34pt frequency, +0.25pt graph) should be read as directionally
  reliable, not as a precise, non-overlapping decomposition.
- **The 0.948-baseline / 52.9%-gain-share headline number in HANDOFF_B has no single preserved
  raw log in this package** (unlike the §4 findings). Four independent re-derivations of a
  closely related "base, 45 features" run (`dist_exp.log` 0.94925, `freq_exp.log` 0.94925,
  `c_exp_F.log`/`c_exp_F_ablation.log` 0.94925 at the same threshold grid) corroborate a
  baseline in the same 0.948–0.949 range, so the number is credible, but it is Category B
  (reproducible from script + closely related logs), not Category A (a single directly-verified
  raw output) for the *exact* 0.948/52.9% figures as stated.
- **The France diagnostics (`france_diag.py`) are label-free by construction** (France has no
  training truth anywhere), so "France (test)" rows in HANDOFF_C §6 are genuinely structural/
  retrieval-confidence statistics, not accuracy measurements — there is no way, in this package
  or in canonical Phase 1, to directly measure France match accuracy pre-submission.
- **Possible leakage vector not fully closed:** the Indic dictionary's held-out evaluation
  (`deva_exp.py`) splits by S1 entity for its *quality* metrics (purity, held-out coverage),
  which is correct — but the **production artifact actually shipped in `artifacts/`** was
  "built on 80% of train entities" per the teammate's own note, not on a
  Phase-3-style `matcher_train`-only subset. Before any future canonical experiment reuses
  this exact dictionary, it must be rebuilt strictly inside whatever `matcher_train` manifest
  Phase 3 eventually creates (`RESEARCH_TO_EXPERIMENT_PLAN.md`'s supervised-matcher development
  split), not reused as-is — this is already the teammate's own stated caveat (HANDOFF_A §9),
  restated here as a binding condition for any reuse.

## 7. Compliance Review

| Item | Status | Reasoning |
|---|---|---|
| Reverse retrieval, word TF-IDF, Indic dictionary (train-derived), numeric-conflict/frequency features, graph/stacking features | **CLEAR** | Pure statistical/lexical techniques over supplied data only; MIT/Apache/BSD libraries (`sparse_dot_topn`, `polars`, `scikit-learn`, `rapidfuzz`, `lightgbm`) already CLEAR per `LICENSE_AND_TOOLING_MATRIX.md`; no external entity lookup, no external business/geo data. |
| IDF/df-pruning **fitted on the France partition of test inputs** (transductive vectorization) | **REVIEW REQUIRED (methodology, not fair-play)** | Uses only the competition's own supplied unlabeled test records — no external data, no labels. Not a fair-play violation under `docs/OFFICIAL_REQUIREMENTS.md`'s external-lookup ban. It **is** a genuine open methodology question (fitting any statistic on test inputs, even unsupervised) that canonical project rules have not yet explicitly addressed. Classification per mission taxonomy: **(A) clearly allowed** on a fair-play reading, **(B) unclear** on general ML-methodology grounds — must be explicitly documented in the eventual `Documentation_template.md` as a judgment call (the teammate's own handoffs already flag this) rather than silently adopted. |
| **Label-free ("prior-matching") threshold for France** | **REVIEW REQUIRED (methodology, same class as above)** | Same reasoning — uses only the unlabeled test partition's own prediction-rate statistics, no external data, no labels. Directionally reasonable and self-evidenced as an improvement (§1 of HANDOFF_C), but it is a transductive technique with no precedent yet in this project's decision-logic policy (`TECHNIQUE_EVIDENCE_MATRIX.md` §7/§8 currently assume a single tuned threshold, not a partition-adaptive one). Not blocking for Phase 2 (it is a Phase-3/4 decision-rule question). |
| **Self-training on France (pseudo-labels from mutual-best pairs)** | **REVIEW REQUIRED, but empirically REJECTED anyway** | Same transductive class as above, and additionally measured to *hurt* (§5). No action needed — already a negative finding, not a candidate for adoption. |
| **Hand-written French legal-suffix/street-type abbreviation list** (`sarl, sas, sasu, sa, eurl, sci, snc`; `r→rue, av→avenue`, etc., citing AFNOR NF Z10-011) | **REJECT as currently scoped — direct conflict with an existing canonical decision** | `RESEARCH_DOSSIER.md` §23 ("What NOT To Do") and `DEC-014`/`TECHNIQUE_EVIDENCE_MATRIX.md` §11 explicitly instruct: "Do not build a France-specific or India-specific abbreviation dictionary — prioritize script/vocabulary-agnostic techniques instead," calling exactly this pattern (a hand-curated country-specific normalization list) an anti-pattern the research brief warns against. The teammate herself flagged this as a judgment call requiring team confirmation ("I believe it's allowed; please confirm and document it") — this reconciliation does **not** confirm it. It is external, hand-authored domain knowledge about French addressing conventions, sourced from a named national standard (AFNOR), not learned from any supplied data — qualitatively different from the train-derived Indic dictionary (§6 above), which is learned entirely from `train_ground_truth.tsv` pairs with no outside vocabulary. Reversing the existing canonical REJECT would require an explicit new decision by the human team, not a silent import via this reconciliation. |
| Indic-script train-derived dictionary, reused **as shipped in this package** (built on 80% of entities, not `matcher_train`) | **REVIEW REQUIRED (reproduction condition, not fair-play)** | See §6's leakage caveat — CLEAR on fair-play grounds (learned only from supplied training pairs, held-out entity-level evaluation), but must be rebuilt inside the eventual `matcher_train` manifest before any canonical experiment scores it, per the project's own supervised-matcher development-split policy. |

**No item in this package shows evidence of:** external business/entity lookup, commercial ER
APIs, government registry lookups, geocoding, scraped business data, competitor-repository
reuse, or leaked labels. `libpostal` was explicitly identified and rejected by the teammate
herself for exactly this reason (OSM-trained model data).

## 8. Comparison With Canonical Phase 1.5

**Agreement (independently re-derived, not merely cited):**
- Zero cross-S1 target reuse / country 100% consistency on true links (`audit1.txt`
  reproduces `docs/DATASET_AUDIT.md` §8/§9 exactly, from independent code).
- Numeric-address-token agreement as a strong signal: `audit1.txt` measures
  "share ≥1 digit token: 0.7988 / share|both have: 0.916" on true links — same phenomenon as
  `docs/DATASET_AUDIT.md` §12's 63.49% *exact-set* figure (a related but not identical
  statistic — "share ≥1 token" vs "exact-set-equal" — both point the same direction).
- Name-only blocking has a real coverage ceiling (teammate: 85.6% name-token share; canonical
  §11: 10% zero-overlap tail) — same phenomenon, compatible numbers.
- Character-level retrieval's cost/complexity concern is corroborated (§5 above), reinforcing
  rather than contradicting the canonical round-3 framing.
- Graph/collective-ER-style exclusivity (`is_doc_best`, one-doc-one-S1) independently converges
  with canonical `EXP-M008`'s already-planned capacity-one conflict-resolution ablation — the
  teammate's matcher-feature framing (`is_doc_best`, `gap_doc`, `rev_rank` as *soft* GBDT
  features, checked with a *global* exclusivity post-pass as "a safety net", not a hard rule)
  matches this project's own §15 policy almost exactly: *prefer soft features first, treat a
  hard exclusivity rule as a separate, later ablation* — this is independent convergence on the
  same design principle, not something borrowed from canonical docs.

**New contributions (not previously in canonical Phase 1.5):**
- Reverse (doc→S1) retrieval as a first-class blocking route, empirically shown to outperform
  forward retrieval alone and to combine with it for large recall gains (§4 Finding #1). Not
  present anywhere in `RESEARCH_TO_EXPERIMENT_PLAN.md`'s EXP-B001–B004 today.
- Word-token joint name+address TF-IDF as a specifically-costed, directly-comparable
  alternative to character n-grams (§4 Finding #2, §5).
- The Indic-script transliteration dictionary as a concrete, leakage-conscious, data-derived
  normalization technique with a measured effect size (§4 Finding #3) — canonical Phase 1.5
  identified France/multilingual transfer as the single largest open risk (§17/§14 of the
  dossier) but had no concrete India-side script-normalization proposal.
- Numeric-conflict and entity-frequency matcher features with measured gains (§4 Findings
  #6/#7) that refine, with specifics, the canonical dossier's already-directionally-correct
  guidance (§7 of the dossier: "a name-similar candidate pair with conflicting numeric tokens
  is a specific, interpretable, high-value feature").
- Two-stage stacking / co-duplicate support features (§4 Finding #8) — not previously
  considered in any canonical document.
- A measured, negative finding on self-training for France (§5) — closes off a plausible P1
  direction before it would have cost a submission.

**Contradictions:** none found at the level of established canonical facts. The one apparent
tension (character n-grams "not worth it" vs. canonical P0 status) is resolved in §5 as
compatible once both are read at the precision they actually claim (primary-route cost vs.
incremental/residual value and feature value) — not a genuine contradiction.

## 9. Promoted Findings

| Finding | Class | Why |
|---|---|---|
| Reverse (doc→S1) retrieval | **PROMOTE TO P0** | Directly verified (§4 #1), cheap, exploits an already-canonical structural fact (§8's zero target-reuse / partition property), materially raises recall over forward-only. |
| Word-token joint name+address TF-IDF as the primary lexical retrieval implementation detail | **PROMOTE TO P0** (refines EXP-B001's implementation, does not replace it) | Directly verified cost/recall tradeoff (§4 #2); consistent with, not contradicting, Sparkly's P0 status for char n-grams as a complementary/residual route. |
| Train-derived Indic-script dictionary | **PROMOTE TO P1, REPRODUCE FIRST** | Verified effect (§4 #3), verified leakage-safe *methodology*, but the specific shipped artifact must be rebuilt on a `matcher_train`-only manifest (§6) before any canonical number is trusted. |
| Numeric-conflict features (`num_first_logdiff`, `core_name_equal`, `core_token_diff`, sibling/co-duplicate signals) | **PROMOTE TO P0** (Phase-3 feature backlog) | Directly verified gain (§4 #6), directly extends canonical §7/§21's already-P0 numeric-token-conflict guidance with concrete, measured feature definitions. |
| Entity-frequency features (address/name-sharing counts, label-free) | **PROMOTE TO P1** (Phase-3 feature backlog) | Directly verified (§4 #7), label-free and country-agnostic by construction, particularly relevant to France's measured "crowded candidate list" structural profile. |
| Two-stage stacking + co-duplicate support ("graph-lite") features | **PROMOTE TO P1** | Directly verified (§4 #8), cheap relative to gain, does not require any graph algorithm (consistent with the existing REJECT on transitive-closure/correlation-clustering ER). |
| Retrieval-context features (`rev_rank`, `is_doc_best`, `gap_doc`, `doc_margin`) | **KEEP AS P2/BACKLOG, already substantially covered** | `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-F001 already lists exclusivity/list-context features; this reconciliation confirms they matter (with the caveat in §6 that the teammate's own gain-share number is a lower bound from an approximated reverse pass) rather than introducing something new. |
| Self-training on unseen country | **REJECT (evidence-based, Category C magnitude)** | Reported to hurt (§5); mechanism sound, magnitude not independently re-derived from a preserved log in this session. |
| IDF/threshold fitted on unlabeled test inputs (France) | **COMPLIANCE REVIEW REQUIRED** | See §7 — not a fair-play violation, but an open methodology question needing explicit documentation before adoption. Not blocking Phase 2. |
| Hand-written French abbreviation/street-type dictionary | **REJECT as scoped** | Direct conflict with existing `DEC-014`/dossier §23 policy (§7 above); would require an explicit new human decision to reverse, not a silent import. |
| Composite token-pair hash-key blocking (`keyblock_exp.py`) | **DEFER / KEEP AS BACKUP** | Implemented, not yet run to a documented conclusion by the teammate herself ("script ready, not yet run" per HANDOFF_A §4); no verified evidence either way in this package. |
| Distractor near-copy profile (`distractor_profile.py`) | **KEEP AS P1 evidence**, Category C | Cited numbers were not independently re-derived from a preserved raw log in this session (script was read; its specific output table was not); directionally consistent with the numeric-conflict feature results that *were* directly verified (§4 #6), so treated as credible but unverified-in-this-session. |

## 10. Rejected / Quarantined Findings and Why

- **Self-training on France:** rejected on the teammate's own measured evidence (−1.0 to
  −1.7pt); no reason to revisit absent new evidence.
- **Hand-written French abbreviation/street-type list:** quarantined — conflicts with an
  existing explicit canonical decision (`DEC-014`, dossier §23). Not imported. If the human
  team wants to revisit the underlying canonical REJECT itself (i.e., decide that a
  standards-sourced, non-supplied-data abbreviation list is acceptable domain knowledge rather
  than an anti-pattern), that is a new decision for the team to make explicitly — this
  reconciliation flags the conflict but does not resolve it either way.
- **Character n-gram retrieval as a full-population primary blocking route:** not rejected
  outright — demoted to exactly the role canonical round-3 already assigns it (incremental/
  residual-population test, EXP-B002) and to a matcher-stage feature (`n_c3`/`a_c3`, already
  in both the teammate's and canonical EXP-F001's feature lists).
- **Composite token-pair hash-key blocking:** not rejected, simply unevaluated by the teammate
  herself — kept as backup per her own framing, not promoted further here.
- **Postal-code blocking, phonetic keys, MinHash-LSH, sorted neighborhood, libpostal:**
  quarantined, consistent with existing canonical REJECTs (§5) — no new evidence changes
  anything here.

## 11. Phase-2 Changes

Applied directly to `RESEARCH_TO_EXPERIMENT_PLAN.md` (see that file's own revision history for
the exact diff; summarized here):

1. **New EXP-B001R — Reverse (doc→S1) retrieval**, scheduled immediately alongside EXP-B001
   (parallel, not sequential — both are cheap, independent routes over the same TF-IDF
   matrices), promoted to P0 given the teammate's directly-verified evidence and the existing
   canonical partition-structure fact (`docs/DATASET_AUDIT.md` §8) it exploits.
2. **EXP-B001 implementation note added:** the word-token joint name+address TF-IDF
   (cosine/dot-product scoring via a sparse top-k matmul, not raw inverted-index token overlap)
   is now named as the concrete P0 implementation approach for the token-blocking route,
   citing the teammate's directly-measured cost/recall tradeoff. This does not change
   EXP-B001's question or metrics, only its implementation guidance.
3. **New EXP-B001I — Indic-script train-derived normalization ablation**, added as a
   **REPRODUCE FIRST** conditional P1 experiment, explicitly gated on rebuilding the dictionary
   inside a `matcher_train`-only manifest (not the teammate's as-shipped 80%-of-train-entities
   artifact) and re-evaluating on an entity-disjoint held-out set, per §6/§9 above.
4. **EXP-B002's existing round-3 framing (incremental value after cheaper routes) is
   reinforced, not changed** — the teammate's cost measurement is added as corroborating
   evidence in `TECHNIQUE_EVIDENCE_MATRIX.md` §1.2, not a reason for further revision.
5. **EXP-B004 (multi-route union) scope note added:** the union now explicitly includes the
   reverse route (EXP-B001R) as a fourth input alongside token/char-n-gram/numeric-address
   routes, since it is cheap and independently evidenced.

No experiment ID was renumbered; all additions use the existing naming convention with a
letter suffix, per the mission's own instruction to avoid unnecessary renumbering.

## 12. Phase-3 Backlog Changes

Applied to `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-F001's feature inventory and
`TECHNIQUE_EVIDENCE_MATRIX.md` §2/§7 (summarized; see those files for the exact wording):

- **Promoted to the explicit P0 feature inventory:** numeric-conflict features
  (`num_conflict`, `num_first_logdiff`/absolute-difference, `num_all_equal`) as a named,
  concrete refinement of the already-P0 "explicit conflict flag" requirement.
- **Promoted to P1 feature backlog:** core-name-after-suffix-strip equality/token-diff
  (`core_name_equal`, `core_token_diff`) — flagged with the same over-normalization caution
  `TECHNIQUE_EVIDENCE_MATRIX.md` §2.3 already states (must be paired with numeric/address
  disambiguation, never used alone).
- **Promoted to P1 feature backlog:** entity-frequency features (address/name-sharing counts,
  label-free, computed per country partition).
- **Promoted to P1 feature backlog:** co-duplicate/sibling "graph-lite" support features
  (`sup_max`/`sup_sum`/`sup_nhi`-style signals) as a two-stage-stacking addition, explicitly
  not a graph algorithm and not a reversal of the existing collective-ER REJECT.
- **Confirmed, not changed:** retrieval-context/exclusivity features (`rev_rank`, `is_doc_best`,
  `gap_doc`, `doc_margin`) were already P0 in EXP-F001; this reconciliation adds no new
  feature here, only corroborating evidence that they matter in practice.
- **Not promoted:** the teammate's specific LightGBM hyperparameters, exact threshold values
  (0.65/0.7/0.75/0.8), and exact model architecture choices — these are sample-specific tuning
  artifacts from a non-canonical split (§6/§8) and must be re-derived on `matcher_train`/
  `dev_eval`/`calibration_holdout` when Phase 3 actually builds that manifest.

## 13. Open Questions

- Whether the reverse-retrieval + word-TF-IDF + Indic-dictionary + numeric/frequency-feature
  combination, measured together on canonical `development` at full scale, reaches recall and
  candidate-load numbers consistent with the teammate's sample-scale results — this is exactly
  what canonical EXP-B001/EXP-B001R/EXP-B001I/EXP-B004 now exist to test; not resolved here.
- Whether the France-partition transductive IDF/threshold fitting (§7) is intended to be
  permitted — genuinely open, requires either an organizer clarification or an explicit human
  project decision; not resolved by this reconciliation.
- Whether the human team wants to revisit the existing REJECT on country-specific
  hand-authored normalization dictionaries (the France abbreviation list) — flagged, not
  decided, in §7/§10.
- Whether the composite token-pair hash-key blocking route (`keyblock_exp.py`, implemented but
  unevaluated) is worth running as a real experiment — no evidence either way exists yet.
- The exact magnitude of full-scale (not sampled) gains from each promoted feature — inherently
  unknowable until Phase 3 builds the real feature table on `matcher_train`.

## 14. Independent Technical Audit

An independent, read-only technical audit was run against this document and the updated
`RESEARCH_TO_EXPERIMENT_PLAN.md` before Phase 2, consistent with this project's practice of
independently reviewing research-phase deliverables that materially change experiment
priorities. Full summary: `docs/research/audit/RECONCILIATION_AUDIT_SUMMARY.md`.

**Findings and disposition (see that summary for the full issue/impact/repair table):**
- The audit's own record was first saved in a form that conflicted with this project's
  artifact-hygiene policy — fixed by replacing it with the concise, tool-neutral summary now
  linked above.
- DEC-023's literal "TEAMMATE EXPLORATORY EVIDENCE" label was not applied everywhere a
  teammate score is cited in `RESEARCH_TO_EXPERIMENT_PLAN.md` and
  `TECHNIQUE_EVIDENCE_MATRIX.md` — fixed by adding the label at each teammate-sourced figure
  in both files' new/edited sections (EXP-B001R, EXP-B001I, EXP-B001's implementation note,
  TECHNIQUE_EVIDENCE_MATRIX.md §1.1/§1.2/§1.9).
- Self-training was listed in §4's "directly verified" table despite its own note admitting
  Category C (not independently re-derived from a preserved log) — fixed by removing it from
  §4's table and discussing it only in §5 with its category stated up front (see §4/§5 above —
  this edit already reflects the fix).
- The teammate package itself is not reproducible from this repository (only its SHA-256 and
  this document's citations are preserved) — acknowledged as a deliberate, accepted limitation
  in §1 above, following this reconciliation's own governing instruction not to copy the
  teammate's archive into the repository.

**Positive finding:** the audit confirmed EXP-B001R/EXP-B001I and the feature-backlog
additions are consistent across `TEAMMATE_RESEARCH_RECONCILIATION.md`, `DECISIONS.md`,
`PROJECT_STATE.md`, `experiments/EXPERIMENT_LOG.md`, `RESEARCH_TO_EXPERIMENT_PLAN.md`,
`TECHNIQUE_EVIDENCE_MATRIX.md`, and `RESEARCH_DOSSIER.md`; the France abbreviation dictionary
remains quarantined; `validation_v1`'s manifest hash is untouched; Phase 2 remains explicitly
not started.
