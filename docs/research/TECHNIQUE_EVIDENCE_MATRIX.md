# Technique Evidence Matrix — Phase 1.5

Every row ties a technique to: the Phase-1 evidence trigger, research evidence strength,
transferability, and a priority. Priorities: **P0** (test early — high
information/competitive value), **P1** (strong follow-up), **P2** (only if time/results
justify it), **REJECT** (poor fit). CPU/GPU/RAM/runtime figures are labeled MEASURED
(this machine), DOCUMENTED (publisher/paper claim), or ESTIMATED (our reasoning) — never
asserted as fact without a label.

**Note on "complete-coverage"/"candidate-recall" mentions below (added in the round-2
methodology repair):** these per-technique rows use those terms as informal qualitative
labels (HIGH/MEDIUM/LOW expected effect), not as the formal metric. The precise, binding
definitions — including the round-2 correction separating non-singleton candidate
recall/complete true-link coverage from singleton candidate exposure rate/load — live in
`RESEARCH_TO_EXPERIMENT_PLAN.md`'s "Formal blocking-metric definitions" section; that
document, not this one, is authoritative for exact metric semantics.

Machine reference (MEASURED, Phase 1): AMD Ryzen 5 7500F (6C/12T), 31.6GB RAM, RTX 5060 Ti
(VRAM unverified by a reliable local source — do not assume any figure until `nvidia-smi`
is actually run), ~132GB free disk at Phase-1 measurement, Windows.

---

## 1. Candidate generation / blocking

### 1.1 Token/word blocking (name + address tokens, union of routes)
- **Problem solved:** first-pass recall at tractable candidate volume.
- **Phase-1 trigger:** §2 (2.3×10¹³ Cartesian pairs, infeasible); §11 (10% zero-name-token-
  overlap tail).
- **Research basis:** standard baseline across every blocking survey (Papadakis et al.
  1905.06167; Christophides et al. 1905.06397).
- **Evidence strength:** STRONG (textbook-level, reproduced across dozens of benchmarks).
- **Transferability:** HIGH — directly matches our record shape (short structured text
  fields).
- **Expected benefit:** high recall on the ~90% of true links with nonzero name-token
  overlap (§11); cheap, embarrassingly parallel. **Corrected (round 3 — EXP-B001 semantic
  repair):** because this route unions name-token *and* address-token retrieval (not
  name-token retrieval alone), the §11 ~90%/~10% split is a ceiling only for the name-token
  half of this route, not for the route as a whole — some fraction of the zero-name-
  token-overlap tail may still be recovered via address-token overlap. That fraction is an
  empirical question `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-B001 now measures directly
  (diagnostic (D), "address-rescued zero-name-overlap links"), not something assumed here
  either way.
- **Precision effect:** neutral (blocking, not a decision rule) but candidate load grows on
  generic-name collision groups (§14, up to ~460-way collisions).
- **Recall/candidate-recall/complete-coverage effect:** HIGH for the ~90% name-token-overlap
  population; for the ~10% zero-name-token-overlap tail, **not assumed to be zero** —
  address-token retrieval may partially rescue it (measured by EXP-B001 diagnostic (D));
  whatever residual remains after that measured rescue is what EXP-B002/B003 exist to
  cover, not the full raw ~10%.
- **France robustness:** MEDIUM — token blocking is script/vocabulary agnostic in
  mechanism, but France-specific tokenization quality (legal-form tokens like "SARL",
  accented characters) is UNKNOWN by construction (no France training data, §17).
- **CPU/GPU/RAM/disk/runtime:** ESTIMATED — inverted-index construction over ~10M records
  is a single streaming pass; RAM bounded by vocabulary size, well within the 31.6GB budget
  if implemented as true streaming (Phase 1 already demonstrated the memory risk of
  *not* streaming at this row count — EXPERIMENT_LOG.md, one computation dropped free RAM
  to ~3GB and was rewritten).
- **Engineering complexity:** LOW.
- **Compliance/license:** CLEAR (no external data).
- **Biggest failure mode:** the *name-token* route's ~10% zero-overlap tail (§11) is
  structurally invisible to name-token retrieval specifically; whether the address-token
  route rescues part of it is measured, not assumed (see the round-3 correction above) —
  whatever residual is *not* rescued by either sub-route remains structurally invisible to
  this technique.
- **Earliest cheap falsification test:** measure link recall of the nonzero-name-token-
  overlap population on `development` — if it materially undershoots ~90%, something about
  tokenization (normalization version, stopword handling) is wrong, not the technique
  itself. Separately, measure the address-rescue fraction on the zero-name-overlap
  population as its own diagnostic, not a pass/fail check.
- **Corroborating independent evidence (added after teammate-research reconciliation —
  `TEAMMATE_RESEARCH_RECONCILIATION.md` §4/§8):** **TEAMMATE EXPLORATORY EVIDENCE** — an
  independent, single-machine measurement found word-token joint name+address TF-IDF (sparse
  top-k cosine, IDF-weighted, df-pruned) achieves forward-per-source recall in the 87-95% range
  on a train sample (not `development`) at 2-8 ms/query,
  and that this same joint retrieval, run in reverse (doc-to-S1), achieves even higher recall
  (92-97%) more cheaply. This is corroborating evidence for this route's P0 status and directly
  informs its implementation (§11 of the experiment plan); it also motivates the new
  EXP-B001R reverse-retrieval experiment.
- **Priority: P0.**

### 1.2 Character n-gram / q-gram retrieval (TF-IDF or raw n-gram overlap)
- **Problem solved:** recall on the name-token-zero-overlap tail (§11) and typo/OCR-like
  noise; language-agnostic by construction.
- **Phase-1 trigger:** §11 (10% zero-token-overlap tail), §17 (France transfer risk —
  character-level methods degrade more gracefully across scripts/vocabularies than
  token-level ones).
- **Research basis:** **Sparkly** (Paulsen, Govind, Doan; PVLDB 16(6), 2023) is the single
  strongest piece of evidence here: top-k TF-IDF blocking (3-gram tokenization in their
  Manual variant) outperformed 8 state-of-the-art blockers, including DL-based ones, on 15
  EM benchmarks, and scaled to 10M tuples in <100 minutes and 26M tuples in ~130 minutes on
  a modest distributed cluster — a scale class directly comparable to our ~10M-row target
  tables. Also: Cohen/Ravikumar/Fienberg (CMU, 2003) found TF-IDF/SoftTFIDF-style hybrids
  competitive with or better than pure edit-distance methods for name matching.
- **Evidence strength:** STRONG (Sparkly is a peer-reviewed VLDB paper with published,
  reproducible benchmark results and public code/artifacts).
- **Transferability:** HIGH — same problem shape (blocking-then-matching over structured
  short-text tables), comparable scale, explicit claim that "tf/idf blocking needs more
  attention" as a *general* finding, not a dataset-specific fluke.
- **Expected benefit:** recovers name-token-zero-overlap true links (§11) that §1.1's
  name-token route cannot reach, and is far less sensitive to word-order/abbreviation noise
  than pure token blocking. **Corrected (round 3):** its actual target population is
  whatever remains after §1.1's own address-token sub-route has already rescued part of
  that tail (`RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-B001 diagnostic (D)/EXP-B002's corrected
  question) — not the full raw ~10% figure, some of which §1.1 may already cover.
- **Precision effect:** neutral at blocking stage.
- **Recall/candidate-recall effect:** HIGH — directly targets the residual population
  §1.1's combined name+address token routes miss entirely.
- **Complete-coverage effect:** contributes toward it as a complementary route (union, not
  intersection — §6 anti-anchoring rule).
- **France robustness:** MEDIUM-HIGH — character n-grams degrade gracefully across
  vocabularies/scripts (still UNKNOWN in the strict sense — no France ground truth exists,
  §17 — but the mechanism is inherently less English/India-vocabulary-specific than
  token-dictionary approaches).
- **CPU/GPU/RAM/disk/runtime:** DOCUMENTED (Sparkly: 10M tuples in <100 min on an AWS
  cluster, not a single machine) / ESTIMATED for our single-machine, non-Spark setting —
  a single-node TF-IDF top-k implementation (e.g., scikit-learn sparse `TfidfVectorizer` +
  a sparse nearest-neighbor / inverted-index query) at ~10M rows is plausible within a few
  hours on the measured 6-core/12-thread machine, but this specific runtime is NOT
  independently measured for our hardware and should be treated as ESTIMATED until timed.
- **Engineering complexity:** LOW-MEDIUM (single-node sparse TF-IDF + top-k retrieval is
  substantially simpler than Sparkly's distributed Lucene-on-Spark implementation; the
  simplification is a reasonable engineering choice, not a like-for-like reproduction).
- **Compliance/license:** CLEAR (pure lexical technique, no external data, no pretrained
  model dependency at all).
- **Biggest failure mode:** naive full sparse cosine similarity across ~2.2M×10M would
  itself be a scale problem; must use inverted-index/top-k retrieval (as Sparkly does), not
  a dense similarity matrix.
- **Earliest cheap falsification test:** measure candidate recall + candidate load of
  character-n-gram top-k retrieval alone on `development`, focused specifically on the §11
  zero-name-token-overlap subset — if it fails to recover most of that subset, the
  technique underperforms its literature precedent on our data specifically.
- **Corroborating independent evidence, with a scope caveat (added after teammate-research
  reconciliation — `TEAMMATE_RESEARCH_RECONCILIATION.md` §4/§5/§8):** **TEAMMATE EXPLORATORY
  EVIDENCE** — an independent, single-machine measurement (train sample, not `development`)
  found character 4-gram retrieval costs ~27× a word-token
  equivalent (60 ms vs 2.2 ms per query) for only marginal recall gain as a *primary,
  full-population* retrieval route, and that character 3-grams failed to complete at scale
  unpruned. This does **not** contradict Sparkly's own evidence — Sparkly benchmarks a
  purpose-built, distributed implementation at cluster scale, not a single-machine generic
  vectorizer — but it directly substantiates this project's own round-3 framing (below) that
  already treats character retrieval as an *incremental/residual-population* test rather than
  an assumed full-population P0 backbone, and additionally supports its role as a
  matcher-stage feature (char-3-gram cosine, already in EXP-F001's inventory) over its role as
  a primary retrieval mechanism.
- **Priority: P0.**

### 1.3 Numeric-address-token blocking (exact and partial numeric-set overlap)
- **Problem solved:** a second, largely orthogonal blocking key that does not depend on
  name similarity at all.
- **Phase-1 trigger:** §12 (63.49% of all true links have exactly matching numeric-token
  sets extracted from address — the single strongest cheap signal found in the entire
  audit); §21 explicitly flags this as the most promising cheap, language-agnostic signal.
- **Research basis:** general address-matching literature consistently treats numeric
  fields (house number, unit, postal-like codes) as handled separately from named fields,
  often with exact or near-exact matching rather than fuzzy text similarity (Babel Street
  Address Similarity docs; Robin Linacre's address-matching writeup, both Tier
  2/3-practitioner but consistent with the pattern). No paper was found benchmarking this
  exact numeric-token-set technique at our scale — this is a project-specific empirical
  finding more than a literature-imported one.
- **Evidence strength:** STRONG *for our data specifically* (Phase-1 is a direct, full-
  population measurement, not a sample); MODERATE as a generalizable literature claim
  (limited independent benchmark evidence beyond the general "numeric fields matched
  separately" pattern).
- **Transferability:** the underlying finding is ours (not imported from a paper), so
  transferability is by definition HIGH *to itself*; the open question is whether it holds
  on France (§17), which is UNKNOWN.
- **Expected benefit:** recovers a large fraction of true links independent of name
  similarity — directly complementary to routes 1.1/1.2.
- **Precision effect:** neutral at blocking (candidate generation), but promising as a
  *matching-stage feature* too (see §8 below) because numeric-token disagreement is a
  strong contradictory-evidence signal for rejecting false merges.
- **Recall/candidate-recall effect:** HIGH for the ~63.5% numeric-overlap population;
  requires §12's caveat that full address equality is far rarer (7.44%) — surrounding text
  varies heavily even when numbers match, so this is a genuinely separate signal from
  address-text blocking, not a subset of it.
- **Complete-coverage effect:** strong complementary contribution; must be combined with
  1.1/1.2 to cover cases where address is missing (§10 — S1 address never missing, but
  target-side missing in ~4.4% of true links; a numeric-only route cannot help those).
- **France robustness:** MEDIUM — France address numeric-token presence in test S1 is
  99.58% (§17 table), essentially the same order as US (99.9998%) and higher than India
  (91.27%), so the *presence* of numeric tokens is not a France-specific risk; whether
  France's postal/numbering conventions produce numeric tokens that behave the same way
  under exact-set matching is UNKNOWN (no France ground truth).
- **CPU/GPU/RAM/disk/runtime:** ESTIMATED — numeric-token extraction and exact-set-based
  inverted indexing is computationally cheaper than either lexical route above (smaller
  effective vocabulary); Phase 1 already extracted numeric tokens at full population scale
  for the audit (§12) inside the ~4-minute true-link-difficulty audit runtime (shared with
  other measurements in that pass), so extraction cost is known to be tractable.
- **Engineering complexity:** LOW.
- **Compliance/license:** CLEAR.
- **Biggest failure mode:** common/generic numeric tokens (e.g., a very frequent unit
  number or short postal prefix) could produce large false-candidate blocks, analogous to
  the name-collision groups in §14 — collision-rate auditing of numeric-token blocks has
  not yet been done and should be Phase 2's first pass on this route.
- **Earliest cheap falsification test:** measure candidate recall AND candidate-load
  distribution (mean/median/p95/p99/max) of numeric-token blocking alone on `development`;
  also measure numeric-token collision-group sizes analogous to §14's name-collision
  analysis, since Phase 1 did not yet do this for numeric tokens specifically.
- **Priority: P0.**

### 1.4 Multi-route union (token + character n-gram + numeric-address, deduplicated)
- **Problem solved:** the actual candidate-generation strategy, not a single technique —
  combines 1.1-1.3 (and optionally 1.5/1.6 below) by union, not intersection.
- **Phase-1 trigger:** the whole difficulty distribution (§13) — no single field/signal
  covers the majority of true links, so no single route can either.
- **Research basis:** every major blocking survey frames complementary/composite blocking
  as standard practice once a single method's recall ceiling is reached (Papadakis et al.
  1905.06167, "composite methods" category; Sparkly's own paper frames its contribution as
  *one* strong route, not a complete pipeline).
- **Evidence strength:** STRONG (structural argument, not a single fragile citation).
- **Transferability:** HIGH — this is an architectural principle, not a dataset-specific
  trick.
- **Expected benefit:** each route's recall gaps are filled by the others' strengths (name-
  token-zero-overlap filled by 1.2/1.3; missing-address filled by 1.1/1.2).
- **Candidate-load effect:** the critical risk — naive union can multiply candidate volume;
  must be measured explicitly (mean/median/p95/p99/max per S1, §7 of the research brief)
  and may need per-route caps (e.g., top-k per route) analogous to Sparkly's top-k design.
- **Complete-coverage effect:** this is the metric the union is explicitly optimizing for,
  given ~48% of entities need 4+ correct targets recovered simultaneously (§6).
- **France robustness:** inherits the per-route characteristics above; a union is at least
  as robust as its most-robust component route.
- **CPU/GPU/RAM/disk/runtime:** ESTIMATED — dominated by whichever component route is most
  expensive; deduplication of unioned candidate IDs per S1 is a cheap hash-set operation at
  our scale.
- **Engineering complexity:** MEDIUM (mostly in orchestration/bookkeeping, not in any
  single algorithm).
- **Compliance/license:** CLEAR.
- **Biggest failure mode:** candidate-set explosion on high-collision entities (generic
  names × generic numeric tokens) inflating `candidate_pairs.tsv` size and downstream
  matcher cost without proportional recall gain.
- **Earliest cheap falsification test:** this *is* the Phase-2 mission — see
  `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-B004.
- **Priority: P0.**

### 1.5 Sorted neighborhood / canopy clustering
- **Problem solved:** an alternative/simplification to full inverted-index blocking under
  tighter engineering constraints.
- **Phase-1 trigger:** none specific — this is a Tier-C completeness check, not evidence-
  driven.
- **Research basis:** classical (Hernandez & Stolfo, sorted neighborhood; McCallum et al.,
  canopy) — MODERATE/textbook-standard technique, not the current state of the art for
  large-scale structured-record blocking.
- **Evidence strength:** MODERATE (well-established but largely superseded by inverted-
  index/token-blocking methods for structured short-text records at this scale).
- **Transferability:** MEDIUM — the "sort by a key, slide a window" mechanism does not
  cleanly handle multi-route union or the word-order-transposition noise called out in the
  problem statement.
- **Expected benefit:** LOW incremental value over 1.1-1.4 given token/n-gram/numeric
  routes already cover the recall-driving signals found in Phase 1.
- **Priority: P2 — only revisit if 1.1-1.4's engineering cost proves higher than expected
  and a cheaper fallback is needed; do not build proactively.**

### 1.6 MinHash/LSH and PPJoin-style set-similarity joins
- **Problem solved:** approximate Jaccard-similarity retrieval at scale without full
  pairwise comparison.
- **Phase-1 trigger:** §11/§12 Jaccard quantile evidence (name-token Jaccard median 0.667,
  address-token Jaccard median 0.625) — these are exactly the similarity measures
  MinHash/LSH and PPJoin approximate/accelerate.
- **Research basis:** MinHash/LSH (Broder et al. lineage) and PPJoin/PPJoin+/MPJoin (Xiao et
  al. lineage, benchmarked in Jiang/Li's experimental evaluation and Mann et al.'s
  empirical evaluation of set-similarity-join techniques) are both STRONG, mature, well-
  benchmarked techniques.
- **Evidence strength:** STRONG (multiple independent benchmark papers).
- **Transferability:** HIGH in principle (token-set similarity retrieval is exactly our
  1.1 problem, just with an exact/near-exact Jaccard-threshold guarantee instead of
  TF-IDF's soft weighting).
- **Expected benefit:** MEDIUM incremental value over 1.1/1.2 specifically for our data —
  since Sparkly's own paper (an apples-to-apples comparison) found simple top-k TF-IDF
  blocking outperformed multiple state-of-the-art blockers, a PPJoin/MinHash-LSH
  implementation is not obviously superior to the TF-IDF route already prioritized as P0,
  and adds implementation complexity (threshold tuning, band/row LSH parameter search).
- **CPU/RAM/runtime:** ESTIMATED comparable to or cheaper than TF-IDF top-k at exact-
  threshold settings, but MinHash's approximation error and LSH's band/row tuning add a
  hyperparameter-search cost TF-IDF top-k blocking avoids.
- **Compliance/license:** CLEAR.
- **Priority: P1 — a reasonable second blocking route to benchmark against the P0
  TF-IDF/token/numeric union if that union's candidate load or recall falls short; not
  first in line given Sparkly's direct evidence that a simpler TF-IDF approach already
  outperforms more complex alternatives on comparable tasks.**

### 1.7 Dense/embedding retrieval (bi-encoder + ANN) for blocking
- **Problem solved:** semantic/soft retrieval that could in principle recover cases lexical
  methods miss (e.g., genuinely different-language name pairs with no shared n-grams).
- **Phase-1 trigger:** §17 France transfer risk (motivates a language-robust route);
  weakly by §11's zero-overlap tail (though §21 already notes the address numeric-token
  signal is the stronger, cheaper answer to that specific gap).
- **Research basis:** **Zeakis et al. (PVLDB 16(9), 2023)** performed exactly this
  comparison across 12 language models and 17 benchmarks and is the strongest available
  evidence: embeddings are a real, viable blocking signal but are not shown to
  categorically dominate lexical/TF-IDF methods, and carry meaningfully higher
  vectorization/compute cost (a research question the paper explicitly investigates:
  "how large is their vectorization overhead"). **Sparkly (contradictory evidence)** shows
  a "surprisingly strong" pure lexical TF-IDF baseline beating a suite of prior blockers,
  including DL-based ones, which tempers any assumption that embeddings are a strictly
  necessary upgrade. DeepBlocker/AutoBlock (Thirumuruganathan et al.) provide a middle
  ground: learned/dense blocking helps in some settings but requires either labeled data
  or careful unsupervised design.
- **Evidence strength:** STRONG evidence exists on both sides (embeddings help in some
  settings; a strong lexical baseline is competitive/superior in others) — this is
  precisely a case for the contradictory-evidence rule (§13 of the research brief): treat
  as genuinely unresolved for our specific data, not resolved in embeddings' favor.
- **Transferability:** MEDIUM — the benchmarks underlying both papers are EM-standard
  datasets (products, publications, restaurants), not specifically noisy multilingual
  business name/address pairs at our scale; France-language transfer specifically is
  UNKNOWN (§17) and is exactly the scenario where embeddings are theoretically most useful
  but empirically least verified for this project.
- **Expected benefit:** UNCERTAIN — plausible upside on hard-tail/France cases, unproven
  necessity given the lexical routes' already-strong coverage of the audited difficulty
  distribution (§13: only the D+F "hard middle" — 74% of true links — genuinely needs
  fuzzy/learned signal, and D already clears ≥0.5 token Jaccard on both fields, i.e. within
  reach of lexical methods).
- **Precision effect:** embeddings retrieved via ANN can introduce *semantic* false
  positives (similar-meaning but different-entity businesses) that lexical methods would
  not — a specific failure mode noted in general dense-retrieval literature and directly
  relevant given the metric's 2× precision weighting.
- **CPU/GPU/RAM/runtime:** DOCUMENTED (Zeakis et al. explicitly study vectorization
  overhead as a research question — the finding is that it is nontrivial, not free) /
  ESTIMATED for our scale: embedding ~10M+17M records with a ~0.1-0.6B-parameter model is a
  multi-hour CPU job or a much faster GPU job — GPU availability/VRAM is UNVERIFIED
  (Phase 1 could not get a reliable VRAM figure from WMI; `nvidia-smi` must be run before
  committing to this route).
- **Compliance/license:** REVIEW REQUIRED per `LICENSE_AND_TOOLING_MATRIX.md`'s
  pretrained-model interpretation section — the specific models identified (LaBSE,
  multilingual-e5-large, paraphrase-multilingual-MiniLM-L12-v2) are individually CLEAR on
  license/scale, but the *general* policy question of whether using any pretrained model at
  all is intended to be permitted is not itself verified against an organizer clarification.
- **Biggest failure mode:** committing engineering time to a route whose necessity is not
  established by the Phase-1 evidence (the hard middle is dominated by moderate-lexical-
  overlap cases, not zero-overlap ones) and whose compliance status carries an open
  question.
- **Earliest cheap falsification test:** before any embedding infrastructure work, first
  measure how much of the §13 "hard middle" (D+F buckets) the P0 lexical/numeric union
  (1.1-1.4) already recovers at the candidate-generation stage; only if a materially large,
  well-characterized residual gap remains (e.g., disproportionately concentrated in France
  or in the zero-name-overlap tail) does an embedding route's expected value clear its
  cost.
- **Priority: P1 (hard-tail/France-robustness follow-up), not P0.**

### 1.8 Learned/supervised blocking (SC-Block, Block-SCL, supervised meta-blocking edge
pruning)
- **Problem solved:** using labels to learn which candidate pairs from a coarse blocking
  pass are worth keeping (pruning), or to learn a blocking representation directly.
- **Phase-1 trigger:** weak — we do have labels (`train_ground_truth.tsv`), so supervision
  is *available*, but Phase 1 found no evidence that unsupervised lexical/numeric routes
  are recall-insufficient, which is the usual justification for learned blocking's added
  complexity.
- **Research basis:** SC-Block (arXiv:2303.03132), Block-SCL (arXiv:2207.02008), Supervised
  Meta-blocking (Papadakis et al., PVLDB 7) — all MODERATE-STRONG within their own
  benchmarks.
- **Evidence strength:** MODERATE (real technique, meaningful results, but on benchmarks
  much smaller than our scale).
- **Transferability:** MEDIUM — training/maintaining a learned blocker adds a full model-
  training loop to the *blocking* stage, on top of the matching-stage model already
  required; the added engineering and compute cost is significant relative to demonstrated
  necessity for our specific difficulty distribution.
- **Expected benefit:** LOW-MEDIUM incremental value given the P0 union's expected recall
  profile; potentially useful later as a *pruning* step if candidate load from the P0 union
  proves too large for the matcher to process economically.
- **Priority: P2 — revisit only if the P0 union's candidate load (not recall) becomes the
  bottleneck.**

### 1.9 Reverse (doc→S1) retrieval (new, added after teammate-research reconciliation)
- **Problem solved:** the same candidate-generation problem as §1.1-1.4, queried in the
  opposite direction — each S2/S3 document retrieves its own top-r nearest S1 entities,
  rather than each S1 retrieving its top-k documents.
- **Phase-1 trigger:** `docs/DATASET_AUDIT.md` §8 (VERIFIED) — every true-matched S2/S3 id
  belongs to exactly one S1 entity (`max_s1_degree_for_any_target == 1`), so the target-side
  relationship is a partition; this is precisely the structural precondition reverse
  nearest-neighbor retrieval exploits (a document has at most one correct S1 to be retrieved
  by), and is the same fact EXP-M008's capacity-one conflict check already relies on at the
  matching stage.
- **Research basis:** **TEAMMATE EXPLORATORY EVIDENCE** —
  `TEAMMATE_RESEARCH_RECONCILIATION.md` §4 Finding #1 (directly verified from raw script/log
  output, not merely cited, but on a non-canonical train sample, not `development`) — reverse@1
  92.4-96.1%, reverse@2 94.0-97.1%, exceeding forward-per-source@10 (86.9-93.8%) on the same
  vectors, with a forward∪reverse union reaching 94.8-98.4% pair recall on that sample. S1 (the
  query index for the reverse
  pass) is also the smaller, already-deduplicated table, which is a favorable cost profile.
- **Evidence strength:** STRONG for the structural argument (directly tied to a VERIFIED
  Phase-1 fact); MODERATE-STRONG for the magnitude (one independent sample-scale measurement,
  not yet reproduced at `development` scale).
- **Transferability:** HIGH — the partition property is dataset-wide (VERIFIED over all
  7,638,365 training true links), not country- or sample-specific.
- **Expected benefit:** recovers true links the forward route's per-S1 top-k budget misses
  (e.g. a document whose true S1 doesn't rank in that S1's own forward retrieval order but for
  whom this document is nonetheless its nearest S1), at low marginal cost since it reuses
  EXP-B001's vectors.
- **Precision effect:** neutral at blocking; the reverse rank/margin/`is_doc_best` signals it
  naturally produces are separately valuable as P0 matching-stage exclusivity features
  (`TECHNIQUE_EVIDENCE_MATRIX.md` §10's carve-out; already in EXP-F001's inventory).
- **Recall/candidate-recall effect:** HIGH, complementary to §1.1 — measured to exceed forward
  retrieval alone in the teammate's sample.
- **France robustness:** MEDIUM — same script/vocabulary-agnostic mechanism as §1.1 (shares
  its vectors); untested for France specifically since France has no training truth.
- **CPU/GPU/RAM/disk/runtime:** MEASURED by the teammate at sample scale, ESTIMATED at our
  scale — 0.25-8 ms/document depending on memory pressure (their own reported finding: under
  memory pressure, reverse cost rose 10-30×). This sensitivity should be budgeted explicitly in
  EXP-B000-style preflight estimation for this route.
- **Engineering complexity:** LOW — reuses EXP-B001's vectorization; only the query direction
  changes.
- **Compliance/license:** CLEAR (no external data, same libraries already CLEAR in
  `LICENSE_AND_TOOLING_MATRIX.md`).
- **Biggest failure mode:** a small number of S1 entities acting as a "reverse attractor" for
  many unrelated documents (analogous to §14's name-collision groups), inflating candidate
  load on those S1s specifically — not yet measured, should be checked alongside EXP-B003's
  collision-group analysis.
- **Earliest cheap falsification test:** EXP-B001R itself — see
  `RESEARCH_TO_EXPERIMENT_PLAN.md`.
- **Priority: P0.**

---

## 2. Name / address similarity and normalization

### 2.1 Character-level similarity (Levenshtein, Jaro-Winkler, q-gram cosine)
- **Phase-1 trigger:** §11 edit-ratio quantiles (median 0.866 on the sampled 300K true
  links, but a real 5-10% tail below ~0.1); §13 bucket taxonomy.
- **Research basis:** Cohen/Ravikumar/Fienberg (2003) — STRONG, direct empirical comparison
  of exactly these metrics for name matching; RapidFuzz implements all of them with a
  documented MIT license and C++ performance.
- **Transferability:** HIGH.
- **Priority: P0 (feature engineering baseline)** — cheap, well-understood, and RapidFuzz
  makes it computationally trivial at our scale for a bounded candidate set (post-blocking,
  not the full Cartesian product).

### 2.2 Token-level similarity (Jaccard, Dice, overlap coefficient, TF-IDF cosine)
- **Phase-1 trigger:** §11/§12 Jaccard quantiles directly; these are the same statistics
  Phase 1 already computed for the audit, so the features are proven measurable at our
  scale.
- **Research basis:** standard, STRONG (same literature as §1.1-1.2 above).
- **Priority: P0 (feature engineering baseline).**

### 2.3 Business-name structural handling (legal suffixes, abbreviations, DBA, word order)
- **Phase-1 trigger:** README noise-pattern documentation (Corp/Corporation, Pvt/Private,
  Ltd/Limited, DBA/trade names, punctuation, word-order transpositions) plus §17's
  France-specific finding of a recurring "SARL" legal-form collision pattern — the *same
  class* of noise as US/India's Ltd/Pvt Ltd, different vocabulary.
- **Research basis:** practitioner-consensus literature (Databar, Openprise, Tilores,
  BizVaultPro writeups) converges on: strip common legal suffixes for matching *features*
  but retain an exception path for names where the "suffix" is integral to brand identity
  (their own example: "The Limited"), and always normalize before deduplication/blocking.
  This is Tier 2/3 practitioner literature, not peer-reviewed, but the over-normalization
  risk it describes is directly corroborated by our own Phase-1 evidence: normalized names
  are only 88-93% unique within a source (§14), meaning aggressive normalization that
  collapses more names together makes the collision problem *worse*, not better, unless
  paired with the address/numeric-token disambiguation already prioritized above.
- **Evidence strength:** MODERATE (practitioner consensus + directly corroborating Phase-1
  data, but not a peer-reviewed benchmark of suffix-stripping policies specifically).
- **Transferability:** MEDIUM-HIGH for US/India (matches documented noise patterns exactly);
  UNKNOWN for France's specific legal-form vocabulary (only one example pattern observed in
  §14/§17, not a validated list).
- **Expected benefit:** feature-quality improvement for the matcher stage; a real but
  secondary lever compared to blocking coverage.
- **Biggest failure mode:** aggressive suffix-stripping that increases name collisions
  (§14) without a compensating address/numeric-token check — this is explicitly why
  Phase 1's own evidence (§14, §21) frames joint name+address disambiguation as necessary,
  not name normalization alone.
- **Priority: P1** — normalization-version experiments belong in the feature/matcher stage,
  not blocking, and should be evaluated by their effect on matcher precision specifically
  (measure false-merge rate before/after a normalization change), not assumed safe.

### 2.4 Address matching without geocoding (numeric-token exact/partial, component
handling)
- See §1.3 above for the blocking-stage treatment. As a **matching-stage feature**, the
  same numeric-token signal (exact-set agreement, partial overlap, conflicting numeric
  tokens as *negative* evidence) is separately valuable: a name-similar pair with
  conflicting house numbers is a specific, interpretable, high-value feature for rejecting
  false merges, directly serving the metric's precision weighting.
- **Priority: P0** (as a matcher feature, in addition to its P0 blocking-route role).

---

## 3. Classical matching models

### 3.1 Fellegi-Sunter / probabilistic linkage (e.g., via Splink)
- **Phase-1 trigger:** none specific to justify as the *sole* matcher, but the frozen
  metric and validation infrastructure (§ VALIDATION_DESIGN.md) is compatible with any
  matcher family, including this one.
- **Research basis:** foundational, STRONG; Splink is a mature, MIT-licensed, actively
  maintained implementation with documented scale up to 100M+ records via its Spark
  backend and ~1M records/minute on a laptop via its DuckDB backend.
- **Transferability:** HIGH for the general framework; MEDIUM for whether it, specifically,
  beats a GBDT-on-engineered-features baseline on *this* data — no direct evidence either
  way from Phase 1.
- **Expected benefit:** a well-understood, interpretable, fast-to-iterate baseline matcher;
  EM-estimated match weights are a genuinely different modeling paradigm from a
  discriminatively-trained GBDT and provide a useful cross-check.
- **Priority: P1** — a strong second baseline once feature engineering exists, run
  alongside (not necessarily instead of) the GBDT baseline in §3.2.

### 3.2 GBDT on engineered pair features (LightGBM/XGBoost/CatBoost)
- **Phase-1 trigger:** §13's difficulty-bucket distribution (74% of true links need genuine
  fuzzy/learned combination of moderate signals — exactly the setting where a
  discriminative model combining several engineered similarity features tends to work
  well); §21 explicitly frames this as the appropriate next step ahead of embedding models,
  "per the operating principle against 'fanciest model = best.'"
- **Research basis:** GBDTs are the dominant practical choice for tabular pairwise-
  similarity-feature classification across record-linkage and entity-matching literature
  and industry practice; well-documented strength on engineered nonlinear feature
  interactions, missingness handling (native in all three libraries), and calibration
  behavior (Niculescu-Mizil & Caruana specifically studied boosting's calibration needs).
- **Evidence strength:** STRONG (widely reproduced, not a single fragile result).
- **Transferability:** HIGH — this is a generic, well-understood technique family, not
  benchmark-specific.
- **Expected benefit:** primary candidate for the first real matcher; directly consumes the
  features from §2.1-2.4/§1.3.
- **Compliance/license:** CLEAR on all three implementations (MIT/Apache-2.0), trivially
  within the 8B-parameter ceiling (GBDT parameter counts are not comparable to neural
  parameter counts in the same sense, but total model size is orders of magnitude smaller
  than any LLM-scale constraint concern).
- **Biggest failure mode:** naive training on random negatives will likely be far too easy
  given the collision structure in §14 — see hard-negative mining in §5 below.
- **Priority: P0.**

### 3.3 dedupe / recordlinkage (Python libraries)
- **Evidence strength:** MODERATE — real, MIT/BSD-licensed tools, but `dedupe`'s active-
  learning-oriented workflow and `recordlinkage`'s pandas-centric design are both better
  suited to small-to-medium labeling-in-the-loop workflows than to our already-labeled,
  multi-million-row batch setting.
- **Transferability:** MEDIUM — useful as a reference implementation / sanity-check tool,
  not as the production pipeline backbone.
- **Priority: P2** — reference/comparison only.

---

## 4. Deep / transformer entity matching

### 4.1 Ditto / cross-encoder transformer pair classifiers
- **Phase-1 trigger:** §13's hard middle again, plus §17's France transfer question (a
  multilingual pretrained transformer could plausibly help there specifically).
- **Research basis:** Ditto (Li et al., PVLDB 2020) reports up to 29% F1 improvement over
  prior SOTA and reaching prior SOTA with roughly half the labeled data, using BERT-family
  models with domain-knowledge injection and data augmentation; extended in *Effective
  entity matching with transformers* (VLDB Journal 2023). This is STRONG evidence *of the
  technique's benchmark performance*, on Magellan/EM-benchmark-scale datasets (hundreds of
  thousands of pairs, not tens of millions of candidate pairs).
- **Evidence strength:** STRONG for benchmark performance; the *scale* question (running a
  transformer cross-encoder over however many millions of post-blocking candidate pairs our
  pipeline produces) is not directly addressed by the paper's own benchmarks, which are
  smaller.
- **Transferability:** MEDIUM — task type (pairwise EM) is a strong match; scale
  (inference over potentially tens of millions of candidate pairs) is a real gap between
  the paper's setting and ours, and cross-encoder inference cost scales linearly with
  candidate-pair count, unlike a GBDT's near-negligible per-pair cost.
- **Compliance/license:** REVIEW REQUIRED (same pretrained-model policy question as §1.7)
  plus a *separate* practical question: which specific BERT-family checkpoint, its exact
  license, and its parameter count would need individual verification before any commitment
  (not done in this pass since no specific checkpoint was selected).
- **Expected benefit:** plausible quality improvement specifically on the hard-middle/
  France-transfer cases; unproven necessity given GBDT-on-engineered-features is not yet
  tried or shown insufficient.
- **Biggest failure mode:** inference cost at candidate-pair scale, and committing to a
  more complex, harder-to-audit, license-ambiguous component before establishing whether
  the simpler GBDT baseline already reaches the target range.
- **Earliest cheap falsification test:** only justified once the GBDT baseline (§3.2) has a
  measured validation macro F0.5 and an error-taxonomy breakdown (§35 of the research
  brief) showing a specific, sizable residual failure mode (e.g., hard-tail lexical cases,
  or French entities specifically) that a cross-encoder is plausibly positioned to fix —
  i.e., restrict to the hard tail only, not the full candidate set, if adopted at all.
- **Priority: P1, scoped narrowly to a hard-tail reranker — not P0, not a full-candidate-
  set matcher.**

### 4.2 DeepMatcher-style RNN/attention aggregation models
- **Research basis:** DeepMatcher (Mudgal et al.) predates and is generally outperformed by
  transformer-based approaches like Ditto on the same Magellan benchmark family (per the
  "Neural Networks for Entity Matching" survey and Ditto's own reported comparisons).
- **Evidence strength:** MODERATE, largely superseded.
- **Priority: REJECT** — no evidence it would outperform either the GBDT baseline or a
  transformer reranker if one is eventually justified; adds engineering cost for a weaker
  expected outcome than either alternative already on the list.

---

## 5. Embeddings (retrieval and features) — see §1.7 for the blocking-stage treatment

As a **matching-stage feature** (e.g., cosine similarity between name/address embeddings as
one input feature to the GBDT, rather than as the retrieval mechanism itself): lower risk
than full embedding-based retrieval since it doesn't gate recall, but still carries the
same compliance-interpretation flag and the same "unproven necessity given the difficulty
distribution" caveat as §1.7.
- **Priority: P2** — only after the GBDT baseline's error analysis shows a residual gap
  embeddings are plausibly positioned to close.

---

## 6. Hard-negative mining

- **Phase-1 trigger:** §14 (name-collision groups up to ~460-way; near-perfect (name,
  address) joint uniqueness) is direct, measured evidence that random negatives would be
  far too easy — a matcher trained only on random non-matches would not learn to
  distinguish "same generic name, different business" from "same name, same business,"
  which is exactly the case the metric's precision weighting punishes most.
- **Research basis:** Kalantidis et al. (NeurIPS 2020) on synthesizing hard negatives;
  NV-Retriever (arXiv:2407.15831) on the *false-negative contamination* risk in naive
  hard-negative mining (their finding that ~70% of the "hardest" similarity-ranked
  negatives in MS MARCO were actually mislabeled positives) — directly relevant caution:
  a naive "hardest blocked candidates that aren't in ground truth" mining strategy risks
  mislabeling true matches that the ground truth happens not to enumerate as candidates
  (not a concern here specifically, since our ground truth is complete per §4 audit
  integrity — zero orphaned targets — but the *general* principle that hard-negative
  mining can silently corrupt training data if applied carelessly still applies to any
  synthetic hard-negative construction from blocking output).
- **Evidence strength:** STRONG for the general principle; the specific construction
  strategy for our data (same/similar name + conflicting numeric address token; high-
  blocking-score non-matches) is a direct, low-risk application of Phase-1's own collision
  findings, not a literature import needing separate validation.
- **Transferability:** HIGH.
- **Expected benefit:** directly targets the metric's precision weighting and the specific
  collision structure Phase 1 measured.
- **Priority: P0** — construct hard negatives from (a) the P0 blocking union's own output
  (candidates that are not true matches) and (b) explicit collision groups (§14), using the
  frozen `development` split only, never `validation_v1` (protects the frozen benchmark per
  `docs/VALIDATION_DESIGN.md`'s model-selection policy).

---

## 7. Multi-match decision logic

- **Phase-1 trigger:** §6 (47.96% of entities need 4+ correct targets; max 11); §7 (80.48%
  of non-singleton entities need matches in *both* S2 and S3 simultaneously).
- **Research basis:** no single canonical paper "solves" set-valued multi-match prediction
  for this problem shape; the relevant literature is a combination of (a) independent
  pairwise classification with per-entity aggregation (standard in Fellegi-Sunter and GBDT
  pipelines), and (b) selective-prediction/threshold theory (Chow 1970 and modern
  successors) for where to set the accept/reject boundary per candidate.
- **Evidence strength:** MODERATE-STRONG for the component pieces; WEAK/absent for a
  ready-made "multi-match decision policy" specific to macro F0.5 optimization — this is
  primarily an empirical, project-specific design question.
- **Transferability:** the component techniques (calibrated independent scoring +
  threshold) are HIGH-transferability building blocks; the *policy* (fixed global
  threshold vs. per-entity adaptive threshold vs. top-k) must be tuned/selected on our own
  `development` data.
- **Recommended direction (research-backed, not proven):** independent calibrated pairwise
  scoring per candidate, then accept all candidates clearing a *tuned* threshold per
  entity — NOT argmax/top-1 (would be structurally wrong for 94% of entities, §6) and NOT a
  hard one-match-per-source constraint (unsupported — 80.48% of entities have matches in
  *both* sources simultaneously, §7).
- **Priority: P0** (this decision logic must exist for any submission to be scored at all;
  the open question is *which* thresholding policy, to be settled empirically in Phase
  3+/4+, not literature-resolvable alone).

---

## 8. Singleton / abstention

- **Phase-1 trigger:** §5 (5.58% singleton rate — real but not dominant); §19 (singleton-
  only ceiling ≈0.944, i.e. most of the score budget is won on non-singletons, not
  singleton precision alone).
- **Research basis:** Chow's (1970) reject-option framework and modern selective-prediction
  literature formalize exactly this tradeoff (confidence threshold that minimizes risk given
  an abstention cost) — MODERATE-STRONG as a general framework, though not entity-
  resolution-specific in its most commonly cited form.
- **Transferability:** MEDIUM-HIGH — the "predict empty when below-threshold" decision is a
  direct application of the same accept/reject threshold used for multi-match logic (§7
  above); a singleton is simply the case where *no* candidate clears the threshold, not a
  structurally separate classifier necessarily needed.
- **Expected benefit:** a well-calibrated shared threshold, tuned on `development`, is
  expected (INFERRED, not proven) to handle singletons adequately given their proportionally
  small share of the entity population — a dedicated singleton classifier is added
  complexity whose necessity is unproven by Phase 1.
- **Priority: P1** — start with the shared-threshold approach (P0 §7); treat a dedicated
  singleton classifier as a P1/P2 follow-up only if error analysis shows systematic
  singleton-specific failure the shared threshold cannot fix.

---

## 9. F0.5 / calibration

- **Phase-1 trigger:** the metric itself (macro F0.5, 2× precision weighting, §19 error-
  budget analysis).
- **Research basis:** Niculescu-Mizil & Caruana (boosted-tree calibration), Platt scaling,
  isotonic regression — STRONG, standard toolkit; ScienceDirect's F-measure threshold-
  optimization paper directly addresses macro-averaged F-measure threshold selection.
- **Transferability:** HIGH.
- **Expected benefit:** calibrated probabilities make the shared-threshold policy in §7/§8
  meaningful and tunable rather than an arbitrary raw-score cutoff; isotonic regression is
  preferred over Platt scaling once the calibration set is large (our `development` split
  is enormous relative to the ~2000-case threshold literature flags as Platt's comparative
  advantage zone) — GBDT calibration should default toward isotonic regression given
  `development`'s scale, with Platt as a fallback if isotonic overfits on any thin stratum.
- **Priority: P0** (required for §7/§8 to function as intended, not merely a nice-to-have).

---

## 10. Graph / collective ER

- **Phase-1 trigger:** §8 — measured **zero** target reuse across Source-1 entities; no
  connected-component grouping was even needed for the validation split.
- **Research basis:** correlation clustering / collective ER literature is STRONG in
  general but explicitly targets problems where matches propagate transitively across a
  graph with shared targets — a structural precondition Phase 1 measured as **absent**
  from this dataset.
- **Transferability:** LOW *for this dataset specifically* — the technique family is
  well-evidenced elsewhere, but the burden-of-proof condition set in
  `docs/DATASET_AUDIT.md` §8 ("if literature shows little expected value here, deprioritize")
  is met: there is no cross-S1 target-sharing structure for collective/graph methods to
  exploit.
- **Priority: REJECT** for *transitive-closure/correlation-clustering* modeling
  specifically — do not build that logic; independent per-(S1, candidate) classification is
  structurally sufficient given the measured graph shape.
- **Narrow carve-out (added after independent audit review — this is not a reversal of the
  REJECT above, it is a scope clarification):** the same target-reuse evidence that rules
  out graph modeling (max true-target degree of exactly 1) also licenses one cheap,
  deterministic, non-graph precision control: if two different S1 entities' *predicted*
  match sets both claim the same S2/S3 ID, at most one of those assignments can be correct
  under the measured training invariant, so the lower-confidence one should be dropped. This
  is a single conflict check over already-independently-scored pairs, not transitive
  closure, not clustering, and does not require building any graph structure. See
  `RESEARCH_TO_EXPERIMENT_PLAN.md` EXP-M008. Priority for this narrow check specifically:
  **P1** (cheap, low-risk, worth testing once a baseline matcher exists — not a load-bearing
  P0 dependency).

---

## 11. Multilingual / France transfer

- **Phase-1 trigger:** §17 (structural similarity but unknown transfer), §19 (France is
  ~15% of every test table — a live risk, not a marginal edge case).
- **Research basis:** multilingual entity-matching literature (arXiv:2205.15712 Polish
  product-matching benchmark; arXiv:2010.09828 cross-lingual entity-linking transfer)
  consistently finds cross-lingual transfer is possible but imperfect and benchmark/
  language-pair-dependent — MODERATE evidence, not a guarantee.
- **Transferability:** MEDIUM — France is a single new country/language, not a fully
  unseen script (Latin alphabet, related legal-form vocabulary pattern per §17/§14), which
  is a substantially easier transfer case than the cross-script examples in most cross-
  lingual entity-linking literature.
- **Recommended direction:** prioritize language-robust building blocks already prioritized
  above for other reasons (character n-gram/TF-IDF blocking §1.2, numeric-token signal
  §1.3/§2.4, Unicode-safe normalization) rather than introducing a France-specific
  component (an "India-abbreviation dictionary" or "US address parser" anti-pattern
  explicitly warned against in the research brief) — this is the same set of techniques
  already P0 for other reasons, so France-robustness is mostly a byproduct of good general
  design, not a separate workstream.
- **Priority:** the *techniques* are already P0 (§1.2, §1.3, §2.4); the *specific
  validation* of France behavior is impossible pre-submission (no France ground truth
  exists) and must rely on structural proxies (candidate load/recall statistics broken out
  by country on `development`, even though `development`'s France population is also zero)
  plus post-submission public-leaderboard monitoring **as secondary evidence only**, per
  `docs/VALIDATION_DESIGN.md`'s model-selection policy.

---

## 12. Techniques explicitly deprioritized (Tier C / REJECT summary)

| Technique | Reason |
|---|---|
| DeepMatcher-style RNN/attention EM | superseded by transformer approaches; no evidence of advantage over GBDT or Ditto-style reranking for this task |
| Full graph/collective ER (transitive closure, correlation clustering) | Phase-1 measured zero target-reuse structure for it to exploit (§8) |
| Sorted neighborhood / canopy clustering as primary blocking | superseded by inverted-index/token/TF-IDF methods for this record shape and scale; no Phase-1 evidence trigger |
| dedupe / recordlinkage as production pipeline backbone | workflow/scale mismatch (active-learning-oriented, smaller-scale design center); fine as reference tools only |
| Full-candidate-set transformer cross-encoder matching | inference cost scales with candidate-pair count; unproven necessity before a GBDT baseline exists; license/compliance review still open |
| Country as a hard blocking filter | 100% consistent on train (§9) but UNVERIFIED for France (§17) — hard-filtering risks catastrophic recall loss on ~15% of test if the pattern doesn't hold; must be measured with and without before any decision (§20/§21) |
