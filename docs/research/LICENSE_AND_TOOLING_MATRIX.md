# License & Tooling Matrix — Phase 1.5

Constraint being checked against, per `docs/OFFICIAL_REQUIREMENTS.md` (VERIFIED from both
official PDFs): **final model must be MIT or Apache-2.0 licensed, and ≤ 8B parameters.**
Nothing below was installed, downloaded, or executed — this is a compliance/feasibility
survey only (§55 no-implementation gate).

Status legend: **CLEAR** (license/scale verified compatible, no other flag), **REVIEW
REQUIRED** (a genuine open question remains — stated explicitly), **REJECT** (incompatible
on license, scale, or another hard constraint).

**Scope clarification (added after independent audit review):** a `CLEAR` status below
means that specific library or model's own license and parameter count are compatible with
the stated constraint (MIT/Apache-2.0, ≤8B parameters) as an input to a final submission —
it is **not** itself a determination that the *final assembled submission* is fully
compliant. A GBDT library being MIT-licensed says nothing about whether normalization code,
data handling, or any other pipeline component introduces a separate compliance issue; the
final-submission compliance check (per `docs/FINAL_DELIVERABLE_CHECKLIST.md`) is a separate,
later step this research phase does not perform.

## Libraries — general-purpose / tabular ML

| Name | Type | License (verified from) | Windows | Py 3.12 | Scale fit (our data) | Status |
|---|---|---|---|---|---|---|
| scikit-learn | ML toolkit | BSD-3-Clause (well-established, standard package) | Yes | Yes | Sparse-matrix ops fine at our scale for feature engineering / calibration (`CalibratedClassifierCV`), not for the full candidate matrix in memory | CLEAR |
| LightGBM | GBDT | MIT (verified: repo `LICENSE` file) | Yes | Yes | Designed for large tabular data; leaf-wise growth, histogram-based — fits a multi-million-row pair-feature table | CLEAR |
| XGBoost | GBDT | Apache-2.0 (verified: repo `LICENSE` file) | Yes | Yes | Same scale class as LightGBM | CLEAR |
| CatBoost | GBDT | Apache-2.0 (verified: repo `LICENSE` file) | Yes | Yes | Same scale class; native categorical handling (relevant for `country`/source features) | CLEAR |
| RapidFuzz | string similarity (Levenshtein, Jaro-Winkler, token-set/-sort ratio, etc.) | MIT (verified: repo statement) | Yes | Yes | C++-backed, built for high-throughput pairwise string scoring — fits per-candidate feature computation at our scale | CLEAR |

## Libraries — entity-resolution-specific

| Name | Type | License (verified from) | Windows | Py 3.12 | Scale fit | Status |
|---|---|---|---|---|---|---|
| Splink | probabilistic (Fellegi-Sunter + EM) linkage, DuckDB/Spark backends | MIT (verified: repo `LICENSE` file) | Yes (DuckDB backend) | Yes | Publisher-documented ~1M records/minute on a laptop (DuckDB backend); Spark backend for 100M+. Directly relevant scale class for our 2.2M-17M row tables | CLEAR |
| dedupe (Python) | active-learning fuzzy matching/dedup | MIT (verified: PyPI page) | Yes | REVIEW — check current package's stated Python floor/ceiling before use; not verified in this pass | Designed more for human-in-the-loop labeling workflows on smaller data; scale fit to our row counts is UNVERIFIED | REVIEW REQUIRED (scale/workflow fit, not license) |
| recordlinkage (Python) | indexing + comparison + classifiers | BSD-3-Clause (verified: PyPI page) | Yes | REVIEW — not verified in this pass | Built on pandas; likely fine for feature computation on a pre-blocked candidate set, not for full Cartesian indexing at our scale | REVIEW REQUIRED (scale/workflow fit, not license) |

## ANN / vector search (only if a dense/embedding route is used)

| Name | Type | License (verified from) | Windows | Py 3.12 | Scale fit | Status |
|---|---|---|---|---|---|---|
| FAISS | ANN library (IVF, PQ, HNSW coarse quantizer) | MIT (verified: repo `LICENSE` file) | Yes (via conda/pip builds; native Windows build support varies by version — verify at install time) | Yes | PQ-compressed indexes documented to hold billions of vectors in <100GB RAM — comfortably covers our ~10M-vector scale if an embedding route is used | CLEAR (license); Windows build path should be re-checked at implementation time |
| hnswlib | ANN library (pure HNSW) | Apache-2.0 (verified: repo `LICENSE` file) | Yes (header-only C++/Python) | Yes | Best when index fits in RAM; ~10M float32 vectors at e.g. 384-dim ≈ 15GB — feasible within our measured 31.6GB RAM budget but leaves little headroom alongside the rest of the pipeline in the same process | CLEAR (license); RAM budget flagged |

## Pretrained embedding models (only relevant if an embedding retrieval/feature route is adopted — not yet decided)

| Name | Params | License (verified from model card) | Languages | ≤8B? | Status |
|---|---|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ~0.1B (118M) | Apache-2.0 | 50 | Yes | CLEAR on license/scale |
| `intfloat/multilingual-e5-large` | ~0.56B (24 layers, 1024-dim) | MIT | 100 (via XLM-R base) | Yes | CLEAR on license/scale |
| `sentence-transformers/LaBSE` | ~0.5B | Apache-2.0 (ported weights; original Google LaBSE) | 109 | Yes | CLEAR on license/scale |

All three are comfortably inside the 8B-parameter ceiling and carry MIT/Apache-2.0
licenses on the model card as published. **Compliance interpretation status — REVIEW
REQUIRED (see below):** whether using *any* pretrained general-purpose model (embeddings
or otherwise) is permitted at all under the fair-play rule is not itself settled by a
license check; that is a separate policy question addressed next.

## Pretrained-model permissibility — compliance interpretation

The official requirements (`docs/OFFICIAL_REQUIREMENTS.md`, VERIFIED against both PDFs)
state two things that must be read together:

1. **Fair play ban:** "No external business/entity identity lookup or enrichment: no
   commercial entity-resolution APIs, no government business-registration lookups, no
   geocoding APIs, no internet enrichment of the supplied entities... General technical
   research (algorithms, libraries, papers, ML methods) is a separate matter and not
   restricted."
2. **Model restriction:** "Final model must be MIT or Apache 2.0 licensed, and ≤ 8 billion
   parameters" — stated as a property of the *final model*, with no separate clause
   distinguishing a model trained from scratch on supplied data from one that starts from
   general-purpose pretrained weights.

**Reading (INFERRED, not an organizer clarification):** the ban targets looking up or
enriching the *specific supplied business records* (name/address lookups against external
registries, geocoders, maps, or other ER services) — not the *general* practice of using a
pretrained language/embedding model whose weights encode no information about the specific
S1/S2/S3 records in this dataset. Under this reading:

- (A) External lookup on supplied entities — PROHIBITED, unambiguous.
- (B) Using a generally pretrained representation (e.g., a multilingual sentence-embedding
  model trained on public web text unrelated to this competition) as a component, subject
  to the MIT/Apache-2.0 + ≤8B constraint — appears PERMITTED under this reading.
- (C) Fine-tuning such a model only on the supplied training data — appears PERMITTED,
  same reasoning.
- (D) Introducing an external *business/entity dataset* to join against or enrich supplied
  records (e.g., a public business registry snapshot, a knowledge-graph dump of company
  names) — PROHIBITED, same class of violation as (A).

**Status: COMPLIANCE REVIEW REQUIRED.** No FAQ, clarification thread, or organizer
Q&A was located distinguishing (B)/(C) explicitly from (A)/(D) beyond the two PDFs already
read into `docs/OFFICIAL_REQUIREMENTS.md`. The reading above is the most natural one
available from the text actually supplied, but it is an interpretation, not a verified
organizer statement, and should not be treated as a green light to adopt an embedding
model as a load-bearing P0 dependency without one more explicit check (re-reading any
later-released FAQ/clarification, or a direct organizer query if a channel exists) before
committing engineering time. This status is therefore **not promoted to an unconditional
P0 recommendation** per the research brief's own instruction (§4), consistent with
`TECHNIQUE_EVIDENCE_MATRIX.md`'s treatment of embedding-based routes as P1/P2, not P0.

## What was explicitly NOT done here (§55 compliance)

- No package was installed.
- No model was downloaded.
- No external business/entity dataset was queried or joined.
- No supplied business name or address was pasted into any search engine, license page,
  or model card lookup — every query above was generic (library names, model names,
  algorithm names) or referenced a specific *technical* GitHub/PyPI/HuggingFace page, never
  a competition record.
