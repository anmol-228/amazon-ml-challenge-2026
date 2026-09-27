# Sources — Phase 1.5 Research

Grouped by category. "Why relevant" is stated for every entry. Tier reflects the source
hierarchy in the research brief (Tier 1 = primary/high-authority: peer-reviewed papers,
official docs/licenses; Tier 2 = strong secondary: mature docs, practitioner writeups;
Tier 3 = discovery-only, not used as sole evidence for any recommendation below).

**Explicitly excluded during this research pass:** two GitHub repositories that a general
web search surfaced under names indicating they are other teams' solutions for this same
live Amazon ML Challenge 2026 competition (`pulkitchoudhary1602/Amazon-ML-Challenge`,
`AaryanVerma17/amazon-ml-entity-resolution-2026`). Per the fair-play rule (no competitor
solution harvesting) these were not opened, read, or used as sources anywhere in this
dossier — noted here only so the exclusion is auditable.

## Foundations

- Fellegi, I.P. & Sunter, A.B. (1969), *A Theory for Record Linkage* — canonical
  probabilistic linkage model; accessed via secondary treatments (JMIR 2022 "Data-Adaptive
  Fellegi-Sunter Model," ScienceDirect 2022 mixed-type extension, arXiv:1911.01874
  "Revisiting the probabilistic method of record linkage"). Tier 1 (foundational, widely
  reproduced).
- Papadakis, G. et al., *A Survey of Blocking and Filtering Techniques for Entity
  Resolution*, arXiv:1905.06167. Tier 1.
- Christophides, V. et al., *End-to-End Entity Resolution for Big Data: A Survey*,
  arXiv:1905.06397. Tier 1.
- Papadakis, G. et al., *(Almost) All of Entity Resolution*, arXiv:2008.04443. Tier 1.

## Blocking / Candidate Generation

- Papadakis, G. et al., *Comparative Analysis of Approximate Blocking Techniques for
  Entity Resolution*, PVLDB 9. http://www.vldb.org/pvldb/vol9/p684-papadakis.pdf. Tier 1.
- Papadakis, G. et al., *Supervised Meta-blocking*, PVLDB 7.
  http://www.vldb.org/pvldb/vol7/p1929-papadakis.pdf. Tier 1.
- Simonini, G. et al., *BLAST: a Loosely Schema-aware Meta-blocking Approach*, PVLDB 9.
  http://www.vldb.org/pvldb/vol9/p1173-simonini.pdf. Tier 1.
- **Paulsen, D., Govind, Y., Doan, A. (2023), Sparkly: A Simple yet Surprisingly Strong
  TF/IDF Blocker for Entity Matching, PVLDB 16(6):1507-1519,
  doi:10.14778/3583140.3583163.** Read in full (first 2 pages, abstract + intro +
  method). Directly relevant P0 evidence: top-k TF-IDF blocking outperformed 8 SOTA
  blockers (including DL-based) on 15 EM benchmark datasets; scales to 10M tuples in
  <100 min and 26M tuples in ~130 min on a modest Spark cluster. Tier 1, HIGH
  transferability (same problem shape: tabular records, blocking-then-matching, scale in
  the 10-30M range).
- Thirumuruganathan, S. et al. (2021), *Deep Learning for Blocking in Entity Matching: A
  Design Space Exploration*, PVLDB 14(11):2459-2472 — DeepBlocker.
  https://github.com/qcri/DeepBlocker. Tier 1.
- Zhang, W. et al., *AutoBlock* — similarity-preserving representation learning + ANN
  search for blocking. Referenced via DeepBlocker/embeddings-survey citations. Tier 2
  (accessed through secondary description, not full-text read).
- *Towards Universal Dense Blocking for Entity Resolution*, arXiv:2404.14831. Tier 1.
- **Zeakis, A., Papadakis, G., Skoutas, D., Koubarakis, M. (2023), Pre-trained Embeddings
  for Entity Resolution: An Experimental Analysis, PVLDB 16(9):2225-2238,
  doi:10.14778/3598581.3598594.** Read in full (first 2 pages, abstract + related work +
  taxonomy). Systematic comparison of 12 language models (static, BERT-based,
  Sentence-BERT) across 17 benchmarks for both blocking and matching, supervised and
  unsupervised. Directly informs the embeddings-vs-lexical tradeoff question. Tier 1,
  HIGH transferability.
- *Resource-efficient blocking: Optimizing the trade-off between effectiveness and
  scalability in entity resolution*, ScienceDirect (2026), S0306437926000992. Tier 1.
- *The impact of fine-tuning on entity resolution: An experimental evaluation*,
  ScienceDirect (2026), S095070512600170X. Tier 1.
- *Evaluating Blocking Biases in Entity Matching*, arXiv:2409.16410 — negative/critical
  evidence on blocker robustness. Tier 1.
- *SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines*,
  arXiv:2303.03132. Tier 1.
- *Block-SCL: Blocking Matters for Supervised Contrastive Learning in Product Matching*,
  arXiv:2207.02008. Tier 1.

## String / Set Similarity, Sparse Retrieval

- Jiang, Y., Li, G. et al., *String Similarity Joins: An Experimental Evaluation*, PVLDB
  (Tsinghua DB group) — PPJoin/PPJoin+/MPJoin comparison.
  https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/vldb2014-exp.pdf. Tier 1.
- Mann, W. et al., *An Empirical Evaluation of Set Similarity Join Techniques*, PVLDB 9.
  https://www.vldb.org/pvldb/vol9/p636-mann.pdf. Tier 1.
- *ShallowBlocker: Improving Set Similarity Joins for Blocking*, arXiv:2312.15835. Tier 1.
- Shrivastava, A. & Li, P., *Asymmetric Minwise Hashing*, arXiv:1411.3787. Tier 1.
- Cohen, W., Ravikumar, P., Fienberg, S. (2003), *A Comparison of String Distance Metrics
  for Name-Matching Tasks*, CMU / IJCAI-2003 IIWeb workshop.
  https://www.cs.cmu.edu/~wcohen/postscript/ijcai-ws-2003.pdf — classic comparison
  including edit-distance, Jaro-Winkler, and TF-IDF/SoftTFIDF hybrid methods for name
  matching. Tier 1.
- Jaro, M. (1989); Winkler, W. (1990) — Jaro-Winkler distance, original U.S. Census Bureau
  record-linkage work; accessed via standard secondary treatments. Tier 2.

## Sorted Neighborhood / Canopy

- *Dynamic Sorted Neighborhood Indexing Technique for Real-Time Entity Resolution*
  (Centaur repository / ResearchGate). Tier 2.
- *Parallel Sorted Neighborhood Blocking with MapReduce*, arXiv:1010.3053. Tier 1.

## Classical Matching Models

- Fellegi-Sunter family, see Foundations above.
- Splink — moj-analytical-services/splink (also ADBond fork), MIT license (verified
  directly from repository LICENSE file). https://github.com/moj-analytical-services/splink.
  Tier 1 (official repo + license).
- dedupe (Python) — MIT license (verified from PyPI project page).
  https://pypi.org/project/dedupe/. Tier 1.
- recordlinkage (Python) — BSD-3-Clause (verified from PyPI project page).
  https://pypi.org/project/recordlinkage/. Tier 1.
- LightGBM — MIT license (verified from repository LICENSE file).
  https://github.com/microsoft/LightGBM. Tier 1.
- XGBoost — Apache License 2.0 (verified from repository LICENSE file).
  https://github.com/dmlc/xgboost. Tier 1.
- CatBoost — Apache License 2.0 (verified from repository LICENSE file).
  https://github.com/catboost/catboost. Tier 1.
- RapidFuzz — MIT license (verified from repository README/license statement).
  https://github.com/rapidfuzz/RapidFuzz. Tier 1.

## Deep / Transformer Entity Matching

- Li, Y., Li, J. et al. (2020), *Deep Entity Matching with Pre-Trained Language Models*
  (Ditto), PVLDB, doi via arXiv:2004.00584. https://github.com/megagonlabs/ditto. Tier 1.
- *Effective entity matching with transformers*, The VLDB Journal (2023) — extended Ditto
  results and analysis. Tier 1.
- Mudgal, S., Li, H. et al., DeepMatcher / Magellan (SIGMOD 2018 lineage); accessed via
  https://github.com/anhaidgroup/deepmatcher and *Neural Networks for Entity Matching: A
  Survey*, arXiv:2010.11075. Tier 1/2.
- *A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching
  Algorithms*, arXiv:2307.01231 — contradictory/negative evidence on benchmark validity.
  Tier 1.
- *Heterogeneity in Entity Matching: A Survey and Experimental Analysis*,
  arXiv:2508.08076. Tier 1.
- *Machamp: A Generalized Entity Matching Benchmark*, arXiv:2106.08455. Tier 1.

## Multilingual / Transliteration

- *Multilingual Transformers for Product Matching -- Experiments and a New Benchmark in
  Polish*, arXiv:2205.15712. Tier 1.
- *Cross-Lingual Transfer in Zero-Shot Cross-Language Entity Linking*,
  arXiv:2010.09828. Tier 1.
- ParaNames (multilingual name resource, ~140M names / 16.8M entities / 400+ languages),
  referenced via search-result summary; not independently verified in depth. Tier 2.

## Hard-Negative / Contrastive Learning

- Kalantidis, Y. et al. (NeurIPS 2020), *Hard Negative Mixing for Contrastive Learning*.
  Tier 1.
- *Effective Hard Negative Mining for Contrastive Learning-Based Code Search*, ACM TOSEM.
  Tier 1.
- *NV-Retriever: Improving text embedding models with effective hard-negative filtering*,
  arXiv:2407.15831 — false-negative contamination in naive hard-negative mining (finding
  that ~70% of "hardest" MS MARCO negatives by similarity were mislabeled positives).
  Tier 1. Directly informs the false-negative-contamination risk noted in §11 below.
- Hard negative mining, general overview, Wikipedia. Tier 3 (context only).

## Calibration / Decision Theory

- Niculescu-Mizil, A. & Caruana, R., *Obtaining Calibrated Probabilities from Boosting*.
  https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf. Tier 1.
- Platt, J. (1999), Platt scaling — accessed via standard secondary treatment
  (Wikipedia) for the mechanism description; original mechanism is textbook-standard.
  Tier 2 for the secondary source, Tier 1 concept.
- Leathart, T. et al., *Probability Calibration Trees*, arXiv:1808.00111. Tier 1.
- Chow, C.K. (1970) — foundational reject-option / selective-prediction theory; accessed
  via secondary description in modern selective-prediction papers. Tier 2.
- *Threshold optimization for F measure of macro-averaged precision and recall*,
  ScienceDirect S003132032030056X. Tier 1.

## Scalability / ANN

- FAISS — MIT license (verified from repository LICENSE file).
  https://github.com/facebookresearch/faiss. Tier 1.
- hnswlib — Apache License 2.0 (verified from repository LICENSE file).
  https://github.com/nmslib/hnswlib. Tier 1.
- Malkov, Y. & Yashunin, D., *Efficient and robust approximate nearest neighbor search
  using Hierarchical Navigable Small World graphs*, arXiv:1603.09320 (HNSW paper).
  Tier 1.

## Model Cards / Licenses (Embeddings)

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — Apache-2.0, ~0.1B
  params, 384-dim, 50 languages. https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
  Tier 1 (official model card).
- `intfloat/multilingual-e5-large` — MIT license, 24 layers / 1024-dim (~560M params per
  publisher's technical report), 100 languages (via XLM-RoBERTa base).
  https://huggingface.co/intfloat/multilingual-e5-large. Tier 1.
- `sentence-transformers/LaBSE` — Apache-2.0, ~0.5B params, 109 languages; ported from
  Google's original LaBSE (TF-Hub). https://huggingface.co/sentence-transformers/LaBSE.
  Tier 1.

## Address Matching (Practitioner / Industrial)

- Babel Street, Address Similarity API docs — field-weighted matching (house number vs.
  street vs. ZIP handled by different comparators), general pattern description.
  https://docs.babelstreet.com/API/en/address-similarity.html. Tier 3 (practitioner
  documentation, used only to corroborate a pattern already evidenced by our own Phase-1
  numeric-token statistic, not as standalone justification).
- Robin Linacre (Splink's original author), *Building Accurate Address Matching Systems*.
  https://www.robinlinacre.com/address_matching/. Tier 2 (practitioner writeup from a
  credible, named ER-tooling author).

## Graph / Collective ER

- *Transforming Pairwise Duplicates to Entity Clusters for High-quality Duplicate
  Detection*, ACM JDIQ, doi:10.1145/3352591. Tier 1.
- Hassanzadeh, O. et al., *Framework for evaluating clustering algorithms in duplicate
  detection*, PVLDB, doi:10.14778/1687627.1687771. Tier 1.
- *(Almost) All of Entity Resolution*, arXiv:2008.04443 (also listed under Foundations) —
  covers correlation clustering formulation and NP-hardness. Tier 1.

## Software / Libraries — see LICENSE_AND_TOOLING_MATRIX.md for the full compliance table;
sources for each entry are the same repositories/package pages cited above.

## Notes on source-count discipline

This list intentionally stops short of exhaustive coverage of every discoverable paper.
Per the research brief's own saturation rule (§42/§59), search was continued until major
method families (blocking, string/set similarity, classical and neural matchers,
multilingual transfer, hard-negative mining, calibration, ANN/scalability) were each
represented by at least one Tier-1 primary source and, where available, one piece of
contradictory/critical evidence, and until repeated searches began returning the same
paper set rather than new families. Source count is not treated as a quality signal.
