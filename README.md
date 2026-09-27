# Amazon ML Challenge 2026 — Business Entity Resolution

Solution developed by Team Et Cetera for the Amazon ML Challenge 2026.

This repository contains our team's solution developed for the Amazon ML Challenge 2026.
It is an independent hackathon project and is not an official Amazon project or affiliated
with Amazon in any way.

## Overview

The task is business entity resolution: given a primary table of business records
(Source 1) and two target tables (Source 2, Source 3), predict which target records refer
to the same real-world business as each Source-1 record. Each record has a business name,
address, and country. Training data covers the United States and India only; the test set
additionally includes France, with zero training examples for that country — an explicit
open-set generalization requirement running through the whole design.

Scoring is a macro-average, per Source-1 entity, of F0.5 (precision weighted 2x over
recall). Singleton entities (no true match) score full credit only when predicted with an
empty match set, and any false match on a true singleton scores zero — so precision
discipline matters as much as recall.

## Approach

```
raw records
  -> normalization (script-aware, alias map learned from training links only)
  -> candidate retrieval (hashed combination-key TF-IDF, per-country document frequency)
  -> 51 pairwise features (fuzzy similarity, token/number overlap, IDF, retrieval rank)
  -> stage-1 LightGBM matcher
  -> stage-2 LightGBM matcher (entity-level / collective competition features)
  -> per-entity adaptive-threshold decoder
  -> matching_results.tsv, candidate_pairs.tsv
```

Full technical detail, including the validation design, feature definitions, and the
France open-set handling, is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Validation

All model and decision-rule development used a frozen, entity-level, stratified train/dev/
calibration split (seed 42); a further holdout partition was reserved for a single
confirmatory read late in the project. The official macro-F0.5 evaluator was independently
verified against the competition's worked example plus 16 additional cases before any
model development began. See [`docs/VALIDATION_DESIGN.md`](docs/VALIDATION_DESIGN.md).

## Results

Metric: macro-averaged F0.5, evaluated on the public leaderboard.

| | Score |
|---|---|
| Best verified public result (final submission) | **0.973462** |
| `dev_eval` (held-out, same pipeline) | 0.981851 |
| `calibration_holdout` | 0.981899 |

The full submission history — including architectures that were tried and not adopted —
is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Every figure reported anywhere in this
repository is read directly from a logged artifact or file hash; nothing is estimated.

## Repository structure

```
README.md, LICENSE, .gitignore
docs/                                    methodology, validation design, experiment
                                          history, reproducibility, and research notes
student_resource/
  utils/validate_submission.py           official submission-format validator
  code/business_entity_resolution/
    src/                                 normalization, retrieval, features, stage-1/2
                                          models, evaluation
    scripts/                             one script per pipeline stage (see its README)
    tests/                               unit and integrity tests
    artifacts/                           small, non-regenerable artifacts required for
                                          exact reproduction: frozen train/dev/calibration
                                          splits and the frozen decoder rule (with SHA-256
                                          checksums)
    README.md                            exact reproduction commands, in pipeline order
    requirements.txt
```

`experiments/` and `submissions/` (large intermediate artifacts and staged submission
snapshots produced while running the pipeline) are not published — see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for exactly how to regenerate them
and how the small, non-regenerable pieces they'd otherwise hold are preserved in
`artifacts/` instead.

## Setup and reproduction

```
py -3.12 -m venv .venv
.venv/Scripts/pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

Obtain the official Amazon ML Challenge 2026 dataset yourself (see Data Notice below) and
place it at `student_resource/dataset/{train,test}/`. Then follow
[`student_resource/code/business_entity_resolution/README.md`](student_resource/code/business_entity_resolution/README.md)
for the exact command sequence, or [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
for a dependency-graph overview. Full reproduction from raw data (retrieval through
model training) is computationally nontrivial — peak memory around 22 GB during feature
construction, and multiple pipeline stages take tens of minutes each on a modern
multi-core machine.

## Validating an output

```
cd student_resource
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Limitations

- The final decoder's per-entity adaptive thresholds are validated only for countries seen
  during training (US, India); the pipeline falls back to a single global threshold for
  any other country, including France, rather than extrapolating an unvalidated rule.
- LightGBM training is seed-fixed but multi-threaded and not configured for bit-exact
  determinism; re-running the training commands reliably reproduces the reported scores
  but is not guaranteed to produce byte-identical model files (see
  [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)).
- A small number of predicted test entities (well under 0.1%) exceed the maximum match
  count observed in training, concentrated in generic business names in the France
  partition — a known, quantified, low-impact artifact of the open-set distribution shift.

## Data and Competition Notice

Competition data is not included in this repository. Obtain the official Amazon ML
Challenge 2026 dataset through the official challenge source, subject to its terms. This
repository contains only our team's source code, documentation, and derived (non-dataset)
artifacts required to reproduce our pipeline.

## Team

**Et Cetera**

- Anmol Trivedi
- Ashmita Dutta — Research

## License

Code and documentation in this repository are MIT licensed — see [`LICENSE`](LICENSE).
This does not extend to the competition's own problem statement, dataset, or other
challenge-provided materials, which are not included here. Third-party dependencies
retain their own licenses.
