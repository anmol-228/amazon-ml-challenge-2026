# Business Entity Resolution — reproduction guide

Pipeline: data → v2 normalization (Latin diacritics + train-derived script→Latin alias map) → pair-shingle
candidate retrieval → vectorized pair features with per-country, scale-invariant IDF → LightGBM stage 1
(trained with IDF-scale augmentation on half of the rows) → LightGBM stage 2 on cross-fitted stage-1
probabilities with owner-competition and source-split entity-support features → entity-adaptive decoder →
`output/matching_results.tsv`, `output/candidate_pairs.tsv`.

The entity-adaptive decoder uses per-entity thresholds conditioned on the target source and on the entity's
number of confident candidates. It is applied to countries present in training; any other country keeps the
global stage-2 threshold.

## Layout and environment

The scripts resolve paths relative to the official starter-kit layout. Place this folder at
`<root>/student_resource/code/business_entity_resolution/`, next to the supplied
`<root>/student_resource/dataset/{train,test}/` and `<root>/student_resource/utils/validate_submission.py`.
Intermediate artifacts are written under `<root>/experiments/`; final outputs under
`<root>/student_resource/output/`.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Python 3.12
```

Run every command below from `<root>/student_resource/code/business_entity_resolution/` as
`python -u scripts/<name>.py ...`. Peak memory is about 22 GB (feature construction, run it alone); a 32 GB
machine is sufficient. No external data, lookup, enrichment or pretrained model is used; LightGBM is MIT-licensed.

## 0. Frozen entity splits

Training Source-1 entities are split entity-disjointly: `validation_v1.tsv` (development vs untouched validation
partition) and, inside development, `phase3_split_v1.tsv` (`matcher_train` 70% / `dev_eval` 15% /
`calibration_holdout` 15%, stratified by country, singleton status and match-count bucket, seed 42). The frozen
files are shipped in `artifacts/splits/` (SHA-256 in `artifacts/splits/SHA256SUMS.txt`); copy them to
`<root>/experiments/splits/`. `scripts/build_phase3_split.py` rebuilds `phase3_split_v1.tsv` from
`validation_v1.tsv` deterministically.

## 1. Name alias map (normalization v2, `src/normalization_v2.py`)

NFKC + casefold + whitespace collapse; combining diacritics removed on Latin letters; business-name tokens in
Indic scripts mapped to Latin with `src/alias_map_matcher_train.json`, learned only from ground-truth links of
`matcher_train` entities (positional token alignment, support >= 2, share >= 0.6; 1,470 tokens; sha256
c31a468de9e192a29ed66dad0b707d8ecde41cb6ef760f4b18afe3b218b888d9, byte-identical on rebuild):

```
python -u scripts/build_alias_map.py
```

## 2. Candidate retrieval (`src/pair_retrieval.py`)

Hashed combination keys per record (name tokens and pairs, address tokens and pairs, address numbers with
leading zeros stripped, number x address token, number x name token). Per observed country partition, document
frequency over that partition's Source-1 and Source-2/3 records; keys with df > 2000 or present on one side only
are dropped; TF-IDF + L2; exact sparse cosine: top-20 targets per Source-1 record plus top-5 owners per target.

```
python -u scripts/run_r1_pair_retrieval.py dev  --all-queries --norm v2 --max-df 2000 --kf 20 --kr 5 --tag devall_v2_df2000_kf20_kr5
python -u scripts/run_r1_pair_retrieval.py test --norm v2 --max-df 2000 --kf 20 --kr 5 --tag test_v2_df2000_kf20_kr5
```

## 3. Pair features (`src/features_v3.py`)

```
python -u scripts/build_m003_features.py dev  --norm v2 --idf-scope country --cand-tag devall_v2_df2000_kf20_kr5 --out feat_dev_r3idf
python -u scripts/build_m003_features.py test --norm v2 --idf-scope country --cand-tag test_v2_df2000_kf20_kr5 --out feat_test_r3idf
```

51 features per pair: RapidFuzz name/address similarities, token and number overlaps, IDF-weighted overlaps and
unmatched IDF mass, number conflicts and first-number agreement, lengths, missing-address and script
indicators, retrieval ranks/score, and owner-competition features. `--idf-scope country` computes IDF inside
each country partition of the run being resolved with the ratio (n+1)/(df+1) capped at 2e5, so token IDF
depends on relative frequency, not corpus size.

## 4. Models (final configuration)

Stage 1 is trained on `matcher_train` only (all positives, retrieval-hard negatives, 25% seeded easy
negatives; entity-grouped early stopping). IDF-scale augmentation `--aug-idf half:0.85,1.15` multiplies the six
absolute IDF features of a random half of the training rows by a factor ~ U(0.85, 1.15) (training only;
inference unchanged), which reduces the models' dependence on corpus-dependent IDF scale while keeping
in-distribution accuracy. Stage 2 uses 2-fold entity-level cross-fitted stage-1 probabilities for
`matcher_train` rows and main-model probabilities elsewhere, probability-competition features
(`src/stage2.py`) and collective entity-support features (`src/stage2x.py`: confident candidates of the same
Source-1 entity from the same / other target source excluding the row, best competing probability within each,
and the row's share of the target's and of the entity's probability mass). Thresholds are selected on
`calibration_holdout` (macro F0.5 over the full population), reported on `dev_eval`.

```
python -u scripts/train_m003.py   --feat feat_dev_r3idf --out m003_r3mix --no-crossfit --aug-idf half:0.85,1.15
python -u scripts/train_stage2.py --feat feat_dev_r3idf --stage1 m003_r3mix --out m004_r3mix --aug-idf half:0.85,1.15
python -u scripts/stage2_lab.py   --feat feat_dev_r3idf --stage1 m003_r3mix --oof m004_r3mix --ref m004_r3mix --aug-idf half:0.85,1.15 --out s2x_r3mix
```

`train_stage2.py` produces the cross-fitted stage-1 probabilities (`m004_r3mix/oof_matcher_train.npy`) and a
reference stage-2 model without the collective extras (used only for the reported paired comparison, `--ref`);
`stage2_lab.py` trains the final stage-2 model with the collective features on those probabilities (selected
threshold 0.71, recorded in `s2x_r3mix/summary.json`; development F0.5 0.981677, calibration 0.981641).

## 5. Stage-2 test inference

```
python -u scripts/run_m003_test.py  --feat feat_test_r3idf --model m003_r3mix --decoder global --threshold 0.78 --run-name r3mix_stage1_test --no-export
python -u scripts/run_m004x_test.py --feat feat_test_r3idf --stage1-run r3mix_stage1_test --model s2x_r3mix --threshold 0.71 --run-name final
```

The stage-1 threshold only labels the intermediate scored table. 0.71 is the stage-2 threshold selected on
`calibration_holdout`.

This step writes the stage-2 probabilities of every candidate pair to
`<root>/experiments/m003/test_runs/final/scored.parquet`, used by step 6. It also writes a global-threshold
`output/` (62,870,547 candidate pairs, 5,786,001 matches, `matching_results.tsv` sha256
1b1c72a50f36e8301bdaacf75e44ced224d62dcd0ada1db406140c263dad641c), which step 6 replaces.

## 6. Final decoder: entity-adaptive thresholds

Each pair is accepted if its stage-2 probability p reaches a threshold that depends on two label-free
properties:
- the target's source;
- the number of candidates of the same Source-1 entity with p >= 0.90, capped at 2.

| target source | 0 confident | 1 confident | >= 2 confident |
|---|---|---|---|
| Source 2 | 0.44 | 0.73 | 0.76 |
| Source 3 | 0.47 | 0.67 | 0.78 |

The rule is applied to Source-1 entities whose country occurs in the training data, which is read from
`train_source1.tsv` (US, India). Entities of any other country (France in the test set) keep the global threshold
0.71, because the rule is only validated on labeled countries.

The rule was selected on `calibration_holdout` only, from 20 candidate rules fitted by exact coordinate ascent of
macro F0.5, and frozen before `dev_eval` was scored. The candidates condition on:
- the top candidate;
- the target source;
- the number of confident, predicted and retrieved candidates, and products of these;
- p1/p2 geometric mixing.

The frozen rule and its held-out evaluation are shipped in `artifacts/decoder/`:
- `frozen_rule.json`;
- `fit_result.json`: `dev_eval` 0.981851 vs 0.981677 with the global threshold (+0.000174, paired bootstrap 95% CI
  +0.000041 to +0.000308).

The fit step is optional: re-fitting reproduces the shipped rule and its evaluation exactly. The apply step produces
the final outputs:

```
python -u scripts/fit_adaptive_decoder.py --feat feat_dev_r3idf --stage1 m003_r3mix --stage2 s2x_r3mix --out decoder_c1
python -u scripts/apply_adaptive_decoder.py --test-run final
```

`apply_adaptive_decoder.py` reads `artifacts/decoder/frozen_rule.json` (use `--rule` to point at a re-fitted
rule). It writes the final outputs to `<root>/student_resource/output/`:
- 62,870,547 candidate pairs;
- 5,772,984 matches;
- 1,631,694 Source-1 entities with at least one match;
- `matching_results.tsv` sha256 49590acff1e2d83213c46dac38edc8acff4cca33be10f99832c648493365a90f;
- `candidate_pairs.tsv` sha256 7d571c1131c394550a0bea01a0ae265d4b10d3a14f27a833d0107ba2ef130f89.

`candidate_pairs.tsv` contains every scored (Source-1, target) pair, exactly the set fed to the matcher. The
decoder does not change it. `matching_results.tsv` contains the accepted subset. Validate:

```
cd <root>/student_resource
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Evaluated but not used in the final configuration (scripts kept for reference):
- full-row augmentation (`--aug-idf 0.85,1.15`);
- target-target corroboration features (`src/stage2c.py`, `scripts/stage2_lab_c.py --corr`,
  `scripts/run_m004c_test.py`, `scripts/build_target_strings.py`);
- exact-name twin features;
- rival/anchor corroboration (`--corr2 --no-own-corr`);
- probability blends (`scripts/blend_stage2.py`);
- applying the entity-adaptive decoder to countries absent from training.

## Compliance

All statistics (retrieval DF, IDF, collective features) are computed from the supplied records of the run being
resolved, without labels. Labels are used only for `matcher_train` training, threshold selection on
`calibration_holdout`, and reporting on `dev_eval`. The script→Latin alias map is learned from `matcher_train`
training links only.

The decoder's thresholds are fitted on `calibration_holdout` only. Its conditioning variables (target source, the
entity's count of confident candidates) are label-free and computed from the run being resolved. The list of
training countries comes from the training file.
