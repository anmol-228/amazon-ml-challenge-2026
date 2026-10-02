# Reproducibility

The authoritative, step-by-step reproduction guide is
[`student_resource/code/business_entity_resolution/README.md`](../student_resource/code/business_entity_resolution/README.md) —
it gives the exact command for every pipeline stage, in order, with the parameters and
seeds used for the final submission. This document is a shorter supplement: a
dependency-graph overview, the data/environment prerequisites, and two caveats the code
README doesn't cover.

## Data placement

Obtain the official Amazon ML Challenge 2026 dataset yourself (not redistributed here) and
place it at:

```
student_resource/dataset/train/...
student_resource/dataset/test/...
```

## Environment

```
py -3.12 -m venv .venv
.venv/Scripts/pip install -r student_resource/code/business_entity_resolution/requirements.txt
```

## Dependency graph (final submission → raw data)

```
matching_results.tsv (public macro F0.5 = 0.973462)
  -> scripts/apply_adaptive_decoder.py --test-run final          [code: GITHUB]
       reads: experiments/m003/test_runs/final/scored.parquet    [regenerate below]
              artifacts/decoder/frozen_rule.json                 [GITHUB: shipped, fitted by fit_adaptive_decoder.py]
              raw test data (student_resource/dataset/test/)     [ORIGINAL_CHALLENGE_DATA]

  experiments/m003/test_runs/final/scored.parquet
    -> scripts/run_m003_test.py  --feat feat_test_r3idf --model m003_r3mix --decoder global --threshold 0.78 --run-name r3mix_stage1_test --no-export
       scripts/run_m004x_test.py --feat feat_test_r3idf --stage1-run r3mix_stage1_test --model s2x_r3mix --threshold 0.71 --run-name final

  experiments/m003/{m003_r3mix,m004_r3mix,s2x_r3mix}/  (trained models — kept, see caveat below)
    -> scripts/train_m003.py   --feat feat_dev_r3idf --out m003_r3mix --no-crossfit --aug-idf half:0.85,1.15
       scripts/train_stage2.py --feat feat_dev_r3idf --stage1 m003_r3mix --out m004_r3mix --aug-idf half:0.85,1.15
       scripts/stage2_lab.py   --feat feat_dev_r3idf --stage1 m003_r3mix --oof m004_r3mix --ref m004_r3mix --aug-idf half:0.85,1.15 --out s2x_r3mix

  experiments/m003/feat_{dev,test}_r3idf/
    -> scripts/build_m003_features.py {dev|test} --norm v2 --idf-scope country --cand-tag {devall_v2_df2000_kf20_kr5|test_v2_df2000_kf20_kr5} --out feat_{dev|test}_r3idf

  experiments/r1/{devall_v2_df2000_kf20_kr5,test_v2_df2000_kf20_kr5}/
    -> scripts/run_r1_pair_retrieval.py {dev --all-queries|test} --norm v2 --max-df 2000 --kf 20 --kr 5 --tag ...
       (deterministic — src/pair_retrieval.py contains no randomness)

  src/alias_map_matcher_train.json
    -> scripts/build_alias_map.py (deterministic, sort_keys=True; from matcher_train ground truth only;
         hash-verified byte-identical on rebuild: sha256 c31a468de9e192a29ed66dad0b707d8ecde41cb6ef760f4b18afe3b218b888d9)

  artifacts/splits/{validation_v1,phase3_split_v1}.tsv
    -> shipped directly (SHA-256 in artifacts/splits/SHA256SUMS.txt); phase3_split_v1.tsv is
       also deterministically rebuildable from validation_v1.tsv via scripts/build_phase3_split.py (seed 42).
```

## Stage-2 country-slice dependency

From `student_resource/code/business_entity_resolution/`, after placing the
original data and before the Stage-2 training commands, run:

```
python -u scripts/build_judge_slices.py
```

The tracked producer reads only:

- shipped `artifacts/splits/phase3_split_v1.tsv` (`dev_eval` membership);
- original `student_resource/dataset/train/train_source1.tsv` (`entity_id`, `country`).

It writes `<root>/experiments/judge/slices.parquet`, with string columns
`s1_entity_id`, `country`, in frozen split order. This is the deterministic
chain: original Source-1 data + shipped split → country slices →
`stage2_lab.py` held-out per-country comparison. Its `--ref m004_r3mix`
comparison table is produced by the preceding `train_stage2.py` command.
No additional unpublished input is needed for this comparison.

The retained country-slice artifact was verified against these inputs: all
264,819 entity IDs and country values match. Its historical row order differs
from frozen split order; consumers explicitly reindex by entity ID, so this
is semantically identical. Parquet byte identity with the historical file is
not promised. This does not strengthen the LightGBM determinism guarantee.
The producer refuses to overwrite an existing file; `--output <path>` permits
verification in a separate location without changing retained evidence.

## Determinism caveat (not covered by the code README)

`train_m003.py`/`train_stage2.py` fix `random_state`/seed values but run LightGBM with
`n_jobs=-1` (multi-threaded) and do not set `deterministic=True`. LightGBM's own
documentation notes that multi-threaded histogram-building reduction order is not
guaranteed bit-identical across runs/machines without that flag. Re-running the commands
above reliably reproduces the reported macro-F0.5 scores and model behavior to several
decimal places, but is not proven to yield byte-identical `lgbm_model.txt` /
`stage2_model.txt` files. For this reason the trained model files themselves
(`experiments/m003/{m003_r3mix,m004_r3mix,s2x_r3mix}/`) are kept as committed local
artifacts rather than deleted, even though the training code is fully documented.

## Verified final artifact

- `experiments/final_submission/out_seen/matching_results.tsv`
  SHA-256 `49590acff1e2d83213c46dac38edc8acff4cca33be10f99832c648493365a90f`
  Public leaderboard macro F0.5: **0.973462**. This is byte-identical to what
  `scripts/apply_adaptive_decoder.py` writes to `student_resource/output/` per its own
  documented output hash in the code README.
- Candidate pairs are identical to `submissions/STAGED_submission_011/candidate_pairs.tsv`
  (SHA-256 `7d571c1131c394550a0bea01a0ae265d4b10d3a14f27a833d0107ba2ef130f89`) — the test-time
  candidate set has been unchanged since submission #5.
- Immutable fallback: `submissions/STAGED_submission_011/matching_results.tsv`, public score 0.973142.
