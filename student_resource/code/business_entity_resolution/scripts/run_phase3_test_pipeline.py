"""Phase-3 TEST inference: frozen features -> frozen matcher -> frozen
threshold -> output/matching_results.tsv + output/candidate_pairs.tsv.

Reuses run_phase3_features.compute_batch_features (the identical code path
used for development features) against TEST-side lookup tables (built from
test_source1/2/3.tsv, never train data) for name/address/country/token
attributes. `target_name_rarity` is the one feature that could otherwise be
computed from test's own population; per the leakage-safety repair in
run_phase3_features.py, it instead uses `train_name_rarity_counts()` --
frozen name-collision counts from TRAIN Source-2/3 only, looked up by
normalized name string -- so no statistic in this script is derived from
test's own frequency distribution. The word-TF-IDF vectors folded into
`test_candidates_pruned.parquet`'s forward/reverse routes were produced by
run_phase3_test_candidates.py using the FROZEN train-fit vectorizer only.

No threshold, feature, or candidate-generation choice in this script is
tuned against test data: `MODEL_DIR/selected_threshold.json` and
`MODEL_DIR/feature_columns.json` are read verbatim from
run_phase3_train_model.py's frozen output.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_io import TEST_PATHS, load_source_table  # noqa: E402
from run_phase3_features import build_lookup_tables, compute_batch_features, train_name_rarity_counts  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DIR = PROJECT_ROOT / "experiments" / "phase3" / "test"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model"
OUTPUT_DIR = PROJECT_ROOT / "student_resource" / "output"

BATCH_ROWS = 2_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[test-pipeline] loading test source tables for lookup + feature computation...")
    s1 = load_source_table(TEST_PATHS["source1"])
    s2 = load_source_table(TEST_PATHS["source2"])
    s3 = load_source_table(TEST_PATHS["source3"])
    all_test_s1_ids = s1["entity_id"].tolist()

    s1_attrs, target_attrs = build_lookup_tables(s1, s2, s3)
    name_rarity_counts = train_name_rarity_counts()

    with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as fh:
        feature_columns = json.load(fh)
    with open(MODEL_DIR / "selected_threshold.json", encoding="utf-8") as fh:
        threshold = json.load(fh)["threshold"]
    print(f"[test-pipeline] frozen threshold = {threshold}, {len(feature_columns)} features")

    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))

    pruned_path = TEST_DIR / "test_candidates_pruned.parquet"
    pf = pq.ParquetFile(pruned_path)
    total_rows = pf.metadata.num_rows
    print(f"[test-pipeline] scoring {total_rows:,} pruned test candidate rows...")

    # accumulate accepted (s1, target) pairs and full candidate (s1, target)
    # pairs for candidate_pairs.tsv, keyed by s1 -> set of target ids.
    accepted: dict[str, list[str]] = {}
    all_candidates: dict[str, list[str]] = {}

    processed = 0
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS), start=1):
        t_b = time.time()
        df = batch.to_pandas()
        feats = compute_batch_features(df, s1_attrs, target_attrs, name_rarity_counts)
        for col in feature_columns:
            if feats[col].dtype == bool:
                feats[col] = feats[col].astype(np.int8)
        scores = booster.predict(feats[feature_columns])

        for s1, tid in zip(feats["s1_entity_id"].to_numpy(), feats["target_entity_id"].to_numpy()):
            all_candidates.setdefault(s1, []).append(tid)
        accept_mask = scores >= threshold
        for s1, tid in zip(
            feats.loc[accept_mask, "s1_entity_id"].to_numpy(),
            feats.loc[accept_mask, "target_entity_id"].to_numpy(),
        ):
            accepted.setdefault(s1, []).append(tid)

        processed += len(df)
        print(f"[test-pipeline] batch {batch_i}: {len(df):,} rows in {time.time()-t_b:.1f}s "
              f"({processed:,}/{total_rows:,} = {100*processed/max(1,total_rows):.1f}%) "
              f"elapsed {time.time()-t0:.1f}s")

    print(f"[test-pipeline] scoring done in {time.time()-t0:.1f}s. "
          f"{len(accepted):,} S1 entities have >=1 accepted match out of {len(all_test_s1_ids):,} total.")

    # ---- matching_results.tsv: every test S1 exactly once -----------------
    # Written to a .tmp file first and atomically replaced at the end, so a
    # mid-write interruption (e.g. session teardown) can never leave the
    # canonical output truncated or corrupted -- the old file (or none)
    # remains until the new one is fully flushed and renamed into place.
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    matching_tmp = OUTPUT_DIR / "matching_results.tsv.tmp"
    with open(matching_tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_test_s1_ids:
            ids = sorted(set(accepted.get(s1, ())))
            fh.write(f"{s1}\t{','.join(ids)}\n")
    matching_tmp.replace(matching_path)
    print(f"[test-pipeline] wrote {matching_path}")

    # ---- candidate_pairs.tsv: every test S1 exactly once -------------------
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    candidate_tmp = OUTPUT_DIR / "candidate_pairs.tsv.tmp"
    with open(candidate_tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_test_s1_ids:
            ids = sorted(set(all_candidates.get(s1, ())))
            fh.write(f"{s1}\t{','.join(ids)}\n")
    candidate_tmp.replace(candidate_path)
    print(f"[test-pipeline] wrote {candidate_path}")

    n_predicted_nonempty = sum(1 for s1 in all_test_s1_ids if accepted.get(s1))
    summary = {
        "n_test_s1": len(all_test_s1_ids),
        "n_s1_with_candidates": len(all_candidates),
        "n_s1_with_predicted_match": n_predicted_nonempty,
        "threshold_used": threshold,
        "matching_results_sha256": sha256_file(matching_path),
        "candidate_pairs_sha256": sha256_file(candidate_path),
        "wall_time_s": time.time() - t0,
    }
    with open(TEST_DIR / "test_pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[test-pipeline] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
