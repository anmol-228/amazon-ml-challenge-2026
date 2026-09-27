"""Phase-3 v2 TEST inference: P3 (unpruned, full B004-equivalent) candidate
architecture + EXP-M002 hard-negative-trained matcher (model_v2) + its own
calibrated threshold. Reuses the already-computed, already-deduplicated full
test candidate union from run_phase3_test_candidates.py
(experiments/phase3/test/test_candidates_union.parquet, 199,702,568 rows)
directly -- no re-pruning, since v2's whole point is to score the full
candidate set. No new retrieval, no refitting: identical word vectorizer,
identical routes, only the downstream pruning/model/threshold changed.
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
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model_v2"
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

    print("[test-pipeline-v2] loading test source tables...")
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
    print(f"[test-pipeline-v2] frozen threshold = {threshold}, {len(feature_columns)} features")

    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))

    union_path = TEST_DIR / "test_candidates_union.parquet"
    pf = pq.ParquetFile(union_path)
    total_rows = pf.metadata.num_rows
    print(f"[test-pipeline-v2] scoring {total_rows:,} FULL (unpruned) test candidate rows...")

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

        for s1id, tid in zip(feats["s1_entity_id"].to_numpy(), feats["target_entity_id"].to_numpy()):
            all_candidates.setdefault(s1id, []).append(tid)
        accept_mask = scores >= threshold
        for s1id, tid in zip(
            feats.loc[accept_mask, "s1_entity_id"].to_numpy(),
            feats.loc[accept_mask, "target_entity_id"].to_numpy(),
        ):
            accepted.setdefault(s1id, []).append(tid)

        processed += len(df)
        print(f"[test-pipeline-v2] batch {batch_i}: {len(df):,} rows in {time.time()-t_b:.1f}s "
              f"({processed:,}/{total_rows:,} = {100*processed/max(1,total_rows):.1f}%) "
              f"elapsed {time.time()-t0:.1f}s")

    print(f"[test-pipeline-v2] scoring done in {time.time()-t0:.1f}s. "
          f"{len(accepted):,} S1 entities have >=1 accepted match out of {len(all_test_s1_ids):,} total.")

    matching_path = OUTPUT_DIR / "matching_results.tsv"
    with open(matching_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1id in all_test_s1_ids:
            ids = sorted(set(accepted.get(s1id, ())))
            fh.write(f"{s1id}\t{','.join(ids)}\n")
    print(f"[test-pipeline-v2] wrote {matching_path}")

    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    with open(candidate_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1id in all_test_s1_ids:
            ids = sorted(set(all_candidates.get(s1id, ())))
            fh.write(f"{s1id}\t{','.join(ids)}\n")
    print(f"[test-pipeline-v2] wrote {candidate_path}")

    n_predicted_nonempty = sum(1 for s1id in all_test_s1_ids if accepted.get(s1id))
    summary = {
        "n_test_s1": len(all_test_s1_ids),
        "n_s1_with_candidates": len(all_candidates),
        "n_s1_with_predicted_match": n_predicted_nonempty,
        "threshold_used": threshold,
        "matching_results_sha256": sha256_file(matching_path),
        "candidate_pairs_sha256": sha256_file(candidate_path),
        "wall_time_s": time.time() - t0,
    }
    with open(TEST_DIR / "test_pipeline_v2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[test-pipeline-v2] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
