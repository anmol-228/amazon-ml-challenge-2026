"""Phase-3 candidate pruning + labeling (DuckDB, columnar, no Python row loop).

Frozen upstream pruning rule (measured in run_phase3_pruning_check.py; adopted
here specifically to bound expensive RapidFuzz feature computation to a
tractable row count under the submission-#1 deadline -- mission spec section
18): keep a B004 candidate row iff

    route_exact_name_address OR route_numeric_exact_set
    OR (forward_rank <= 10) OR (reverse_rank <= 3)

Measured effect (development, full 228,440,608-row B004 union): 151,644,764
rows retained (33.6% cut), link_recall 0.8616 (vs 0.8914 unpruned), oracle
Amazon macro F0.5 0.933854 (vs 0.950159 unpruned) -- see
experiments/blocking/PHASE_3/pruning_check_results.json. This rule is
IDENTICAL for development and test (frozen, no test-derived tuning); the
same fwd<=10/rev<=3/exact/numeric predicate is applied to test candidates in
run_phase3_test_pipeline.py.

This script additionally:
  - attaches `label` (True iff the pair is a real train_ground_truth match)
  - attaches `phase3_split` (matcher_train / dev_eval / calibration_holdout)
  - for `matcher_train` ONLY, subsamples negatives (keeps every positive,
    keeps a random fraction of negatives sized to hit a target negative:
    positive ratio) -- this is a training-efficiency measure, not a change
    to what candidate_pairs.tsv means for test (which always uses the full
    pruned set, never subsampled)
  - writes three parquet files under experiments/phase3/pruned/:
    matcher_train.parquet (subsampled), dev_eval.parquet (full pruned),
    calibration_holdout.parquet (full pruned)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_PATH = PROJECT_ROOT / "experiments" / "blocking" / "B004" / "union_candidates.parquet"
GT_PATH = PROJECT_ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"
OUT_DIR = PROJECT_ROOT / "experiments" / "phase3" / "pruned"

NEG_PER_POS_TARGET = 8.0
FWD_CAP = 10
REV_CAP = 3


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    con.execute(f"""
        CREATE VIEW gt_pairs AS
        SELECT source1_entity_id AS s1_entity_id, UNNEST(string_split(matched_entity_ids, ',')) AS target_entity_id
        FROM read_csv('{GT_PATH.as_posix()}', delim='\t', header=true, columns={{
            'source1_entity_id': 'VARCHAR', 'matched_entity_ids': 'VARCHAR'
        }})
        WHERE matched_entity_ids != ''
    """)

    con.execute(f"""
        CREATE VIEW split_map AS
        SELECT source1_entity_id, phase3_split
        FROM read_csv('{SPLIT_PATH.as_posix()}', delim='\t', header=true)
    """)

    where = (
        f"route_exact_name_address OR route_numeric_exact_set "
        f"OR (forward_rank IS NOT NULL AND forward_rank <= {FWD_CAP}) "
        f"OR (reverse_rank IS NOT NULL AND reverse_rank <= {REV_CAP})"
    )

    con.execute(f"""
        CREATE VIEW pruned_labeled AS
        SELECT
            c.*,
            (gt.target_entity_id IS NOT NULL) AS label,
            sm.phase3_split AS phase3_split
        FROM read_parquet('{B004_PATH.as_posix()}') c
        JOIN split_map sm ON sm.source1_entity_id = c.s1_entity_id
        LEFT JOIN gt_pairs gt
            ON gt.s1_entity_id = c.s1_entity_id AND gt.target_entity_id = c.target_entity_id
        WHERE {where}
    """)

    counts = {}
    for split_name in ("dev_eval", "calibration_holdout"):
        t1 = time.time()
        out_path = OUT_DIR / f"{split_name}.parquet"
        con.execute(f"""
            COPY (SELECT * FROM pruned_labeled WHERE phase3_split = '{split_name}')
            TO '{out_path.as_posix()}' (FORMAT PARQUET)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        n_pos = con.execute(f"SELECT SUM(label::INT) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        counts[split_name] = {"n_rows": n, "n_pos": n_pos}
        print(f"[prune] {split_name}: {n:,} rows ({n_pos:,} positive) written in {time.time()-t1:.1f}s")

    # matcher_train: keep all positives + subsampled negatives.
    t1 = time.time()
    n_pos_train = con.execute(
        "SELECT COUNT(*) FROM pruned_labeled WHERE phase3_split = 'matcher_train' AND label"
    ).fetchone()[0]
    n_neg_train = con.execute(
        "SELECT COUNT(*) FROM pruned_labeled WHERE phase3_split = 'matcher_train' AND NOT label"
    ).fetchone()[0]
    neg_fraction = min(1.0, (n_pos_train * NEG_PER_POS_TARGET) / max(1, n_neg_train))
    print(f"[prune] matcher_train: {n_pos_train:,} positives, {n_neg_train:,} negatives available, "
          f"sampling negatives at fraction {neg_fraction:.4f}")

    out_path = OUT_DIR / "matcher_train.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM pruned_labeled
            WHERE phase3_split = 'matcher_train'
              AND (label OR random() < {neg_fraction})
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_pos = con.execute(f"SELECT SUM(label::INT) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    counts["matcher_train"] = {"n_rows": n, "n_pos": n_pos, "neg_fraction_sampled": neg_fraction}
    print(f"[prune] matcher_train: {n:,} rows ({n_pos:,} positive) written in {time.time()-t1:.1f}s")

    meta_path = PROJECT_ROOT / "experiments" / "phase3" / "pruning_labeling_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fwd_cap": FWD_CAP,
                "rev_cap": REV_CAP,
                "neg_per_pos_target": NEG_PER_POS_TARGET,
                "counts": counts,
            },
            fh,
            indent=2,
        )
    print(f"[prune] done, total wall time {time.time()-t0:.1f}s, metadata -> {meta_path}")


if __name__ == "__main__":
    main()
