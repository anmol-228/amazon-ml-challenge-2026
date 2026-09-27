"""Phase-3 v2 candidate labeling for Submission #2 (post-M001 upgrade
mission): (a) relaxes upstream pruning to P3 = the full, unpruned B004 union
(DEC-027's own architecture, oracle macro F0.5 0.950159 -- see
run_phase3_pruning_check2.py for the measured P0/P1/P2/P3 grid that
motivated this choice: P3 recovers the full available candidate ceiling for
only ~15-20 extra minutes of scoring time versus P2, well within the
09:00 IST budget), and (b) replaces `run_phase3_prune_label.py`'s uniform
random negative sampling for `matcher_train` with a two-tier hard/easy
mixture (EXP-M002), all computed from cheap, already-materialized B004
columns (route flags, forward/reverse rank+score) -- no RapidFuzz needed at
this stage, consistent with the two-stage cheap-then-expensive feature
architecture (mission spec section 8).

Hard-negative tier definition (mission spec section 4's suggested evidence,
restricted to what's available before RapidFuzz features exist): a false
candidate (label=0) is "hard" if it has strong plausible-match evidence --
`forward_score >= 0.5 OR reverse_score >= 0.5 OR route_exact_name_address OR
route_numeric_exact_set OR (>=2 routes support it)`. `dev_eval` and
`calibration_holdout` are NOT subsampled at all (every pruned -- here,
unpruned -- candidate row is kept, matching real inference-time scoring).

Note: candidate_pairs.tsv semantics are untouched by this script -- it only
prepares TRAINING data. The actual test-side candidate set (candidate_pairs.tsv)
is controlled separately by run_phase3_test_candidates.py's own pruning
predicate, which must be updated to match (no pruning / full union) for
consistency before test scoring.
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
OUT_DIR = PROJECT_ROOT / "experiments" / "phase3" / "pruned_v2"

HARD_NEG_PER_POS_TARGET = 10.0
EASY_NEG_PER_POS_TARGET = 5.0
# COALESCE is load-bearing here, not cosmetic: forward_score/reverse_score are
# NULL whenever that specific route didn't fire for a row. In SQL's
# three-valued logic, `NULL >= 0.5` evaluates to NULL (not FALSE), which would
# make the whole OR-chain NULL (not FALSE) for many genuinely-easy rows --
# excluding them from BOTH the hard tier (`WHERE ... AND HARD_CONDITION`,
# which only keeps TRUE) and the easy tier (`WHERE ... AND NOT HARD_CONDITION`,
# since NOT NULL = NULL, also excluded). Coalescing to 0 first forces a
# genuine two-valued TRUE/FALSE classification.
HARD_CONDITION = (
    "(COALESCE(forward_score, 0) >= 0.5 OR COALESCE(reverse_score, 0) >= 0.5 "
    "OR route_exact_name_address OR route_numeric_exact_set OR "
    "((route_exact_name_address::INT + route_forward_word::INT + route_reverse_word::INT "
    "+ route_numeric_exact_set::INT) >= 2))"
)


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
    con.execute(f"""
        CREATE VIEW labeled AS
        SELECT c.*, (gt.target_entity_id IS NOT NULL) AS label, sm.phase3_split AS phase3_split
        FROM read_parquet('{B004_PATH.as_posix()}') c
        JOIN split_map sm ON sm.source1_entity_id = c.s1_entity_id
        LEFT JOIN gt_pairs gt ON gt.s1_entity_id = c.s1_entity_id AND gt.target_entity_id = c.target_entity_id
    """)

    counts = {}
    for split_name in ("dev_eval", "calibration_holdout"):
        t1 = time.time()
        out_path = OUT_DIR / f"{split_name}.parquet"
        con.execute(f"""
            COPY (SELECT * FROM labeled WHERE phase3_split = '{split_name}')
            TO '{out_path.as_posix()}' (FORMAT PARQUET)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        n_pos = con.execute(f"SELECT SUM(label::INT) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        counts[split_name] = {"n_rows": n, "n_pos": n_pos}
        print(f"[prune_v2] {split_name} (FULL, unpruned): {n:,} rows ({n_pos:,} positive) "
              f"in {time.time()-t1:.1f}s")

    # ---- matcher_train: all positives + tiered hard/easy negatives --------
    t1 = time.time()
    n_pos_train = con.execute(
        "SELECT COUNT(*) FROM labeled WHERE phase3_split = 'matcher_train' AND label"
    ).fetchone()[0]
    n_hard_available = con.execute(
        f"SELECT COUNT(*) FROM labeled WHERE phase3_split = 'matcher_train' AND NOT label AND {HARD_CONDITION}"
    ).fetchone()[0]
    n_easy_available = con.execute(
        f"SELECT COUNT(*) FROM labeled WHERE phase3_split = 'matcher_train' AND NOT label AND NOT {HARD_CONDITION}"
    ).fetchone()[0]

    hard_fraction = min(1.0, (n_pos_train * HARD_NEG_PER_POS_TARGET) / max(1, n_hard_available))
    easy_fraction = min(1.0, (n_pos_train * EASY_NEG_PER_POS_TARGET) / max(1, n_easy_available))
    print(f"[prune_v2] matcher_train: {n_pos_train:,} positives, "
          f"{n_hard_available:,} hard negatives available (sampling fraction {hard_fraction:.4f}), "
          f"{n_easy_available:,} easy negatives available (sampling fraction {easy_fraction:.4f})")

    out_path = OUT_DIR / "matcher_train.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM labeled
            WHERE phase3_split = 'matcher_train'
              AND (
                  label
                  OR (({HARD_CONDITION}) AND random() < {hard_fraction})
                  OR ((NOT ({HARD_CONDITION})) AND random() < {easy_fraction})
              )
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_pos = con.execute(f"SELECT SUM(label::INT) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_hard_sampled = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}') WHERE NOT label AND {HARD_CONDITION}"
    ).fetchone()[0]
    n_easy_sampled = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}') WHERE NOT label AND NOT ({HARD_CONDITION})"
    ).fetchone()[0]
    counts["matcher_train"] = {
        "n_rows": n, "n_pos": n_pos, "n_hard_neg": n_hard_sampled, "n_easy_neg": n_easy_sampled,
        "hard_fraction_sampled": hard_fraction, "easy_fraction_sampled": easy_fraction,
    }
    print(f"[prune_v2] matcher_train: {n:,} rows ({n_pos:,} positive, {n_hard_sampled:,} hard-neg, "
          f"{n_easy_sampled:,} easy-neg) in {time.time()-t1:.1f}s")

    meta_path = PROJECT_ROOT / "experiments" / "phase3" / "pruning_labeling_metadata_v2.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "pruning": "NONE (full B004 union, P3)",
            "hard_condition_sql": HARD_CONDITION,
            "hard_neg_per_pos_target": HARD_NEG_PER_POS_TARGET,
            "easy_neg_per_pos_target": EASY_NEG_PER_POS_TARGET,
            "counts": counts,
        }, fh, indent=2)
    print(f"[prune_v2] done, total wall time {time.time()-t0:.1f}s, metadata -> {meta_path}")


if __name__ == "__main__":
    main()
