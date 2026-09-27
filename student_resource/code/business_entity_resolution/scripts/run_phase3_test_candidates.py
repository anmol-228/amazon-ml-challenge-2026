"""Phase-3 TEST candidate generation -- frozen architecture, no test adaptation.

Reuses the exact B004 four-route architecture (DEC-027) and the exact
upstream pruning rule frozen in run_phase3_prune_label.py, applied to the
TEST population instead of `development`. No fitting happens here: the word
vectorizer is the one already fit on TRAIN data only
(experiments/blocking/cache/word_vectorizer.pkl, DEC-026) and is only ever
`.transform()`-ed on test text, never refit. No test-derived IDF, no
test-derived frequency thresholds, no France-specific handling -- the same
country-agnostic code path that ran on {US, India} runs unchanged on
{US, India, France}.

Routes (identical mechanism to development, see blocking_routes.py /
blocking_exact.py / blocking_numeric.py, all reused unchanged):
  1. exact(name, address) -- route_exact_name_address
  2. forward word-TF-IDF top-k=50 (S1 -> target)
  3. reverse word-TF-IDF top-r=5 (target -> S1)
  4. numeric-exact-signature, max_group_size=650 (the frozen production
     cutoff measured in EXP-B003/DEC-027; not re-measured here since
     re-measuring on test data would itself be a test-derived statistic)

Then the same frozen pruning predicate as run_phase3_prune_label.py:
    route_exact_name_address OR route_numeric_exact_set
    OR forward_rank <= 10 OR reverse_rank <= 3

Output: experiments/phase3/test/test_candidates_pruned.parquet (identity
(s1_entity_id, target_source, target_entity_id) + route/rank/score columns).
This file, once scored by the frozen matcher, becomes candidate_pairs.tsv.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_exact import exact_candidates  # noqa: E402
from blocking_normalization import joint_text  # noqa: E402
from blocking_numeric import exact_numeric_signature_candidates  # noqa: E402
from blocking_routes import forward_route, reverse_route  # noqa: E402
from blocking_union import union_candidates_partitioned  # noqa: E402
from data_io import TEST_PATHS, load_source_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"
OUT_DIR = PROJECT_ROOT / "experiments" / "phase3" / "test"
ROUTES_TMP = OUT_DIR / "routes_tmp"

FORWARD_K = 50
REVERSE_R = 5
NUMERIC_MAX_GROUP_SIZE = 650
FWD_CAP = 10
REV_CAP = 3
FORWARD_BATCH_SIZE = 1500
REVERSE_BATCH_SIZE = 1500


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROUTES_TMP.mkdir(parents=True, exist_ok=True)

    print("[test-candidates] loading test source tables...")
    s1 = load_source_table(TEST_PATHS["source1"])
    s2 = load_source_table(TEST_PATHS["source2"])
    s3 = load_source_table(TEST_PATHS["source3"])
    print(f"[test-candidates] test S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} "
          f"in {time.time()-t0:.1f}s")

    s1_ids = s1["entity_id"].tolist()
    s1_names = s1["business_name"].tolist()
    s1_addrs = s1["business_address"].tolist()

    target_df = pd.concat([s2, s3], axis=0, ignore_index=True)
    target_ids = target_df["entity_id"].tolist()
    target_names = target_df["business_name"].tolist()
    target_addrs = target_df["business_address"].tolist()

    # ---- route 1: exact -------------------------------------------------
    t1 = time.time()
    exact_df = exact_candidates(s1_ids, s1_names, s1_addrs, target_ids, target_names, target_addrs)
    exact_df = exact_df[exact_df["route_exact_name_address"]][
        ["s1_entity_id", "target_entity_id", "route_exact_name_address"]
    ]
    exact_df.to_parquet(ROUTES_TMP / "exact.parquet", index=False)
    print(f"[test-candidates] exact route: {len(exact_df):,} rows in {time.time()-t1:.1f}s")
    del exact_df

    # ---- route 4: numeric exact signature --------------------------------
    t1 = time.time()
    numeric_df = exact_numeric_signature_candidates(
        s1_ids, s1_addrs, target_ids, target_addrs, max_group_size=NUMERIC_MAX_GROUP_SIZE
    )
    numeric_df["route_numeric_exact_set"] = True
    numeric_df.to_parquet(ROUTES_TMP / "numeric.parquet", index=False)
    print(f"[test-candidates] numeric route: {len(numeric_df):,} rows in {time.time()-t1:.1f}s")
    del numeric_df

    # ---- routes 2/3: forward/reverse word TF-IDF (frozen train vectorizer) ---
    t1 = time.time()
    # Pickle load is safe here: word_vectorizer.pkl is produced by this same
    # local pipeline (scripts/run_b000_preflight.py), never an externally
    # supplied or downloaded file (same pattern as run_b001_b001r_word.py).
    with open(CACHE_DIR / "word_vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)
    s1_texts = [joint_text(n, a) for n, a in zip(s1_names, s1_addrs)]
    target_texts = [joint_text(n, a) for n, a in zip(target_names, target_addrs)]
    s1_matrix = vectorizer.transform(s1_texts).tocsr()
    target_matrix = vectorizer.transform(target_texts).tocsr()
    print(f"[test-candidates] transformed test text with frozen train vectorizer "
          f"in {time.time()-t1:.1f}s (s1={s1_matrix.shape}, target={target_matrix.shape})")

    t1 = time.time()
    fwd_df = forward_route(s1_ids, s1_matrix, target_ids, target_matrix, k=FORWARD_K, batch_size=FORWARD_BATCH_SIZE)
    fwd_df.to_parquet(ROUTES_TMP / "forward.parquet", index=False)
    print(f"[test-candidates] forward route: {len(fwd_df):,} rows in {time.time()-t1:.1f}s")
    del fwd_df

    t1 = time.time()
    rev_df = reverse_route(s1_ids, s1_matrix, target_ids, target_matrix, r=REVERSE_R, batch_size=REVERSE_BATCH_SIZE)
    rev_df.to_parquet(ROUTES_TMP / "reverse.parquet", index=False)
    print(f"[test-candidates] reverse route: {len(rev_df):,} rows in {time.time()-t1:.1f}s")
    del rev_df, s1_matrix, target_matrix, vectorizer

    # ---- union (memory-safe, partitioned) --------------------------------
    t1 = time.time()
    union_path = OUT_DIR / "test_candidates_union.parquet"
    diag = union_candidates_partitioned(
        [ROUTES_TMP / "exact.parquet", ROUTES_TMP / "forward.parquet",
         ROUTES_TMP / "reverse.parquet", ROUTES_TMP / "numeric.parquet"],
        union_path,
        n_buckets=128,
        tmp_dir=OUT_DIR / "union_buckets_tmp",
    )
    print(f"[test-candidates] union done in {time.time()-t1:.1f}s: {diag}")

    # ---- frozen pruning (identical predicate to development) -------------
    t1 = time.time()
    con = duckdb.connect()
    where = (
        f"route_exact_name_address OR route_numeric_exact_set "
        f"OR (forward_rank IS NOT NULL AND forward_rank <= {FWD_CAP}) "
        f"OR (reverse_rank IS NOT NULL AND reverse_rank <= {REV_CAP})"
    )
    pruned_path = OUT_DIR / "test_candidates_pruned.parquet"
    con.execute(f"""
        COPY (SELECT * FROM read_parquet('{union_path.as_posix()}') WHERE {where})
        TO '{pruned_path.as_posix()}' (FORMAT PARQUET)
    """)
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{union_path.as_posix()}')").fetchone()[0]
    n_pruned = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pruned_path.as_posix()}')").fetchone()[0]
    n_s1_with_candidates = con.execute(
        f"SELECT COUNT(DISTINCT s1_entity_id) FROM read_parquet('{pruned_path.as_posix()}')"
    ).fetchone()[0]
    print(f"[test-candidates] pruning: {n_total:,} -> {n_pruned:,} rows "
          f"({n_s1_with_candidates:,}/{len(s1_ids):,} S1 entities have >=1 candidate) "
          f"in {time.time()-t1:.1f}s")

    print(f"[test-candidates] TOTAL wall time {time.time()-t0:.1f}s -> {pruned_path}")


if __name__ == "__main__":
    main()
