"""Phase-3 v2 TEST inference, sharded / resumable production variant of
run_phase3_test_pipeline_v2.py.

Scoring semantics are identical to run_phase3_test_pipeline_v2.py:
  - candidates: the full, unpruned test union
    experiments/phase3/test/test_candidates_union.parquet (every row scored)
  - features: run_phase3_features.compute_batch_features (same function used
    to build the model_v2 training/dev/calibration features), with the frozen
    train-only target_name_rarity lookup
  - model: experiments/phase3/model_v2/lgbm_model.txt, feature order from
    model_v2/feature_columns.json, bool features cast to int8 as in training
  - decision: accept iff score >= model_v2/selected_threshold.json threshold
  - candidate_pairs.tsv: every scored (s1, target) pair, i.e. the exact set fed
    to the matcher; matching_results.tsv: the accepted subset.

Differences are operational only:
  1. Each group of parquet row groups is scored into its own shard file
     (s1_entity_id, target_entity_id, score + the model features), written
     atomically (tmp + rename). Existing shards are skipped on restart, so a
     killed run resumes from the last completed shard.
  2. No per-pair Python dicts are accumulated in memory; the two output TSVs are
     produced from the shards by DuckDB and written to temporary files that are
     atomically renamed into student_resource/output/.
  3. A status JSON is updated after every shard.

Usage:
  python -u run_phase3_test_pipeline_v2_sharded.py score      # score shards (resumable)
  python -u run_phase3_test_pipeline_v2_sharded.py finalize   # build output TSVs
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_io import TEST_PATHS, load_source_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_DIR = PROJECT_ROOT / "experiments" / "phase3" / "test"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model_v2"
RUN_DIR = PROJECT_ROOT / "experiments" / "phase3" / "test_v2"
SHARD_DIR = RUN_DIR / "shards"
STATUS_PATH = RUN_DIR / "status.json"
OUTPUT_DIR = PROJECT_ROOT / "student_resource" / "output"
UNION_PATH = TEST_DIR / "test_candidates_union.parquet"

ROW_GROUPS_PER_SHARD = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_status(**kw) -> None:
    kw["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    kw["pid"] = os.getpid()
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kw, fh, indent=2)
    os.replace(tmp, STATUS_PATH)


def load_model_config():
    with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as fh:
        feature_columns = json.load(fh)
    with open(MODEL_DIR / "selected_threshold.json", encoding="utf-8") as fh:
        threshold = json.load(fh)["threshold"]
    return feature_columns, threshold


def run_config() -> dict:
    """Everything that determines shard contents. Shards are only reused when
    this fingerprint is identical to the one recorded when they were written."""
    import inspect

    import run_phase3_features

    feature_columns, threshold = load_model_config()
    model_path = MODEL_DIR / "lgbm_model.txt"
    pf = pq.ParquetFile(UNION_PATH)
    return {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "feature_columns": feature_columns,
        "threshold": threshold,
        "union_path": str(UNION_PATH),
        "union_num_rows": pf.metadata.num_rows,
        "union_num_row_groups": pf.metadata.num_row_groups,
        "union_row_group_rows": [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)],
        "union_file_bytes": UNION_PATH.stat().st_size,
        "row_groups_per_shard": ROW_GROUPS_PER_SHARD,
        "feature_code_sha256": hashlib.sha256(inspect.getsource(run_phase3_features).encode("utf-8")).hexdigest(),
    }


def score() -> None:
    import lightgbm as lgb
    from run_phase3_features import build_lookup_tables, compute_batch_features, train_name_rarity_counts

    t0 = time.time()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    config = run_config()
    feature_columns, threshold = config["feature_columns"], config["threshold"]
    model_path = MODEL_DIR / "lgbm_model.txt"
    config_path = RUN_DIR / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            previous = json.load(fh)
        if previous != config:
            raise SystemExit(f"{config_path} differs from the current configuration; existing shards were "
                             f"produced by a different model/candidate/feature setup. Use a fresh RUN_DIR.")
    elif any(SHARD_DIR.glob("shard_*.parquet")):
        raise SystemExit(f"shards exist in {SHARD_DIR} without a config.json; refusing to reuse them.")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    pf = pq.ParquetFile(UNION_PATH)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    shards = [list(range(i, min(i + ROW_GROUPS_PER_SHARD, n_rg))) for i in range(0, n_rg, ROW_GROUPS_PER_SHARD)]
    todo = [(k, rgs) for k, rgs in enumerate(shards) if not (SHARD_DIR / f"shard_{k:04d}.parquet").exists()]
    print(f"[v2-sharded] {total_rows:,} rows, {n_rg} row groups, {len(shards)} shards, {len(todo)} to do", flush=True)
    if not todo:
        write_status(stage="score_done", shards_done=len(shards), shards_total=len(shards))
        return

    print("[v2-sharded] loading test source tables + lookup tables...", flush=True)
    s1 = load_source_table(TEST_PATHS["source1"])
    s2 = load_source_table(TEST_PATHS["source2"])
    s3 = load_source_table(TEST_PATHS["source3"])
    s1_attrs, target_attrs = build_lookup_tables(s1, s2, s3)
    del s1, s2, s3
    name_rarity_counts = train_name_rarity_counts()
    booster = lgb.Booster(model_file=str(model_path))
    print(f"[v2-sharded] setup done in {time.time()-t0:.1f}s, threshold={threshold}", flush=True)

    rows_done = sum(pf.metadata.row_group(r).num_rows for k, rgs in enumerate(shards)
                    if (SHARD_DIR / f"shard_{k:04d}.parquet").exists() for r in rgs)
    for j, (k, rgs) in enumerate(todo):
        t_b = time.time()
        df = pf.read_row_groups(rgs).to_pandas()
        feats = compute_batch_features(df, s1_attrs, target_attrs, name_rarity_counts)
        for col in feature_columns:
            if feats[col].dtype == bool:
                feats[col] = feats[col].astype(np.int8)
        scores = booster.predict(feats[feature_columns])
        out = {
            "s1_entity_id": pa.array(feats["s1_entity_id"].to_numpy(), type=pa.string()),
            "target_entity_id": pa.array(feats["target_entity_id"].to_numpy(), type=pa.string()),
            "score": pa.array(scores.astype(np.float64)),
        }
        for col in feature_columns:
            out[col] = pa.array(feats[col].to_numpy())
        table = pa.table(out)
        final = SHARD_DIR / f"shard_{k:04d}.parquet"
        tmp = SHARD_DIR / f"shard_{k:04d}.parquet.tmp"
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, final)
        rows_done += len(df)
        n_acc = int((scores >= threshold).sum())
        el = time.time() - t0
        print(f"[v2-sharded] shard {k} ({j+1}/{len(todo)}): {len(df):,} rows, {n_acc:,} accepted, "
              f"{time.time()-t_b:.1f}s; {rows_done:,}/{total_rows:,} ({100*rows_done/total_rows:.1f}%) elapsed {el:.0f}s",
              flush=True)
        write_status(stage="scoring", shards_done=len(shards) - len(todo) + j + 1, shards_total=len(shards),
                     rows_done=rows_done, rows_total=total_rows, elapsed_s=el)
        del df, feats, scores, table, out
    write_status(stage="score_done", shards_done=len(shards), shards_total=len(shards), rows_done=rows_done,
                 rows_total=total_rows, elapsed_s=time.time() - t0)
    print(f"[v2-sharded] scoring complete in {time.time()-t0:.1f}s", flush=True)


def verify_for_finalize():
    import duckdb

    t0 = time.time()
    config = run_config()
    with open(RUN_DIR / "config.json", encoding="utf-8") as fh:
        if json.load(fh) != config:
            raise SystemExit("config.json does not match the current model/candidate/feature configuration")
    threshold = config["threshold"]
    pf = pq.ParquetFile(UNION_PATH)
    n_rg = pf.metadata.num_row_groups
    n_shards = (n_rg + ROW_GROUPS_PER_SHARD - 1) // ROW_GROUPS_PER_SHARD
    missing = [k for k in range(n_shards) if not (SHARD_DIR / f"shard_{k:04d}.parquet").exists()]
    if missing:
        raise SystemExit(f"missing shards: {missing[:10]}... ({len(missing)})")

    # Inputs read during scoring must not have changed since the first shard was written.
    from data_io import TRAIN_PATHS
    first_shard_mtime = min((SHARD_DIR / f"shard_{k:04d}.parquet").stat().st_mtime for k in range(n_shards))
    inputs = [UNION_PATH, TEST_PATHS["source1"], TEST_PATHS["source2"], TEST_PATHS["source3"],
              TRAIN_PATHS["source2"], TRAIN_PATHS["source3"], MODEL_DIR / "lgbm_model.txt"]
    changed = [str(p) for p in inputs if p.stat().st_mtime >= first_shard_mtime]
    if changed:
        raise SystemExit(f"inputs modified after scoring started: {changed}")

    # Shard identity: each shard's (s1, target) columns must equal its source row groups exactly.
    for k in range(n_shards):
        rgs = list(range(k * ROW_GROUPS_PER_SHARD, min((k + 1) * ROW_GROUPS_PER_SHARD, n_rg)))
        src = pf.read_row_groups(rgs, columns=["s1_entity_id", "target_entity_id"])
        got = pq.read_table(SHARD_DIR / f"shard_{k:04d}.parquet", columns=["s1_entity_id", "target_entity_id"])
        for col in ("s1_entity_id", "target_entity_id"):
            if not src.column(col).cast(pa.string()).equals(got.column(col).cast(pa.string())):
                raise SystemExit(f"shard {k} column {col} does not match union row groups {rgs}")
    print(f"[v2-sharded] {n_shards} shards verified against union row groups ({time.time()-t0:.1f}s)", flush=True)
    return config, threshold, pf, n_shards, t0


def finalize() -> None:
    import duckdb

    config, threshold, pf, n_shards, t0 = verify_for_finalize()
    n_rg = pf.metadata.num_row_groups

    s1 = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])
    s1_ids = pa.table({"s1_entity_id": pa.array(s1["entity_id"].tolist(), type=pa.string()),
                       "ord": pa.array(np.arange(len(s1), dtype=np.int64))})
    del s1
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{(RUN_DIR / 'duckdb_tmp').as_posix()}'")
    con.register("s1_ids", s1_ids)
    shard_glob = (SHARD_DIR / "shard_*.parquet").as_posix()
    n_scored = con.execute(f"SELECT count(*) FROM read_parquet('{shard_glob}')").fetchone()[0]
    if n_scored != pf.metadata.num_rows:
        raise SystemExit(f"scored rows {n_scored} != union rows {pf.metadata.num_rows}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("candidate_pairs.tsv", "candidate_entity_ids", "TRUE"),
        ("matching_results.tsv", "matched_entity_ids", f"score >= {float(threshold)!r}"),
    ]
    n_parts = 16
    part_dir = RUN_DIR / "agg_parts"
    part_dir.mkdir(exist_ok=True)
    for fname, col, cond in specs:
        # Aggregate per hash partition of s1_entity_id to bound memory, then one ordered join.
        for p in range(n_parts):
            con.execute(f"""
                COPY (
                  SELECT s1_entity_id, string_agg(DISTINCT target_entity_id, ',' ORDER BY target_entity_id) AS ids
                  FROM read_parquet('{shard_glob}')
                  WHERE ({cond}) AND hash(s1_entity_id) % {n_parts} = {p}
                  GROUP BY s1_entity_id
                ) TO '{(part_dir / f"{col}_{p:02d}.parquet").as_posix()}' (FORMAT parquet)
            """)
        parts_glob = (part_dir / f"{col}_*.parquet").as_posix()
        dup = con.execute(f"SELECT count(*) - count(DISTINCT s1_entity_id) FROM read_parquet('{parts_glob}')").fetchone()[0]
        if dup:
            raise SystemExit(f"{dup} S1 ids appear in more than one partition")
        tmp = OUTPUT_DIR / (fname + ".tmp")
        con.execute(f"""
            COPY (
              SELECT s.s1_entity_id AS source1_entity_id, coalesce(a.ids, '') AS {col}
              FROM s1_ids s LEFT JOIN read_parquet('{parts_glob}') a USING (s1_entity_id) ORDER BY s.ord
            ) TO '{tmp.as_posix()}' (FORMAT csv, DELIMITER '	', HEADER true, QUOTE '', ESCAPE '')
        """)
        os.replace(tmp, OUTPUT_DIR / fname)
        print(f"[v2-sharded] wrote {OUTPUT_DIR / fname} ({time.time()-t0:.1f}s)", flush=True)

    cand_glob = (part_dir / "candidate_entity_ids_*.parquet").as_posix()
    match_glob = (part_dir / "matched_entity_ids_*.parquet").as_posix()
    stats = (con.execute(f"SELECT count(*) FROM read_parquet('{cand_glob}')").fetchone()[0],
             con.execute(f"SELECT count(*) FROM read_parquet('{match_glob}')").fetchone()[0],
             con.execute(f"SELECT count(*) FROM read_parquet('{shard_glob}') WHERE score >= {float(threshold)!r}").fetchone()[0])
    summary = {
        "n_test_s1": s1_ids.num_rows,
        "n_s1_with_candidates": stats[0],
        "n_s1_with_predicted_match": stats[1],
        "n_accepted_pairs": stats[2],
        "n_candidate_pairs_scored": n_scored,
        "threshold_used": threshold,
        "matching_results_sha256": sha256_file(OUTPUT_DIR / "matching_results.tsv"),
        "candidate_pairs_sha256": sha256_file(OUTPUT_DIR / "candidate_pairs.tsv"),
        "finalize_wall_time_s": time.time() - t0,
    }
    with open(RUN_DIR / "test_pipeline_v2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_status(stage="finalized", **summary)
    print(json.dumps(summary, indent=2), flush=True)


def finalize_fast() -> None:
    """Same outputs as finalize(), computed with integer codes + one numpy sort
    instead of a 200M-row SQL string aggregation."""
    import pyarrow.compute as pc

    config, threshold, pf, n_shards, t0 = verify_for_finalize()
    s1_ids = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])["entity_id"].tolist()
    t_ids = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
             + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    s1_arr = pa.array(s1_ids, pa.string())
    t_sorted = np.array(sorted(t_ids), dtype=object)  # byte order == Python str order for ASCII ids
    t_arr = pa.array(t_sorted.tolist(), pa.string())
    S, T, A = [], [], []
    for k in range(n_shards):
        tb = pq.read_table(SHARD_DIR / f"shard_{k:04d}.parquet", columns=["s1_entity_id", "target_entity_id", "score"])
        si = pc.index_in(tb["s1_entity_id"], value_set=s1_arr).to_numpy(zero_copy_only=False)
        ti = pc.index_in(tb["target_entity_id"], value_set=t_arr).to_numpy(zero_copy_only=False)
        if np.isnan(si.astype(float)).any() or np.isnan(ti.astype(float)).any():
            raise SystemExit(f"shard {k}: ids not found in test tables")
        S.append(si.astype(np.int32))
        T.append(ti.astype(np.int32))
        A.append(tb["score"].to_numpy() >= threshold)
    S, T, A = np.concatenate(S), np.concatenate(T), np.concatenate(A)
    n_scored = len(S)
    if n_scored != pf.metadata.num_rows:
        raise SystemExit(f"scored rows {n_scored} != union rows {pf.metadata.num_rows}")
    key = S.astype(np.int64) * len(t_sorted) + T.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, S, T, A = key[order], S[order], T[order], A[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    # a duplicated pair counts as accepted if any copy is accepted (same pair => same features => same score)
    A_any = np.maximum.reduceat(A.astype(np.int8), np.flatnonzero(first)).astype(bool)
    S, T, A = S[first], T[first], A_any
    del key, order, first
    print(f"[v2-sharded] {n_scored:,} scored rows -> {len(S):,} distinct pairs ({time.time()-t0:.1f}s)", flush=True)
    bounds = np.searchsorted(S, np.arange(len(s1_ids) + 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, col, mask in (("candidate_pairs.tsv", "candidate_entity_ids", None),
                             ("matching_results.tsv", "matched_entity_ids", A)):
        tmp = OUTPUT_DIR / (fname + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            buf = []
            for i, sid in enumerate(s1_ids):
                lo, hi = bounds[i], bounds[i + 1]
                tt = T[lo:hi] if mask is None else T[lo:hi][mask[lo:hi]]
                buf.append(sid + "\t" + ",".join(t_sorted[tt]) + "\n")
                if len(buf) >= 50_000:
                    fh.write("".join(buf))
                    buf = []
            fh.write("".join(buf))
        os.replace(tmp, OUTPUT_DIR / fname)
        print(f"[v2-sharded] wrote {OUTPUT_DIR / fname} ({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "n_test_s1": len(s1_ids),
        "n_s1_with_candidates": int((np.diff(bounds) > 0).sum()),
        "n_s1_with_predicted_match": int(len(np.unique(S[A]))),
        "n_accepted_pairs": int(A.sum()),
        "n_candidate_pairs_scored": int(n_scored),
        "n_distinct_candidate_pairs": int(len(S)),
        "threshold_used": threshold,
        "matching_results_sha256": sha256_file(OUTPUT_DIR / "matching_results.tsv"),
        "candidate_pairs_sha256": sha256_file(OUTPUT_DIR / "candidate_pairs.tsv"),
        "finalize_wall_time_s": time.time() - t0,
        "finalize_method": "finalize_fast",
    }
    with open(RUN_DIR / "test_pipeline_v2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_status(stage="finalized", **summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"score": score, "finalize": finalize, "finalize_fast": finalize_fast}[mode]()
