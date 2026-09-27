"""Build M003 feature tables from R1 candidates (integer-coded, memory-bounded).

dev  : candidates experiments/r1/<tag>/cand_*.parquet built on the train corpus;
       labeled from train ground truth; one output file per phase3 split.
test : candidates built on the test corpus; one output file (split "test").

Rows carry s1_idx / t_idx (row positions in the source tables as loaded by
data_io.load_source_table; S2 rows first, then S3, for targets). The id
lookup tables are written next to the features (s1_ids.parquet, t_ids.parquet).
Record statistics (token IDF) come from the record set being resolved (train
records for dev, test records for test), computed identically.

Usage: python -u build_m003_features.py {dev|test} --cand-tag TAG --out OUT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_io import TEST_PATHS, TRAIN_PATHS, load_ground_truth, load_source_table  # noqa: E402
from features_v3 import RecordTable, competition_features, corpus_idf, corpus_idf_rows, pair_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SPLIT_PATH = ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"
RANK_SENTINEL = 99.0
BATCH = 3_000_000


def load_candidates(path: Path, s1_ids: pa.Array, t_ids: pa.Array) -> pd.DataFrame:
    tb = pq.read_table(path)
    si = pc.index_in(tb["s1_entity_id"], value_set=s1_ids).to_numpy(zero_copy_only=False)
    ti = pc.index_in(tb["target_entity_id"], value_set=t_ids).to_numpy(zero_copy_only=False)
    if np.isnan(si.astype(float)).any() or np.isnan(ti.astype(float)).any():
        raise SystemExit(f"unmapped ids in {path}")
    f_s = tb["r1_fwd_score"].to_numpy(zero_copy_only=False).astype(np.float32)
    r_s = tb["r1_rev_score"].to_numpy(zero_copy_only=False).astype(np.float32)
    df = pd.DataFrame({
        "s1_idx": si.astype(np.int32), "t_idx": ti.astype(np.int32),
        "r1_fwd_rank": np.nan_to_num(tb["r1_fwd_rank"].to_numpy(zero_copy_only=False).astype(np.float32), nan=RANK_SENTINEL),
        "r1_rev_rank": np.nan_to_num(tb["r1_rev_rank"].to_numpy(zero_copy_only=False).astype(np.float32), nan=RANK_SENTINEL),
        "r1_score": np.fmax(f_s, r_s).astype(np.float32),
    })
    df["in_fwd"] = (df["r1_fwd_rank"] < RANK_SENTINEL).astype(np.float32)
    df["in_rev"] = (df["r1_rev_rank"] < RANK_SENTINEL).astype(np.float32)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--cand-tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--norm", default="v1", choices=["v1", "v2"])
    ap.add_argument("--idf-scope", default="global", choices=["global", "country"],
                    help="country: IDF n and df counted within the candidate file's country partition")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / "experiments" / "m003" / args.out
    out_dir.mkdir(parents=True, exist_ok=False)
    paths = TRAIN_PATHS if args.mode == "dev" else TEST_PATHS

    s1 = load_source_table(paths["source1"])
    t = pd.concat([load_source_table(paths["source2"]), load_source_table(paths["source3"])], ignore_index=True)
    s1_ids = pa.array(s1["entity_id"].tolist(), pa.string())
    t_ids = pa.array(t["entity_id"].tolist(), pa.string())
    pq.write_table(pa.table({"entity_id": s1_ids}), out_dir / "s1_ids.parquet")
    pq.write_table(pa.table({"entity_id": t_ids}), out_dir / "t_ids.parquet")
    is_s2 = np.array([x.startswith("S2-") for x in t["entity_id"]], dtype=np.float32)
    s_country = s1["country"].to_numpy()
    t_country = t["country"].to_numpy()
    S = RecordTable(s1["entity_id"], s1["business_name"], s1["business_address"], norm=args.norm)
    T = RecordTable(t["entity_id"], t["business_name"], t["business_address"], norm=args.norm)
    n_t = len(t)
    del s1, t
    idf_n = corpus_idf([S, T], "ntok")
    idf_a = corpus_idf([S, T], "atok")
    print(f"[m003f] record tables ({time.time()-t0:.0f}s)", flush=True)

    split_code = None
    truth_keys = None
    split_names = ["matcher_train", "dev_eval", "calibration_holdout"]
    if args.mode == "dev":
        gt = load_ground_truth()
        gt = gt[gt["matched_entity_ids"] != ""]
        s_list, t_list = [], []
        for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
            for x in m.split(","):
                s_list.append(s)
                t_list.append(x)
        gs = pc.index_in(pa.array(s_list, pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False)
        gtt = pc.index_in(pa.array(t_list, pa.string()), value_set=t_ids).to_numpy(zero_copy_only=False)
        truth_keys = np.sort(gs.astype(np.int64) * n_t + gtt.astype(np.int64))
        split = pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)
        sp_idx = pc.index_in(pa.array(split["source1_entity_id"].tolist(), pa.string()), value_set=s1_ids)
        split_code = np.full(len(s1_ids), -1, dtype=np.int8)
        codes = split["phase3_split"].map({n: i for i, n in enumerate(split_names)}).to_numpy()
        valid = ~np.isnan(sp_idx.to_numpy(zero_copy_only=False).astype(float))
        split_code[sp_idx.to_numpy(zero_copy_only=False)[valid].astype(np.int64)] = codes[valid]
        print(f"[m003f] truth keys {len(truth_keys):,} ({time.time()-t0:.0f}s)", flush=True)

    writers: dict[str, pq.ParquetWriter] = {}
    files = sorted((ROOT / "experiments" / "r1" / args.cand_tag).glob("cand_*.parquet"))
    for path in files:
        cand = load_candidates(path, s1_ids, t_ids)
        if args.idf_scope == "country":
            ctry = path.stem.split("_", 1)[1]
            parts = [(S, np.flatnonzero(s_country == ctry)), (T, np.flatnonzero(t_country == ctry))]
            assert len(parts[0][1]) and len(parts[1][1]), ctry
            idf_n = corpus_idf_rows(parts, "ntok")
            idf_a = corpus_idf_rows(parts, "atok")
            print(f"[m003f] {ctry}: per-country IDF over {len(parts[0][1]):,} S1 + {len(parts[1][1]):,} targets", flush=True)
        comp = competition_features(cand.rename(columns={"s1_idx": "s1_entity_id", "t_idx": "target_entity_id"}),
                                    "r1_score")
        cand = pd.concat([cand, comp], axis=1)
        del comp
        cand["is_source2"] = is_s2[cand["t_idx"].to_numpy()]
        if args.mode == "dev":
            keys = cand["s1_idx"].to_numpy().astype(np.int64) * n_t + cand["t_idx"].to_numpy().astype(np.int64)
            cand["label"] = np.isin(keys, truth_keys, assume_unique=False)
            cand["split"] = split_code[cand["s1_idx"].to_numpy()]
        print(f"[m003f] {path.name}: {len(cand):,} rows ({time.time()-t0:.0f}s)", flush=True)
        for lo in range(0, len(cand), BATCH):
            hi = min(lo + BATCH, len(cand))
            part = cand.iloc[lo:hi].reset_index(drop=True)
            f = pair_features(S, T, part["s1_idx"].to_numpy(), part["t_idx"].to_numpy(), idf_n, idf_a, args.workers)
            for k, v in f.items():
                part[k] = v
            if args.mode == "dev":
                groups = [(split_names[c], g.drop(columns=["split"])) for c, g in part.groupby("split") if c >= 0]
            else:
                groups = [("test", part)]
            for name, g in groups:
                tbl = pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(out_dir / f"features_{name}.parquet", tbl.schema,
                                                     compression="zstd")
                writers[name].write_table(tbl, row_group_size=1_000_000)
            print(f"[m003f]   batch {lo:,}-{hi:,} ({time.time()-t0:.0f}s)", flush=True)
        del cand
    for w in writers.values():
        w.close()
    # candidate-set provenance read by run_m003_test.py / run_m004_test.py
    (out_dir / "cand_tag.txt").write_text(f"{args.cand_tag} (norm {args.norm}, idf-scope {args.idf_scope})\n",
                                          encoding="utf-8")
    print(f"[m003f] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
