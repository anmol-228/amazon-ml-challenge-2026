"""R1 pair-shingle retrieval: build candidates for dev evaluation or test.

dev  : index = ALL train S1 / ALL train targets (full-corpus realism).
       forward queries = dev_eval S1 (phase3_split_v1), reverse queries = all
       train targets; reverse pairs are kept only when the S1 is in dev_eval.
test : index = all test S1 / all test targets, forward queries = all test S1.

Per country partition (open set): hash keys -> df filter/IDF (partition's own
records) -> forward top-KF and reverse top-KR by sparse cosine.

Usage: python -u run_r1_pair_retrieval.py {dev|test} --max-df 2000 --kf 30 --kr 5 [--cost-only]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_io import TEST_PATHS, TRAIN_PATHS, load_source_table  # noqa: E402
from pair_retrieval import (KeyConfig, hash_records_parallel, prepare_partition,  # noqa: E402
                            retrieve_topk, save_csr, weight_partition)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_ROOT = PROJECT_ROOT / "experiments" / "r1"
SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--max-df", type=int, default=2000)
    ap.add_argument("--kf", type=int, default=30)
    ap.add_argument("--kr", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--all-queries", action="store_true", help="dev mode: forward-query every train S1 and keep all reverse pairs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--norm", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()

    t0 = time.time()
    paths = TRAIN_PATHS if args.mode == "dev" else TEST_PATHS
    tag = args.tag or f"{args.mode}{'all' if args.all_queries else ''}_df{args.max_df}_kf{args.kf}_kr{args.kr}"
    out_dir = OUT_ROOT / tag
    work = OUT_ROOT / "work" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    arg_path = out_dir / "args.json"
    run_args = {k: v for k, v in vars(args).items() if k not in ("workers", "tag")}
    if arg_path.exists():
        with open(arg_path, encoding="utf-8") as fh:
            if json.load(fh) != run_args:
                raise SystemExit(f"{out_dir} was produced with different arguments; use a new --tag")
    elif any(out_dir.glob("cand_*.parquet")):
        raise SystemExit(f"{out_dir} has candidate files but no args.json; refusing to reuse")
    with open(arg_path, "w", encoding="utf-8") as fh:
        json.dump(run_args, fh)

    s1 = load_source_table(paths["source1"])
    t = pd.concat([load_source_table(paths["source2"]), load_source_table(paths["source3"])], ignore_index=True)
    if args.mode == "dev":
        split = pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)
        dev_ids = set(split.loc[split["phase3_split"] == "dev_eval", "source1_entity_id"])
        s1["is_query"] = True if args.all_queries else s1["entity_id"].isin(dev_ids)
    else:
        s1["is_query"] = True
    print(f"[r1] loaded s1={len(s1):,} t={len(t):,} queries={int(s1['is_query'].sum()):,} ({time.time()-t0:.0f}s)", flush=True)

    all_stats = {"args": vars(args), "partitions": {}}
    parts = []
    countries = sorted(set(s1["country"]) | set(t["country"]))
    ckeys = ["".join(ch for ch in c if ch.isalnum()) or "x" for c in countries]
    if len(set(ckeys)) != len(ckeys):
        raise SystemExit(f"country file-key collision: {list(zip(countries, ckeys))}")
    for country in countries:
        tc = time.time()
        ckey = "".join(ch for ch in country if ch.isalnum()) or "x"
        if not args.cost_only and (out_dir / f"cand_{ckey}.parquet").exists():
            print(f"[r1] {country}: already done, skipping", flush=True)
            continue
        s1c = s1[s1["country"] == country].reset_index(drop=True)
        tcc = t[t["country"] == country].reset_index(drop=True)
        if len(s1c) == 0 or len(tcc) == 0:
            continue
        cfg = KeyConfig(norm=args.norm)
        m1 = hash_records_parallel(s1c["business_name"].tolist(), s1c["business_address"].tolist(), cfg, args.workers)
        mt = hash_records_parallel(tcc["business_name"].tolist(), tcc["business_address"].tolist(), cfg, args.workers)
        a1, at, st = weight_partition(m1, mt, args.max_df)
        del m1, mt
        st["n_s1"], st["n_t"] = len(s1c), len(tcc)
        st["nnz_s1"], st["nnz_t"] = int(a1.nnz), int(at.nnz)
        print(f"[r1] {country}: {st} ({time.time()-tc:.0f}s)", flush=True)
        all_stats["partitions"][country] = st
        if args.cost_only:
            continue

        p = prepare_partition(str(work), ckey, a1, at)
        qmask = s1c["is_query"].to_numpy()
        q_idx = np.flatnonzero(qmask)
        qprefix = os.path.join(str(work), f"{ckey}_fq")
        save_csr(qprefix, a1[q_idx])
        del a1, at

        tf = time.time()
        fr, fc, fs, frk = retrieve_topk(qprefix, p["tT"], len(q_idx), args.kf, args.workers)
        print(f"[r1] {country}: forward {len(fr):,} pairs ({time.time()-tf:.0f}s)", flush=True)
        tr = time.time()
        rr, rc, rs, rrk = retrieve_topk(p["t"], p["s1T"], len(tcc), args.kr, args.workers)
        keep = qmask[rc]
        rr, rc, rs, rrk = rr[keep], rc[keep], rs[keep], rrk[keep]
        print(f"[r1] {country}: reverse {len(rr):,} pairs kept ({time.time()-tr:.0f}s)", flush=True)

        s1_ids = s1c["entity_id"].to_numpy()
        t_ids = tcc["entity_id"].to_numpy()
        fwd = pd.DataFrame({"s1_entity_id": s1_ids[q_idx[fr]], "target_entity_id": t_ids[fc],
                            "r1_fwd_rank": frk, "r1_fwd_score": fs})
        rev = pd.DataFrame({"s1_entity_id": s1_ids[rc], "target_entity_id": t_ids[rr],
                            "r1_rev_rank": rrk, "r1_rev_score": rs})
        cand = fwd.merge(rev, on=["s1_entity_id", "target_entity_id"], how="outer")
        cand["country"] = country
        tmp_path = out_dir / f"cand_{ckey}.parquet.tmp"
        pq.write_table(pa.Table.from_pandas(cand, preserve_index=False), tmp_path, compression="zstd")
        os.replace(tmp_path, out_dir / f"cand_{ckey}.parquet")
        parts.append(len(cand))
        all_stats["partitions"][country].update(
            n_fwd=int(len(fwd)), n_rev=int(len(rev)), n_cand=int(len(cand)),
            forward_s=None, total_s=time.time() - tc)
        print(f"[r1] {country}: {len(cand):,} candidate pairs written ({time.time()-tc:.0f}s)", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

    all_stats["wall_s"] = time.time() - t0
    with open(out_dir / ("cost.json" if args.cost_only else "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"[r1] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
