"""M003 test inference: score features_test.parquet with an M003 model, decode
(global threshold, optionally + target exclusivity), and write
student_resource/output/{matching_results,candidate_pairs}.tsv atomically.

candidate_pairs.tsv = every scored (s1, target) row (the exact set fed to the
matcher); matching_results.tsv = decoded subset.

Usage: python -u run_m003_test.py --feat FEATDIR --model MODELDIR --decoder {global,exclusive} --threshold T
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_io import TEST_PATHS, load_source_table  # noqa: E402
from fast_eval import target_exclusive  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "student_resource" / "output"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoder", choices=["global", "exclusive"], required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--no-export", action="store_true", help="only write scored.parquet (stage-1 input for stage 2)")
    args = ap.parse_args()
    if not (np.isfinite(args.threshold) and 0.0 <= args.threshold <= 1.0):
        raise SystemExit(f"invalid threshold {args.threshold}")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    done_marker = OUT / "OUTPUT_COMPLETE.json"
    if done_marker.exists() and not args.no_export:
        done_marker.unlink()
    feat_path = ROOT / "experiments" / "m003" / args.feat / "features_test.parquet"
    mdir = ROOT / "experiments" / "m003" / args.model
    run_dir = ROOT / "experiments" / "m003" / "test_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(mdir / "feature_columns.json", encoding="utf-8") as fh:
        cols = json.load(fh)
    booster = lgb.Booster(model_file=str(mdir / "lgbm_model.txt"))

    pf = pq.ParquetFile(feat_path)
    s1_parts, t_parts, p_parts = [], [], []
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=["s1_idx", "t_idx"] + cols).to_pandas()
        p_parts.append(booster.predict(df[cols]).astype(np.float32))
        s1_parts.append(df["s1_idx"].to_numpy())
        t_parts.append(df["t_idx"].to_numpy())
        if i % 10 == 0:
            print(f"[m003-test] row group {i+1}/{pf.metadata.num_row_groups} ({time.time()-t0:.0f}s)", flush=True)
    s1i = np.concatenate(s1_parts)
    tgi = np.concatenate(t_parts)
    prob = np.concatenate(p_parts)
    del s1_parts, t_parts, p_parts
    fdir = feat_path.parent
    s1_id_arr = pq.read_table(fdir / "s1_ids.parquet")["entity_id"]
    t_id_arr = pq.read_table(fdir / "t_ids.parquet")["entity_id"]
    sel = prob >= args.threshold
    if args.decoder == "exclusive":
        sel = target_exclusive(tgi, prob, sel)
    print(f"[m003-test] {len(prob):,} rows scored, {int(sel.sum()):,} selected ({time.time()-t0:.0f}s)", flush=True)
    scored = pa.table({"s1_entity_id": s1_id_arr.take(pa.array(s1i)), "target_entity_id": t_id_arr.take(pa.array(tgi)),
                       "prob": pa.array(prob), "selected": pa.array(sel)})
    pq.write_table(scored, run_dir / "scored.parquet", compression="zstd")
    meta = {"run": args.run_name, "features": args.feat, "model": args.model,
            "model_sha256": sha256_file(mdir / "lgbm_model.txt"), "decoder": args.decoder, "threshold": args.threshold,
            "n_rows": int(len(prob)), "n_selected": int(sel.sum())}
    cand_args = feat_path.parent / "cand_tag.txt"
    if cand_args.exists():
        meta["candidates"] = cand_args.read_text(encoding="utf-8").strip()
    with open(run_dir / "summary_in.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if args.no_export:
        print(json.dumps(meta, indent=2), flush=True)
        return
    from export_outputs import export
    s1_order = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])["entity_id"].tolist()
    targets = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
               + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    summary = export(scored["s1_entity_id"], scored["target_entity_id"], sel, s1_order, targets, OUT, meta)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
