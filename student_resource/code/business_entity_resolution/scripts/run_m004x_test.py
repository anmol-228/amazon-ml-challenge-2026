"""EXP-S2X variant of run_m004_test.py (adds src/stage2x.py features when the stage-2 model lists them).
M004 test inference: stage-1 probabilities (m003 main model, from a stage-1
test run's scored.parquet, same row order as features_test.parquet) -> stage-2
collective features over the full test candidate table -> stage-2 model ->
global threshold -> output TSVs (numpy exporter).

Usage: python -u run_m004_test.py --feat feat_test_r1 --stage1-run sub003_m003a_global074
                                  --model m004_a --threshold 0.66 --run-name NAME
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_io import TEST_PATHS, load_source_table  # noqa: E402
from export_outputs import export  # noqa: E402
from stage2x import STAGE2X_FEATURES, stage2x_features  # noqa: E402
from stage2 import stage2_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P_MIN = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--stage1-run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()
    if not (np.isfinite(args.threshold) and 0 <= args.threshold <= 1):
        raise SystemExit("invalid threshold")
    t0 = time.time()
    fdir = ROOT / "experiments" / "m003" / args.feat
    mdir = ROOT / "experiments" / "m003" / args.model
    run_dir = ROOT / "experiments" / "m003" / "test_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    s1_run = ROOT / "experiments" / "m003" / "test_runs" / args.stage1_run
    s1_meta = json.load(open(s1_run / "summary_in.json", encoding="utf-8"))
    stage1_model_sha = hashlib.sha256(open(ROOT / "experiments" / "m003" / s1_meta["model"] / "lgbm_model.txt", "rb").read()).hexdigest()
    if stage1_model_sha != s1_meta["model_sha256"]:
        raise SystemExit("stage-1 model changed since the stage-1 run")

    feat_path = fdir / "features_test.parquet"
    idx = pq.read_table(feat_path, columns=["s1_idx", "t_idx"])
    scored = pq.read_table(s1_run / "scored.parquet", columns=["prob"])
    if scored.num_rows != idx.num_rows:
        raise SystemExit("stage-1 scored rows do not align with the feature table")
    s = idx["s1_idx"].to_numpy().astype(np.int64)
    t = idx["t_idx"].to_numpy().astype(np.int64)
    p1 = scored["prob"].to_numpy().astype(np.float32)
    # alignment check: scored ids must equal feature-table ids row by row
    s1_ids = pq.read_table(fdir / "s1_ids.parquet")["entity_id"]
    t_ids = pq.read_table(fdir / "t_ids.parquet")["entity_id"]
    chk = pq.read_table(s1_run / "scored.parquet", columns=["s1_entity_id", "target_entity_id"])
    import pyarrow.compute as pc
    # full row-by-row alignment: every scored id must equal the feature-table id at the same position
    if not (pc.index_in(chk["s1_entity_id"], value_set=s1_ids).to_numpy(zero_copy_only=False) == s).all():
        raise SystemExit("row alignment check failed (s1)")
    if not (pc.index_in(chk["target_entity_id"], value_set=t_ids).to_numpy(zero_copy_only=False) == t).all():
        raise SystemExit("row alignment check failed (target)")
    del chk
    print(f"[m004-test] {len(s):,} rows aligned ({time.time()-t0:.0f}s)", flush=True)

    s2 = stage2_features(s, t, p1)
    keep = p1 >= P_MIN
    feats2 = json.load(open(mdir / "stage2_feature_columns.json", encoding="utf-8"))
    extras = [c for c in feats2 if c in STAGE2X_FEATURES]
    if extras:  # EXP-S2X extra collective features (target source from the S2-/S3- id prefix)
        is_s2 = pc.starts_with(t_ids, "S2-").to_numpy(zero_copy_only=False).astype(np.int64)
        s2 = pd.concat([s2, stage2x_features(s, t, p1, is_s2[t])[extras]], axis=1)
        print(f"[m004x-test] extra stage-2 features {extras}", flush=True)
    base_cols = [c for c in feats2 if c not in s2.columns]
    b2 = lgb.Booster(model_file=str(mdir / "stage2_model.txt"))
    p2 = np.zeros(len(s), dtype=np.float32)
    pf = pq.ParquetFile(feat_path)
    off = 0
    for i in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        m = keep[off:off + n]
        if m.any():
            base = pf.read_row_group(i, columns=base_cols).to_pandas()[m].reset_index(drop=True)
            X = pd.concat([s2.iloc[off + np.flatnonzero(m)].reset_index(drop=True), base], axis=1)
            p2[off + np.flatnonzero(m)] = b2.predict(X[feats2]).astype(np.float32)
        off += n
    sel = p2 >= args.threshold
    print(f"[m004-test] stage-2 scored; {int(keep.sum()):,} rows with p1>={P_MIN}, {int(sel.sum()):,} selected "
          f"({time.time()-t0:.0f}s)", flush=True)
    pq.write_table(pa.table({"s1_entity_id": s1_ids.take(pa.array(s)), "target_entity_id": t_ids.take(pa.array(t)),
                             "p1": pa.array(p1), "prob": pa.array(p2), "selected": pa.array(sel)}),
                   run_dir / "scored.parquet", compression="zstd")
    cand = s1_meta.get("candidates") or (fdir / "cand_tag.txt").read_text(encoding="utf-8").strip()
    meta = {"run": args.run_name, "candidates": cand, "features": args.feat,
            "stage1_model": s1_meta["model"], "stage1_model_sha256": stage1_model_sha, "stage2_model": args.model,
            "stage2_model_sha256": hashlib.sha256(open(mdir / "stage2_model.txt", "rb").read()).hexdigest(),
            "decoder": "global", "threshold": args.threshold, "p_min": P_MIN}
    json.dump(meta, open(run_dir / "summary_in.json", "w", encoding="utf-8"), indent=2)
    s1_order = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])["entity_id"].tolist()
    targets = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
               + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    summary = export(s1_ids.take(pa.array(s)), t_ids.take(pa.array(t)), sel, s1_order, targets,
                     ROOT / "student_resource" / "output", meta)
    json.dump(summary, open(run_dir / "summary.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
