"""Equal-weight blend of two stage-2 runs that score the SAME candidate rows (same feature tables).

dev mode : average pred2_{calibration_holdout,dev_eval}.npy of two lab/model dirs, select the global threshold on
           calibration_holdout, report dev_eval and a paired bootstrap vs a reference entity file.
test mode: average `prob` of two test-run scored.parquet files (row-aligned, ids checked), apply the threshold,
           export matching_results.tsv / candidate_pairs.tsv to student_resource/output.

Usage: python blend_stage2.py dev  --feat feat_dev_r3idf --a s2x_r3aug --b s2x_r3mix --ref m004_r3aug --out blend_ab
       python blend_stage2.py test --feat feat_test_r3idf --a sub_s2x_r3aug --b sub_s2x_r3mix --threshold T --run-name sub_blend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_io import TEST_PATHS, load_source_table  # noqa: E402
from export_outputs import export  # noqa: E402
from fast_eval import macro_f, per_entity_f  # noqa: E402
from train_m003 import GRID, ROOT, entity_codes, population  # noqa: E402

M = ROOT / "experiments" / "m003"


def dev(args) -> None:
    feat = M / args.feat
    out = M / args.out
    out.mkdir(parents=True, exist_ok=False)
    res, preds = {}, {}
    for n in ("calibration_holdout", "dev_eval"):
        t = pq.read_table(feat / f"features_{n}.parquet", columns=["s1_idx", "label"])
        s1, lab = t["s1_idx"].to_numpy(), t["label"].to_numpy().astype(bool)
        p = (np.load(M / args.a / f"pred2_{n}.npy") + np.load(M / args.b / f"pred2_{n}.npy")) / 2
        pop = population(n, feat)
        ent = entity_codes(s1, pop[1])
        res[n] = {float(thr): macro_f(ent, lab, p >= thr, pop[2]) for thr in GRID}
        preds[n] = (p, ent, lab, pop)
    thr = max(res["calibration_holdout"], key=res["calibration_holdout"].get)
    p, ent, lab, (ids, _, truth) = preds["dev_eval"]
    sel = p >= thr
    f = per_entity_f(ent, lab, sel, truth)
    n = len(ids)
    B = pd.DataFrame({"s1_entity_id": ids, "f": f, "tp": np.bincount(ent[sel & lab], minlength=n),
                      "pred": np.bincount(ent[sel], minlength=n), "truth": truth})
    B.to_parquet(out / "entity_dev_stage2_global.parquet")
    A = pd.read_parquet(M / args.ref / "entity_dev_stage2_global.parquet").set_index("s1_entity_id")
    B = B.set_index("s1_entity_id").loc[A.index]
    d = (B["f"] - A["f"]).to_numpy()
    rng = np.random.default_rng(12345)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    summ = {"a": args.a, "b": args.b, "threshold": thr, "calibration_holdout": res["calibration_holdout"][thr],
            "dev_eval": res["dev_eval"][thr], "ref": args.ref, "delta_vs_ref": float(d.mean()),
            "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
    json.dump({"summary": summ, "curves": res}, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ), flush=True)


def test(args) -> None:
    run = M / "test_runs"
    ta = pq.read_table(run / args.a / "scored.parquet", columns=["s1_entity_id", "target_entity_id", "prob"])
    tb = pq.read_table(run / args.b / "scored.parquet", columns=["s1_entity_id", "target_entity_id", "prob"])
    assert ta.num_rows == tb.num_rows
    assert ta["s1_entity_id"].equals(tb["s1_entity_id"]) and ta["target_entity_id"].equals(tb["target_entity_id"])
    p = (ta["prob"].to_numpy() + tb["prob"].to_numpy()) / 2
    sel = p >= args.threshold
    out_dir = run / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.table({"s1_entity_id": ta["s1_entity_id"], "target_entity_id": ta["target_entity_id"],
                             "prob": pa.array(p.astype(np.float32)), "selected": pa.array(sel)}),
                   out_dir / "scored.parquet", compression="zstd")
    meta = {"run": args.run_name, "blend_of": [args.a, args.b], "threshold": args.threshold, "features": args.feat}
    s1_order = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])["entity_id"].tolist()
    targets = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
               + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    summary = export(ta["s1_entity_id"], ta["target_entity_id"], sel, s1_order, targets,
                     ROOT / "student_resource" / "output", meta)
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--feat", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ref", default="m004_r3aug")
    ap.add_argument("--out", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    dev(args) if args.mode == "dev" else test(args)


if __name__ == "__main__":
    main()
