"""Final decoder of the submitted system: per-entity adaptive thresholds on the stage-2 test probabilities.

Input : a stage-2 test run (run_m004x_test.py) -> experiments/m003/test_runs/<run>/{scored.parquet, summary_in.json}
        and the frozen rule (artifacts/decoder/frozen_rule.json, fitted by fit_adaptive_decoder.py on
        calibration_holdout only).
Rule  : group = 3 * [target is a Source-3 record] + min(#candidates of the Source-1 entity with p >= 0.90, 2);
        a pair is selected if p >= thresholds[group]  (Source 2: 0.44 / 0.73 / 0.76, Source 3: 0.47 / 0.67 / 0.78).
Scope : Source-1 entities whose country occurs in the training data. Entities of a country absent from training
        keep the run's global stage-2 threshold (the rule is only validated on labeled countries).
Output: matching_results.tsv, candidate_pairs.tsv (same candidate set as the stage-2 run) in --out.

Usage: python -u scripts/apply_adaptive_decoder.py --test-run final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE / "src"))

from data_io import TEST_PATHS, TRAIN_PATHS, load_source_table  # noqa: E402
from export_outputs import export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-run", default="final", help="dir under experiments/m003/test_runs")
    ap.add_argument("--rule", default=str(CODE / "artifacts" / "decoder" / "frozen_rule.json"))
    ap.add_argument("--out", default=str(ROOT / "student_resource" / "output"))
    args = ap.parse_args()
    t0 = time.time()
    run = ROOT / "experiments" / "m003" / "test_runs" / args.test_run
    rule = json.load(open(args.rule, encoding="utf-8"))
    if rule["family"] != "source_conf_size" or rule["alpha"] != 0 or len(rule["thresholds"]) != 6:
        raise SystemExit(f"unsupported rule {rule}")
    thr = np.asarray(rule["thresholds"], dtype=np.float64)
    thr_global = json.load(open(run / "summary_in.json", encoding="utf-8"))["threshold"]

    tab = pq.read_table(run / "scored.parquet", columns=["s1_entity_id", "target_entity_id", "prob"])
    p32 = tab["prob"].to_numpy()
    p = p32.astype(np.float64)
    s1 = load_source_table(TEST_PATHS["source1"], usecols=["entity_id", "country"])
    train_countries = set(load_source_table(TRAIN_PATHS["source1"], usecols=["country"])["country"])
    s1_order = s1["entity_id"].tolist()
    ent = pc.index_in(tab["s1_entity_id"], value_set=pa.array(s1_order)).to_numpy(zero_copy_only=False)
    if np.isnan(ent.astype(float)).any():
        raise SystemExit("scored Source-1 ids not found in the test table")
    ent = ent.astype(np.int64)
    seen = s1["country"].isin(train_countries).to_numpy()[ent]

    conf = np.minimum(np.bincount(ent[p >= .90], minlength=len(s1_order)), 2)[ent]
    src3 = pc.starts_with(tab["target_entity_id"], "S3-").to_numpy(zero_copy_only=False).astype(np.int64)
    adaptive = p >= thr[3 * src3 + conf]
    sel = np.where(seen, adaptive, p32 >= thr_global)
    print(f"[decoder] {len(p):,} rows; training countries {sorted(train_countries)}; {int(seen.sum()):,} rows in "
          f"training countries; selected {int(sel.sum()):,} (global threshold {thr_global}: "
          f"{int((p32 >= thr_global).sum()):,}) ({time.time() - t0:.0f}s)", flush=True)

    meta = {"decoder": "adaptive per-entity thresholds (training countries) / global (other countries)",
            "rule": rule, "global_threshold": thr_global, "training_countries": sorted(train_countries),
            "stage2_run": args.test_run}
    targets = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
               + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    summary = export(tab["s1_entity_id"], tab["target_entity_id"], sel, s1_order, targets, Path(args.out), meta)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
