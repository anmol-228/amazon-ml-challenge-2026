"""Threshold-transfer stress on dev_eval from saved predictions.

For each system and country: F0.5 at the calibration-selected threshold, the
country's own best threshold and F0.5 (diagnostic only, never used for
selection), and the F0.5 loss when the threshold is shifted by +/-0.05 and
+/-0.10 (sensitivity to a probability-scale shift in an unseen domain).

Usage: python threshold_stress.py name:feat_dir:pred_file:threshold [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_eval import macro_f  # noqa: E402
from train_m003 import GRID, ROOT, entity_codes, population  # noqa: E402


def main() -> None:
    out = {}
    s1c = pd.read_csv(ROOT / "student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str,
                      keep_default_na=False, usecols=["entity_id", "country"])
    country_of = dict(zip(s1c["entity_id"], s1c["country"]))
    for spec in sys.argv[1:]:
        name, feat, pred, thr = spec.split(":")
        thr = float(thr)
        fd = ROOT / "experiments" / "m003" / feat
        tb = pq.read_table(fd / "features_dev_eval.parquet", columns=["s1_idx", "label"])
        ids, pidx, truth = population("dev_eval", fd)
        ent = entity_codes(tb["s1_idx"].to_numpy(), pidx)
        lab = tb["label"].to_numpy().astype(bool)
        p = np.load(ROOT / "experiments" / "m003" / pred)
        ctry = np.array([country_of[i] for i in ids])
        res = {}
        for c in sorted(set(ctry)):
            keep_e = ctry == c
            lut = np.full(len(ids), -1)
            lut[keep_e] = np.arange(keep_e.sum())
            e2 = lut[ent]
            m = e2 >= 0
            f = lambda t: macro_f(e2[m], lab[m], p[m] >= t, truth[keep_e])
            curve = {float(t): f(t) for t in GRID}
            best = max(curve, key=curve.get)
            at = f(thr)
            res[c] = {"at_selected": at, "own_best_thr": best, "own_best": curve[best],
                      "loss_vs_own_best": curve[best] - at,
                      "shift_-0.10": at - f(thr - 0.10), "shift_-0.05": at - f(thr - 0.05),
                      "shift_+0.05": at - f(thr + 0.05), "shift_+0.10": at - f(thr + 0.10)}
        out[name] = {"threshold": thr, "countries": res}
        print(name, json.dumps(res, indent=1), flush=True)
    json.dump(out, open(ROOT / "experiments" / "m003" / "threshold_stress.json", "w"), indent=1)


if __name__ == "__main__":
    main()
