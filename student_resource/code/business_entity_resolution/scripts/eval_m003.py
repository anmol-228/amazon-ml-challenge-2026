"""Evaluate a saved M003 model on calibration_holdout / dev_eval (full populations).

global decoder: exact. Threshold selected on calibration_holdout, reported on dev_eval.
exclusive decoder (APPROXIMATE): exclusivity applied over the union of
calibration_holdout and dev_eval rows only (targets can also be contested by
matcher_train S1s, which are not scored here); the exact version needs
cross-fitted matcher_train probabilities.

Usage: python -u eval_m003.py --feat FEATDIR --model MODELDIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_eval import macro_f, target_exclusive  # noqa: E402
from train_m003 import GRID, ROOT, entity_codes, population, predict_file, write_entity_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    feat_dir = ROOT / "experiments" / "m003" / args.feat
    mdir = ROOT / "experiments" / "m003" / args.model
    cols = json.load(open(mdir / "feature_columns.json", encoding="utf-8"))
    booster = lgb.Booster(model_file=str(mdir / "lgbm_model.txt"))
    names = ["calibration_holdout", "dev_eval"]
    preds, pops, ents = {}, {}, {}
    for n in names:
        preds[n] = predict_file(feat_dir / f"features_{n}.parquet", booster, cols)
        np.save(mdir / f"pred_{n}.npy", preds[n]["prob"])
        pops[n] = population(n, feat_dir)
        ents[n] = entity_codes(preds[n]["s1_idx"], pops[n][1])
        assert (ents[n] >= 0).all()
        print(f"predicted {n}", flush=True)
    all_t = np.concatenate([preds[n]["t_idx"] for n in names])
    all_p = np.concatenate([preds[n]["prob"] for n in names])
    off = np.cumsum([0] + [len(preds[n]["prob"]) for n in names])
    res = {n: {"global": {}, "exclusive_approx": {}} for n in names}
    for thr in GRID:
        sel = all_p >= thr
        exc = target_exclusive(all_t, all_p, sel)
        for j, n in enumerate(names):
            lab = preds[n]["label"].astype(bool)
            res[n]["global"][float(thr)] = macro_f(ents[n], lab, sel[off[j]:off[j + 1]], pops[n][2])
            res[n]["exclusive_approx"][float(thr)] = macro_f(ents[n], lab, exc[off[j]:off[j + 1]], pops[n][2])
    out = {"model": args.model, "results": res}
    for dec in ("global", "exclusive_approx"):
        thr = max(res["calibration_holdout"][dec], key=res["calibration_holdout"][dec].get)
        best_dev = max(res["dev_eval"][dec], key=res["dev_eval"][dec].get)
        out[f"selected_{dec}"] = {"threshold": thr, "calibration_holdout": res["calibration_holdout"][dec][thr],
                                  "dev_eval": res["dev_eval"][dec][thr], "dev_best_thr": best_dev,
                                  "dev_best": res["dev_eval"][dec][best_dev]}
        print(f"SELECTED {dec}: thr={thr} calib={res['calibration_holdout'][dec][thr]:.6f} "
              f"dev={res['dev_eval'][dec][thr]:.6f} (dev-oracle thr {best_dev}: {res['dev_eval'][dec][best_dev]:.6f})",
              flush=True)
        sel = all_p >= thr
        if dec != "global":
            sel = target_exclusive(all_t, all_p, sel)
        j = names.index("dev_eval")
        lab = preds["dev_eval"]["label"].astype(bool)
        write_entity_file(mdir / f"entity_dev_{dec}.parquet", pops["dev_eval"][0], ents["dev_eval"], lab,
                          sel[off[j]:off[j + 1]], pops["dev_eval"][2])
        p = sel[off[j]:off[j + 1]]
        tp = int((p & lab).sum())
        out[f"selected_{dec}"]["dev_pair_precision"] = tp / max(int(p.sum()), 1)
        out[f"selected_{dec}"]["dev_pair_recall"] = tp / int(pops["dev_eval"][2].sum())
    with open(mdir / "eval_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
