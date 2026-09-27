"""EXP-D001: decision-layer experiments on saved M003/M004 probabilities.

Settings are always selected on calibration_holdout and reported on dev_eval.
Writes per-entity dev files for the blind judge into experiments/m003/decoder_lab/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_eval import macro_f  # noqa: E402
from train_m003 import GRID, ROOT, entity_codes, population, write_entity_file  # noqa: E402

FEAT = ROOT / "experiments" / "m003" / "feat_dev_r1"
M3 = ROOT / "experiments" / "m003" / "m003_a"
M4 = ROOT / "experiments" / "m003" / "m004_a"
OUT = ROOT / "experiments" / "m003" / "decoder_lab"


def expected_f_select(ent: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    """Per-entity plug-in expected-F0.5 set choice; k=0 allowed."""
    p = np.clip(p.astype(np.float64), 0.0, 1.0 - 1e-9)
    order = np.lexsort((-p, ent))
    e, ps = ent[order], p[order]
    first = np.ones(len(e), dtype=bool)
    first[1:] = e[1:] != e[:-1]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.append(starts, len(e)))
    gstart = np.repeat(starts, sizes)
    k = np.arange(len(e)) - gstart + 1
    cs = np.cumsum(ps)
    base = np.repeat(np.concatenate([[0.0], cs[starts[1:] - 1]]), sizes)
    cs_g = cs - base
    total = np.repeat(np.add.reduceat(ps, starts), sizes)
    ef_k = 1.25 * cs_g / (0.25 * total + k)
    ef0 = np.exp(np.add.reduceat(np.log1p(-ps), starts))
    best = np.maximum.reduceat(ef_k, starts)
    # first k achieving the group max
    is_best = ef_k >= np.repeat(best, sizes) - 1e-12
    kbest = np.minimum.reduceat(np.where(is_best, k, np.iinfo(np.int64).max), starts)
    kstar = np.where(best >= lam * ef0, kbest, 0)
    sel_sorted = k <= np.repeat(kstar, sizes)
    sel = np.zeros(len(p), dtype=bool)
    sel[order] = sel_sorted
    return sel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = {}
    for n in ("calibration_holdout", "dev_eval"):
        tb = pq.read_table(FEAT / f"features_{n}.parquet", columns=["s1_idx", "t_idx", "label"])
        ids, pidx, truth = population(n, FEAT)
        d[n] = {"ent": entity_codes(tb["s1_idx"].to_numpy(), pidx), "lab": tb["label"].to_numpy().astype(bool),
                "truth": truth, "ids": ids, "p3": np.load(M3 / f"pred_{n}.npy"), "p4": np.load(M4 / f"pred2_{n}.npy")}
    res = {}

    def sweep(name, key_fn):
        curves = {n: {float(t): macro_f(d[n]["ent"], d[n]["lab"], key_fn(n) >= t, d[n]["truth"]) for t in GRID}
                  for n in d}
        thr = max(curves["calibration_holdout"], key=curves["calibration_holdout"].get)
        res[name] = {"threshold": thr, "calib": curves["calibration_holdout"][thr], "dev": curves["dev_eval"][thr]}
        write_entity_file(OUT / f"entity_dev_{name}.parquet", d["dev_eval"]["ids"], d["dev_eval"]["ent"],
                          d["dev_eval"]["lab"], key_fn("dev_eval") >= thr, d["dev_eval"]["truth"])
        print(name, res[name], flush=True)

    sweep("m004_global", lambda n: d[n]["p4"])
    for a in (0.25, 0.5):
        sweep(f"blend_a{a}", lambda n, a=a: a * d[n]["p3"] + (1 - a) * d[n]["p4"])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    c = d["calibration_holdout"]
    iso.fit(c["p4"], c["lab"].astype(np.float64))
    for n in d:
        d[n]["p4c"] = iso.predict(d[n]["p4"]).astype(np.float64)
    ef = {}
    for lam in (0.8, 0.9, 1.0, 1.1, 1.25):
        ef[lam] = {n: macro_f(d[n]["ent"], d[n]["lab"], expected_f_select(d[n]["ent"], d[n]["p4c"], lam), d[n]["truth"])
                   for n in d}
        print("expF lam", lam, ef[lam], flush=True)
    lam = max(ef, key=lambda x: ef[x]["calibration_holdout"])
    res["expF_iso"] = {"lambda": lam, "calib": ef[lam]["calibration_holdout"], "dev": ef[lam]["dev_eval"],
                       "note": "isotonic fit on calibration_holdout; calib score is in-sample for the calibrator"}
    dv = d["dev_eval"]
    write_entity_file(OUT / "entity_dev_expF_iso.parquet", dv["ids"], dv["ent"], dv["lab"],
                      expected_f_select(dv["ent"], dv["p4c"], lam), dv["truth"])
    print("expF_iso", res["expF_iso"], flush=True)
    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
