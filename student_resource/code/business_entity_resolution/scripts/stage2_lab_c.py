"""EXP-S2C lab (copy of stage2_lab.py + --corr target-target corroboration features, src/stage2c.py). EXP-S2X lab: stage-2 retrain on the R2 artifacts with extra collective features (src/stage2x.py).

Reuses saved R2 stage-1 probabilities (m004_r2/oof_matcher_train.npy for matcher_train rows, m003_r2/pred_*.npy for
calibration/dev rows) exactly as train_stage2.py would produce them; stage-2 features over the full concatenated table
(as train_stage2.py and test time); same params / P_MIN / entity-grouped early stopping. Threshold on calibration,
report dev_eval, paired bootstrap vs the promoted #5 run (m004_r2 entity file).

Usage: python -u stage2_lab.py --extras s_n_conf_same,s_n_conf_oth,... --out s2x_all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_eval import macro_f, per_entity_f  # noqa: E402
from stage2 import STAGE2_FEATURES, stage2_features  # noqa: E402
from stage2x import (OPENSET_T_FEATURES, STAGE2T_FEATURES, STAGE2X_FEATURES, stage2_twin_features,  # noqa: E402
                     stage2x_features, target_open_set_features)
from train_m003 import GRID, PARAMS, ROOT, entity_codes, population, predict_file  # noqa: E402
from train_stage2 import P_MIN, read_cols  # noqa: E402
from train_m003 import augment_idf  # noqa: E402
from stage2c import STAGE2C_FEATURES, STAGE2R_FEATURES, corroboration_features, rival_anchor_features  # noqa: E402

M = ROOT / "experiments" / "m003"
FEAT = M / "feat_dev_r2"
NAMES = ["matcher_train", "calibration_holdout", "dev_eval"]
REF = "m004_r2"


def load_ids(n):
    t = pq.read_table(FEAT / f"features_{n}.parquet", columns=["s1_idx", "t_idx", "label"])
    return t["s1_idx"].to_numpy().astype(np.int64), t["t_idx"].to_numpy().astype(np.int64), t["label"].to_numpy()


def score_split(b2, feats2, cols, n, s2_rows, keep_rows):
    base = read_cols(FEAT / f"features_{n}.parquet", cols, keep_rows)
    Xn = pd.concat([s2_rows.iloc[np.flatnonzero(keep_rows)].reset_index(drop=True), base[cols]], axis=1)
    p2 = np.zeros(len(keep_rows), dtype=np.float32)
    p2[keep_rows] = b2.predict(Xn[feats2]).astype(np.float32)
    return p2


def evaluate(p2, lab, pops, ents):
    curves = {n: {} for n in pops}
    for thr in GRID:
        for n in pops:
            curves[n][float(thr)] = macro_f(ents[n], lab[n], p2[n] >= thr, pops[n][2])
    thr = max(curves["calibration_holdout"], key=curves["calibration_holdout"].get)
    return curves, thr


def compare(tag, p2dev, labdev, thr, pops, ents, mdir, out):
    ids, _, truth = pops["dev_eval"]
    sel = p2dev >= thr
    ent = ents["dev_eval"]
    f = per_entity_f(ent, labdev, sel, truth)
    n = len(ids)
    B = pd.DataFrame({"s1_entity_id": ids, "f": f, "tp": np.bincount(ent[sel & labdev], minlength=n),
                      "pred": np.bincount(ent[sel], minlength=n), "truth": truth})
    B.to_parquet(mdir / f"entity_dev_{tag}.parquet")
    A = pd.read_parquet(M / REF / "entity_dev_stage2_global.parquet").set_index("s1_entity_id")
    B = B.set_index("s1_entity_id").loc[A.index]
    d = (B["f"] - A["f"]).to_numpy()
    rng = np.random.default_rng(12345)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    S = pd.read_parquet(ROOT / "experiments" / "judge" / "slices.parquet").set_index("s1_entity_id").loc[A.index]
    res = {"promoted_dev": float(A["f"].mean()), "variant_dev": float(B["f"].mean()),
           "delta_variant_minus_promoted": float(d.mean()),
           "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
           "fp_change": int(((B["pred"] - B["tp"]) - (A["pred"] - A["tp"])).sum()),
           "fn_change": int(((B["truth"] - B["tp"]) - (A["truth"] - A["tp"])).sum()),
           "singleton": {"promoted": float(A.loc[A["truth"] == 0, "f"].mean()),
                         "variant": float(B.loc[B["truth"] == 0, "f"].mean())},
           "country": {str(c): {"promoted": float(A["f"][S["country"] == c].mean()),
                                "variant": float(B["f"][S["country"] == c].mean()),
                                "n": int((S["country"] == c).sum())} for c in S["country"].unique()}}
    out[tag] = res
    print(f"[audit] {tag}: {json.dumps(res)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", default=",".join(STAGE2X_FEATURES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-estimators", type=int, default=1500)
    ap.add_argument("--twin", action="store_true", help="add exact-name twin features (src/stage2x.py)")
    ap.add_argument("--corr", action="store_true", help="add target-target corroboration features")
    ap.add_argument("--openset", action="store_true", help="add target ownership-concentration features")
    ap.add_argument("--corr2", action="store_true", help="with --corr: add rival corroboration + anchor reliability")
    ap.add_argument("--no-own-corr", action="store_true", help="with --corr2: do not use the raw S2C features in the model")
    ap.add_argument("--feat", default="feat_dev_r2")
    ap.add_argument("--stage1", default="m003_r2", help="dir with pred_{calibration_holdout,dev_eval}.npy")
    ap.add_argument("--oof", default="m004_r2", help="dir with oof_matcher_train.npy")
    ap.add_argument("--ref", default="m004_r2", help="reference run for the paired comparison")
    ap.add_argument("--aug-idf", default="", help="lo,hi: IDF-scale augmentation of the stage-2 training matrix")
    args = ap.parse_args()
    global FEAT, REF
    FEAT, REF = M / args.feat, args.ref
    extras = [x for x in args.extras.split(",") if x]
    assert all(x in STAGE2X_FEATURES for x in extras), extras
    t0 = time.time()
    mdir = M / args.out
    mdir.mkdir(parents=True, exist_ok=False)
    cols = json.load(open(M / args.stage1 / "feature_columns.json", encoding="utf-8"))
    ids = {n: load_ids(n) for n in NAMES}
    prob = {"matcher_train": np.load(M / args.oof / "oof_matcher_train.npy"),
            "calibration_holdout": np.load(M / args.stage1 / "pred_calibration_holdout.npy"),
            "dev_eval": np.load(M / args.stage1 / "pred_dev_eval.npy")}
    for n in NAMES:
        assert len(prob[n]) == len(ids[n][0]), n
    pops = {n: population(n, FEAT) for n in ("calibration_holdout", "dev_eval")}
    ents = {n: entity_codes(ids[n][0], pops[n][1]) for n in pops}
    lab = {n: ids[n][2].astype(bool) for n in NAMES}
    off = np.cumsum([0] + [len(prob[n]) for n in NAMES])
    s_all = np.concatenate([ids[n][0] for n in NAMES])
    t_all = np.concatenate([ids[n][1] for n in NAMES])
    p_all = np.concatenate([prob[n] for n in NAMES]).astype(np.float32)
    tids = pq.read_table(FEAT / "t_ids.parquet")["entity_id"]
    import pyarrow.compute as pc
    is_s2 = pc.starts_with(tids, "S2-").to_numpy(zero_copy_only=False).astype(np.int64)
    s2 = stage2_features(s_all, t_all, p_all)
    if extras:
        x = stage2x_features(s_all, t_all, p_all, is_s2[t_all])
        s2 = pd.concat([s2, x[extras]], axis=1)
    if args.openset:
        s2 = pd.concat([s2, target_open_set_features(t_all, p_all)], axis=1)
        extras = extras + OPENSET_T_FEATURES
        print(f"[s2x] open-set ownership features added ({time.time()-t0:.0f}s)", flush=True)
    if args.corr:
        ts = pq.read_table(FEAT / "target_strings_v2.parquet")
        t_name = ts["name"].to_numpy(zero_copy_only=False).astype(object)
        t_addr = ts["addr"].to_numpy(zero_copy_only=False).astype(object)
        c = corroboration_features(s_all, t_all, p_all, is_s2[t_all], t_name, t_addr, p_all >= P_MIN)
        s2 = pd.concat([s2, c], axis=1)
        if not args.no_own_corr:
            extras = extras + STAGE2C_FEATURES
        if args.corr2:
            s2 = pd.concat([s2, rival_anchor_features(s_all, t_all, p_all, c)], axis=1)
            extras = extras + STAGE2R_FEATURES
        del ts, t_name, t_addr, c
        print(f"[s2x] corroboration features added ({time.time()-t0:.0f}s)", flush=True)
    if args.twin:
        from normalization_v2 import normalize_name_v2
        sid = pq.read_table(FEAT / "s1_ids.parquet")["entity_id"].to_pandas()
        src1 = pd.read_csv(ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv", sep="	", dtype=str,
                           keep_default_na=False, usecols=["entity_id", "business_name"]).set_index("entity_id")["business_name"]
        name_code = pd.factorize(sid.map(src1).fillna("").map(normalize_name_v2))[0].astype(np.int64)
        s2 = pd.concat([s2, stage2_twin_features(s_all, t_all, p_all, name_code)], axis=1)
        extras = extras + STAGE2T_FEATURES
        print(f"[s2x] twin features added ({time.time()-t0:.0f}s)", flush=True)
    feats2 = STAGE2_FEATURES + extras + [c for c in cols if c not in STAGE2_FEATURES]
    keep = p_all >= P_MIN
    tr_mask = keep[off[0]:off[1]]
    base = read_cols(FEAT / "features_matcher_train.parquet", cols + ["s1_idx", "label"], tr_mask)
    X = pd.concat([s2.iloc[np.flatnonzero(tr_mask)].reset_index(drop=True), base[cols]], axis=1)
    X = augment_idf(X, args.aug_idf, seed=13)
    y = base["label"].astype(int).to_numpy()
    grp = np.array([zlib.crc32(str(v).encode()) % 100 for v in base["s1_idx"].to_numpy()])
    del base
    print(f"[s2x] training rows {len(y):,}; extras {extras} ({time.time()-t0:.0f}s)", flush=True)
    model = lgb.LGBMClassifier(n_estimators=args.n_estimators, **PARAMS)
    model.fit(X[feats2][grp >= 8], y[grp >= 8], eval_set=[(X[feats2][grp < 8], y[grp < 8])],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=250)])
    best = int(model.best_iteration_ or args.n_estimators)
    model.booster_.save_model(str(mdir / "stage2_model.txt"), num_iteration=best)
    json.dump(feats2, open(mdir / "stage2_feature_columns.json", "w", encoding="utf-8"), indent=1)
    b2 = lgb.Booster(model_file=str(mdir / "stage2_model.txt"))
    del X, y
    out = {"extras": extras, "best_iter": best, "args": vars(args)}
    p2 = {}
    for j, n in enumerate(NAMES):
        if n == "matcher_train":
            continue
        rows = s2.iloc[off[j]:off[j + 1]].reset_index(drop=True)
        p2[n] = score_split(b2, feats2, cols, n, rows, p_all[off[j]:off[j + 1]] >= P_MIN)
        np.save(mdir / f"pred2_{n}.npy", p2[n])
    curves, thr = evaluate(p2, lab, pops, ents)
    out["threshold"] = thr
    out["calibration_holdout"] = curves["calibration_holdout"][thr]
    out["dev_eval"] = curves["dev_eval"][thr]
    print(f"[s2x] SELECTED thr={thr} calib={out['calibration_holdout']:.6f} dev={out['dev_eval']:.6f} best_iter={best}", flush=True)
    compare("s2x", p2["dev_eval"], lab["dev_eval"], thr, pops, ents, mdir, out)
    imp = sorted(zip(feats2, b2.feature_importance("gain").tolist()), key=lambda z: -z[1])[:15]
    out["importance_top"] = imp
    json.dump({"summary": out, "curves": curves}, open(mdir / "summary.json", "w", encoding="utf-8"), indent=1)
    print(f"[s2x] top importance: {[(k, round(v)) for k, v in imp[:10]]}", flush=True)
    print(f"[s2x] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
