"""EXP-M004: stage-2 collective model on top of a stage-1 M003 model.

1. Stage-1 probabilities over the whole train-corpus candidate table:
   matcher_train rows -> 2-fold entity-level cross-fitted models (same
   hyperparameters and iteration count as the stage-1 main model);
   dev_eval / calibration_holdout rows -> stage-1 main model (as at test time).
2. Exact exclusive-decoder evaluation for stage 1 over that full table.
3. Stage-2 features (src/stage2.py) over the full table, stage-2 LightGBM trained
   on matcher_train rows with p1 >= P_MIN (entity-grouped early stopping), then
   threshold/decoder selection on calibration_holdout, report on dev_eval.

Usage: python -u train_stage2.py --feat feat_dev_r1 --stage1 m003_a --out m004_a
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

from fast_eval import macro_f, target_exclusive  # noqa: E402
from stage2 import STAGE2_FEATURES, stage2_features  # noqa: E402
from train_m003 import (GRID, PARAMS, ROOT, augment_idf, entity_codes, fold_of, load_train_sample,  # noqa: E402
                        population, predict_file, write_entity_file)

P_MIN = 0.02


def read_cols(path: Path, cols, row_mask: np.ndarray) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts, off = [], 0
    for i in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        m = row_mask[off:off + n]
        if m.any():
            parts.append(pf.read_row_group(i, columns=cols).to_pandas()[m])
        off += n
    return pd.concat(parts, ignore_index=True)


def evaluate(names, preds, pops, ents, prob_all, t_all, off):
    res = {n: {"global": {}, "exclusive": {}} for n in pops}
    for thr in GRID:
        sel = prob_all >= thr
        exc = target_exclusive(t_all, prob_all, sel)
        for n in pops:
            j = names.index(n)
            lab = preds[n]["label"].astype(bool)
            res[n]["global"][float(thr)] = macro_f(ents[n], lab, sel[off[j]:off[j + 1]], pops[n][2])
            res[n]["exclusive"][float(thr)] = macro_f(ents[n], lab, exc[off[j]:off[j + 1]], pops[n][2])
    return res


def select(res, tag, out, names, preds, pops, ents, prob_all, t_all, off, mdir):
    summ = {}
    for dec in ("global", "exclusive"):
        thr = max(res["calibration_holdout"][dec], key=res["calibration_holdout"][dec].get)
        summ[dec] = {"threshold": thr, "calibration_holdout": res["calibration_holdout"][dec][thr],
                     "dev_eval": res["dev_eval"][dec][thr]}
        print(f"[m004] {tag} SELECTED {dec}: thr={thr} calib={summ[dec]['calibration_holdout']:.6f} "
              f"dev={summ[dec]['dev_eval']:.6f}", flush=True)
        sel = prob_all >= thr
        if dec == "exclusive":
            sel = target_exclusive(t_all, prob_all, sel)
        j = names.index("dev_eval")
        write_entity_file(mdir / f"entity_dev_{tag}_{dec}.parquet", pops["dev_eval"][0], ents["dev_eval"],
                          preds["dev_eval"]["label"].astype(bool), sel[off[j]:off[j + 1]], pops["dev_eval"][2])
    out[tag] = summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-estimators", type=int, default=1500)
    ap.add_argument("--aug-idf", default="", help="lo,hi: training-time IDF-scale augmentation (off by default)")
    args = ap.parse_args()
    t0 = time.time()
    feat_dir = ROOT / "experiments" / "m003" / args.feat
    s1dir = ROOT / "experiments" / "m003" / args.stage1
    mdir = ROOT / "experiments" / "m003" / args.out
    mdir.mkdir(parents=True, exist_ok=False)
    cols = json.load(open(s1dir / "feature_columns.json", encoding="utf-8"))
    main_booster = lgb.Booster(model_file=str(s1dir / "lgbm_model.txt"))
    n_iter = main_booster.current_iteration()

    # 1. cross-fitted stage-1 probabilities for matcher_train
    tr = load_train_sample(feat_dir / "features_matcher_train.parquet", cols, 0.25)
    tr = augment_idf(tr, args.aug_idf, seed=11)
    fo = fold_of(tr["s1_idx"].to_numpy(), 2)
    fold_models = []
    for k in range(2):
        m = lgb.LGBMClassifier(n_estimators=n_iter, **PARAMS)
        m.fit(tr.loc[fo != k, cols], tr.loc[fo != k, "label"].astype(int))
        m.booster_.save_model(str(mdir / f"stage1_fold{k}.txt"))
        fold_models.append(m.booster_)
        print(f"[m004] stage-1 fold model {k} ({n_iter} iters) ({time.time()-t0:.0f}s)", flush=True)
    del tr
    names = ["matcher_train", "calibration_holdout", "dev_eval"]
    preds = {"matcher_train": predict_file(feat_dir / "features_matcher_train.parquet", fold_models, cols, fold_mod=2)}
    for n in ("calibration_holdout", "dev_eval"):
        p_saved = s1dir / f"pred_{n}.npy"
        pr = predict_file(feat_dir / f"features_{n}.parquet", main_booster, cols) if not p_saved.exists() else None
        if pr is None:
            pf = pq.read_table(feat_dir / f"features_{n}.parquet", columns=["s1_idx", "t_idx", "label"])
            pr = {"s1_idx": pf["s1_idx"].to_numpy(), "t_idx": pf["t_idx"].to_numpy(),
                  "label": pf["label"].to_numpy(), "prob": np.load(p_saved)}
        preds[n] = pr
    np.save(mdir / "oof_matcher_train.npy", preds["matcher_train"]["prob"])
    print(f"[m004] stage-1 probabilities ready ({time.time()-t0:.0f}s)", flush=True)

    off = np.cumsum([0] + [len(preds[n]["prob"]) for n in names])
    s_all = np.concatenate([preds[n]["s1_idx"] for n in names]).astype(np.int64)
    t_all = np.concatenate([preds[n]["t_idx"] for n in names]).astype(np.int64)
    p_all = np.concatenate([preds[n]["prob"] for n in names]).astype(np.float32)
    pops = {n: population(n, feat_dir) for n in ("calibration_holdout", "dev_eval")}
    ents = {n: entity_codes(preds[n]["s1_idx"], pops[n][1]) for n in pops}
    out = {"stage1": args.stage1, "stage1_iterations": n_iter}

    # 2. exact exclusive evaluation of stage 1
    res1 = evaluate(names, preds, pops, ents, p_all, t_all, off)
    select(res1, "stage1", out, names, preds, pops, ents, p_all, t_all, off, mdir)

    # 3. stage-2
    s2 = stage2_features(s_all, t_all, p_all)
    keep = p_all >= P_MIN
    print(f"[m004] stage-2 features; rows with p1>={P_MIN}: {int(keep.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    feats2 = STAGE2_FEATURES + [c for c in cols if c not in STAGE2_FEATURES]
    tr_mask = keep[off[0]:off[1]]
    base = read_cols(feat_dir / "features_matcher_train.parquet", cols + ["s1_idx", "label"], tr_mask)
    X = pd.concat([s2.iloc[np.flatnonzero(keep[:off[1]])].reset_index(drop=True), base[cols]], axis=1)
    X = augment_idf(X, args.aug_idf, seed=13)
    y = base["label"].astype(int).to_numpy()
    grp = np.array([zlib.crc32(str(s).encode()) % 100 for s in base["s1_idx"].to_numpy()])
    del base
    model = lgb.LGBMClassifier(n_estimators=args.n_estimators, **PARAMS)
    model.fit(X[feats2][grp >= 8], y[grp >= 8], eval_set=[(X[feats2][grp < 8], y[grp < 8])],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)])
    best = int(model.best_iteration_ or args.n_estimators)
    model.booster_.save_model(str(mdir / "stage2_model.txt"), num_iteration=best)
    b2 = lgb.Booster(model_file=str(mdir / "stage2_model.txt"))
    json.dump(feats2, open(mdir / "stage2_feature_columns.json", "w", encoding="utf-8"), indent=1)
    print(f"[m004] stage-2 model best_iter={best} ({time.time()-t0:.0f}s)", flush=True)
    del X, y

    p2_all = np.zeros_like(p_all)
    for j, n in enumerate(names):
        if n == "matcher_train":
            continue
        m = keep[off[j]:off[j + 1]]
        base = read_cols(feat_dir / f"features_{n}.parquet", cols, m)
        Xn = pd.concat([s2.iloc[off[j] + np.flatnonzero(m)].reset_index(drop=True), base[cols]], axis=1)
        p2 = np.zeros(off[j + 1] - off[j], dtype=np.float32)
        p2[m] = b2.predict(Xn[feats2]).astype(np.float32)
        p2_all[off[j]:off[j + 1]] = p2
        np.save(mdir / f"pred2_{n}.npy", p2)
    res2 = evaluate(names, preds, pops, ents, p2_all, t_all, off)
    select(res2, "stage2", out, names, preds, pops, ents, p2_all, t_all, off, mdir)
    out["importance_stage2_top"] = sorted(zip(feats2, b2.feature_importance("gain").tolist()), key=lambda x: -x[1])[:25]
    json.dump({"summary": out, "stage1_curves": res1, "stage2_curves": res2},
              open(mdir / "summary.json", "w", encoding="utf-8"), indent=1)
    print(f"[m004] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
