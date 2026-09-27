"""AUDIT-S2: falsification tests for the stage-2 collective-feature leakage concern
raised in the independent R2 release review (audit item 2).

Reuses saved R2 stage-1 probabilities (m004_r2/oof_matcher_train.npy, m003_r2/pred_*.npy);
no retrieval / feature rebuild. Same stage-2 hyperparameters, sample, P_MIN and entity-grouped
early stopping as train_stage2.py; NOT tuned.

--mode strict   : stage-2 TRAINING features computed from matcher_train rows only (no dev /
                  calibration probability enters any training feature). Eval features for
                  calibration/dev are built over the full table exactly as in train_stage2.py
                  (and as at test time, where the table is the whole test corpus).
--mode probsrc  : sensitivity of the promoted stage-2 model (m004_r2) to the source of the
                  calibration/dev stage-1 probabilities: replace main-model probs by the mean of
                  the two cross-fit fold models (same model strength as the matcher_train OOF
                  competitors). Stage-2 model unchanged.
Thresholds are selected on calibration_holdout; dev_eval reported. Paired bootstrap vs the
promoted run's dev entity file (entity_dev_stage2_global.parquet).

Usage: python -u audit_stage2_strict.py --mode strict --out m004_r2_strict
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
from train_m003 import GRID, PARAMS, ROOT, entity_codes, population, predict_file  # noqa: E402
from train_stage2 import P_MIN, read_cols  # noqa: E402

M = ROOT / "experiments" / "m003"
FEAT = M / "feat_dev_r2"
NAMES = ["matcher_train", "calibration_holdout", "dev_eval"]


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
    A = pd.read_parquet(M / "m004_r2" / "entity_dev_stage2_global.parquet").set_index("s1_entity_id")
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
    ap.add_argument("--mode", choices=["strict", "probsrc"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-estimators", type=int, default=1500)
    args = ap.parse_args()
    t0 = time.time()
    mdir = M / args.out
    mdir.mkdir(parents=True, exist_ok=False)
    cols = json.load(open(M / "m003_r2" / "feature_columns.json", encoding="utf-8"))
    ids = {n: load_ids(n) for n in NAMES}
    prob = {"matcher_train": np.load(M / "m004_r2" / "oof_matcher_train.npy"),
            "calibration_holdout": np.load(M / "m003_r2" / "pred_calibration_holdout.npy"),
            "dev_eval": np.load(M / "m003_r2" / "pred_dev_eval.npy")}
    for n in NAMES:
        assert len(prob[n]) == len(ids[n][0]), n
    pops = {n: population(n, FEAT) for n in ("calibration_holdout", "dev_eval")}
    ents = {n: entity_codes(ids[n][0], pops[n][1]) for n in pops}
    lab = {n: ids[n][2].astype(bool) for n in NAMES}
    out = {"mode": args.mode}

    if args.mode == "probsrc":
        folds = [lgb.Booster(model_file=str(M / "m004_r2" / f"stage1_fold{k}.txt")) for k in range(2)]
        for n in ("calibration_holdout", "dev_eval"):
            pa = predict_file(FEAT / f"features_{n}.parquet", folds[0], cols)
            pb = predict_file(FEAT / f"features_{n}.parquet", folds[1], cols)
            assert np.array_equal(pa["s1_idx"], ids[n][0]) and np.array_equal(pa["t_idx"], ids[n][1])
            prob[n] = ((pa["prob"] + pb["prob"]) / 2).astype(np.float32)
            np.save(mdir / f"pred_foldmean_{n}.npy", prob[n])
            print(f"[audit] fold-mean stage-1 probs {n} ({time.time()-t0:.0f}s)", flush=True)
        b2 = lgb.Booster(model_file=str(M / "m004_r2" / "stage2_model.txt"))
        feats2 = json.load(open(M / "m004_r2" / "stage2_feature_columns.json", encoding="utf-8"))
        tag = "probsrc_foldmean"
    else:
        # strict: training features from matcher_train rows only
        s_tr, t_tr, _ = ids["matcher_train"]
        p_tr = prob["matcher_train"].astype(np.float32)
        s2_tr = stage2_features(s_tr, t_tr, p_tr)
        keep_tr = p_tr >= P_MIN
        feats2 = STAGE2_FEATURES + [c for c in cols if c not in STAGE2_FEATURES]
        base = read_cols(FEAT / "features_matcher_train.parquet", cols + ["s1_idx", "label"], keep_tr)
        X = pd.concat([s2_tr.iloc[np.flatnonzero(keep_tr)].reset_index(drop=True), base[cols]], axis=1)
        y = base["label"].astype(int).to_numpy()
        grp = np.array([zlib.crc32(str(s).encode()) % 100 for s in base["s1_idx"].to_numpy()])
        del base, s2_tr
        print(f"[audit] strict training rows {len(y):,} ({time.time()-t0:.0f}s)", flush=True)
        model = lgb.LGBMClassifier(n_estimators=args.n_estimators, **PARAMS)
        model.fit(X[feats2][grp >= 8], y[grp >= 8], eval_set=[(X[feats2][grp < 8], y[grp < 8])],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)])
        best = int(model.best_iteration_ or args.n_estimators)
        model.booster_.save_model(str(mdir / "stage2_model.txt"), num_iteration=best)
        b2 = lgb.Booster(model_file=str(mdir / "stage2_model.txt"))
        out["best_iter"] = best
        print(f"[audit] strict stage-2 best_iter={best} ({time.time()-t0:.0f}s)", flush=True)
        del X, y
        tag = "strict_train_mt_only"

    # eval features over the full table (as train_stage2.py / test time)
    off = np.cumsum([0] + [len(prob[n]) for n in NAMES])
    s_all = np.concatenate([ids[n][0] for n in NAMES])
    t_all = np.concatenate([ids[n][1] for n in NAMES])
    p_all = np.concatenate([prob[n] for n in NAMES]).astype(np.float32)
    s2 = stage2_features(s_all, t_all, p_all)
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
    out["dev_eval_at_0.69"] = curves["dev_eval"][0.69]
    print(f"[audit] {tag} SELECTED thr={thr} calib={out['calibration_holdout']:.6f} dev={out['dev_eval']:.6f}", flush=True)
    compare(tag, p2["dev_eval"], lab["dev_eval"], thr, pops, ents, mdir, out)
    json.dump({"summary": out, "curves": curves}, open(mdir / "summary.json", "w", encoding="utf-8"), indent=1)
    print(f"[audit] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
