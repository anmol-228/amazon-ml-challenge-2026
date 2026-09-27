"""M003: LightGBM matcher on R1 candidates + vectorized features.

Training: matcher_train only (all positives + hard negatives + seeded random
share of easy negatives); early stopping on an entity-grouped internal holdout.

Evaluation population: FULL dev_eval / calibration_holdout from phase3_split_v1
(entities without candidates count as empty predictions).

Decoders:
  global    : accept p >= thr
  exclusive : accept p >= thr, then each target kept only for its highest-p S1.
              At test time exclusivity competes across ALL S1, so here it is
              applied over the full train-corpus candidate table: matcher_train
              rows get 2-fold cross-fitted out-of-fold probabilities, dev_eval /
              calibration_holdout rows get the main model's probabilities.
Thresholds are selected on calibration_holdout and reported on dev_eval.

Usage: python -u train_m003.py --feat FEATDIR --out MODELDIR [--drop a,b] [--no-crossfit]
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_io import load_ground_truth  # noqa: E402
from fast_eval import macro_f, per_entity_f, target_exclusive  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SPLIT_PATH = ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"
NON_FEATURES = {"s1_idx", "t_idx", "label", "split", "s1_entity_id", "target_entity_id", "phase3_split", "country"}
GRID = np.round(np.arange(0.20, 0.981, 0.01), 2)
PARAMS = dict(learning_rate=0.05, num_leaves=127, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, objective="binary", n_jobs=-1, random_state=42, verbose=-1)


def fold_of(s1_idx: np.ndarray, mod: int) -> np.ndarray:
    # deterministic entity-level hash (Knuth multiplicative) -> fold id
    return ((s1_idx.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1000003) % np.uint64(mod)).astype(np.int64)


ABS_IDF = ["name_idf_shared_max", "name_idf_s_only", "name_idf_t_only",
           "addr_idf_shared_max", "addr_idf_s_only", "addr_idf_t_only"]


def augment_idf(df: pd.DataFrame, spec: str, seed: int) -> pd.DataFrame:
    """Training-time IDF-scale augmentation (EXP-R3AUG): multiply the absolute IDF-weighted features of
    each row by one factor ~ U(lo, hi), so the model cannot rely on the exact corpus-dependent IDF
    scale. spec = "lo,hi" (every row) or "half:lo,hi" (a random half of the rows keep factor 1);
    empty = off. Inference is unchanged."""
    if not spec:
        return df
    half = spec.startswith("half:")
    lo, hi = (float(x) for x in spec.replace("half:", "").split(","))
    rng = np.random.default_rng(seed)
    f = rng.uniform(lo, hi, len(df)).astype(np.float32)
    if half:
        f[rng.random(len(df)) < 0.5] = 1.0
    for c in ABS_IDF:
        if c in df.columns:
            df[c] = (df[c].to_numpy() * f).astype(np.float32)
    return df


def load_train_sample(path: Path, cols, easy_frac: float) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(42)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=["s1_idx", "t_idx", "label"] + cols).to_pandas()
        hard = ((df["r1_fwd_rank"] <= 5) | (df["r1_rev_rank"] <= 2) | (df["t_rank"] <= 2) | (df["s_rank"] <= 5)).to_numpy()
        keep = df["label"].to_numpy() | hard | (rng.random(len(df)) < easy_frac)
        parts.append(df[keep])
    return pd.concat(parts, ignore_index=True)


def predict_file(path: Path, models, cols, fold_mod=None):
    """Stream-predict a feature file. models: one booster, or list indexed by fold."""
    pf = pq.ParquetFile(path)
    out = {"s1_idx": [], "t_idx": [], "label": [], "prob": []}
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=["s1_idx", "t_idx", "label"] + cols).to_pandas()
        if fold_mod is None:
            p = models.predict(df[cols])
        else:
            p = np.empty(len(df))
            fo = fold_of(df["s1_idx"].to_numpy(), fold_mod)
            for k, m in enumerate(models):
                mk = fo == k
                if mk.any():
                    p[mk] = m.predict(df.loc[mk, cols])
        out["s1_idx"].append(df["s1_idx"].to_numpy())
        out["t_idx"].append(df["t_idx"].to_numpy())
        out["label"].append(df["label"].to_numpy())
        out["prob"].append(p.astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def population(split_name: str, feat_dir: Path):
    split = pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)
    ids = split.loc[split["phase3_split"] == split_name, "source1_entity_id"].tolist()
    s1_ids = pq.read_table(feat_dir / "s1_ids.parquet")["entity_id"]
    import pyarrow as pa
    idx = pc.index_in(pa.array(ids, pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False)
    assert not np.isnan(idx.astype(float)).any()
    gt = load_ground_truth()
    sizes = dict(zip(gt["source1_entity_id"], [0 if m == "" else m.count(",") + 1 for m in gt["matched_entity_ids"]]))
    truth = np.array([sizes.get(i, 0) for i in ids], dtype=np.int64)
    return np.array(ids, dtype=object), idx.astype(np.int64), truth


def entity_codes(s1_idx_rows: np.ndarray, pop_idx: np.ndarray) -> np.ndarray:
    lut = np.full(int(max(pop_idx.max(), s1_idx_rows.max())) + 1, -1, dtype=np.int64)
    lut[pop_idx] = np.arange(len(pop_idx))
    return lut[s1_idx_rows]


def write_entity_file(path, ids, ent, lab, sel, truth) -> None:
    n = len(ids)
    pd.DataFrame({"s1_entity_id": ids, "f": per_entity_f(ent, lab, sel, truth),
                  "tp": np.bincount(ent[sel & lab], minlength=n), "pred": np.bincount(ent[sel], minlength=n),
                  "truth": truth}).to_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--easy-frac", type=float, default=0.25)
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--no-crossfit", action="store_true")
    ap.add_argument("--aug-idf", default="", help="lo,hi: training-time IDF-scale augmentation (off by default)")
    args = ap.parse_args()
    t0 = time.time()
    feat_dir = ROOT / "experiments" / "m003" / args.feat
    out = ROOT / "experiments" / "m003" / args.out
    out.mkdir(parents=True, exist_ok=False)
    drop = set(x for x in args.drop.split(",") if x)
    schema = pq.ParquetFile(feat_dir / "features_matcher_train.parquet").schema_arrow.names
    cols = [c for c in schema if c not in NON_FEATURES and c not in drop]

    tr = load_train_sample(feat_dir / "features_matcher_train.parquet", cols, args.easy_frac)
    tr = augment_idf(tr, args.aug_idf, seed=7)
    grp = np.array([zlib.crc32(str(s).encode()) % 100 for s in tr["s1_idx"].to_numpy()])
    fit, es = tr[grp >= 8], tr[grp < 8]
    print(f"[m003] sample rows {len(tr):,} (pos {int(tr['label'].sum()):,}); fit {len(fit):,} es {len(es):,}; "
          f"{len(cols)} features ({time.time()-t0:.0f}s)", flush=True)
    model = lgb.LGBMClassifier(n_estimators=args.n_estimators, **PARAMS)
    model.fit(fit[cols], fit["label"].astype(int), eval_set=[(es[cols], es["label"].astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)])
    best_iter = int(model.best_iteration_ or args.n_estimators)
    booster = model.booster_
    booster.save_model(str(out / "lgbm_model.txt"), num_iteration=best_iter)
    booster = lgb.Booster(model_file=str(out / "lgbm_model.txt"))
    with open(out / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump(cols, fh, indent=2)
    imp = sorted(zip(cols, booster.feature_importance("gain").tolist()), key=lambda x: -x[1])
    print(f"[m003] main model best_iter={best_iter} ({time.time()-t0:.0f}s)", flush=True)
    del fit, es

    fold_models = None
    if not args.no_crossfit:
        fo = fold_of(tr["s1_idx"].to_numpy(), 2)
        fold_models = []
        for k in range(2):
            m = lgb.LGBMClassifier(n_estimators=best_iter, **PARAMS)
            m.fit(tr.loc[fo != k, cols], tr.loc[fo != k, "label"].astype(int))
            fold_models.append(m.booster_)
            print(f"[m003] fold model {k} ({time.time()-t0:.0f}s)", flush=True)
    del tr

    preds = {}
    for name in ("calibration_holdout", "dev_eval"):
        preds[name] = predict_file(feat_dir / f"features_{name}.parquet", booster, cols)
        np.save(out / f"pred_{name}.npy", preds[name]["prob"])
        print(f"[m003] predicted {name}: {len(preds[name]['prob']):,} rows ({time.time()-t0:.0f}s)", flush=True)
    if fold_models is not None:
        preds["matcher_train"] = predict_file(feat_dir / "features_matcher_train.parquet", fold_models, cols, fold_mod=2)
        print(f"[m003] OOF matcher_train: {len(preds['matcher_train']['prob']):,} rows ({time.time()-t0:.0f}s)", flush=True)

    # full-corpus table for exclusivity
    names = list(preds)
    all_t = np.concatenate([preds[n]["t_idx"] for n in names])
    all_p = np.concatenate([preds[n]["prob"] for n in names])
    offsets = np.cumsum([0] + [len(preds[n]["prob"]) for n in names])

    summary = {"feat": args.feat, "features": cols, "best_iteration": best_iter, "crossfit": fold_models is not None,
               "importance_gain_top": imp[:30]}
    pops = {n: population(n, feat_dir) for n in ("calibration_holdout", "dev_eval")}
    ents = {n: entity_codes(preds[n]["s1_idx"], pops[n][1]) for n in pops}
    for n in pops:
        assert (ents[n] >= 0).all()
    res = {n: {"global": {}, "exclusive": {}} for n in pops}
    for thr in GRID:
        sel_all = all_p >= thr
        excl_all = target_exclusive(all_t, all_p, sel_all)
        for n in pops:
            lo, hi = offsets[names.index(n)], offsets[names.index(n) + 1]
            lab = preds[n]["label"].astype(bool)
            truth = pops[n][2]
            res[n]["global"][float(thr)] = macro_f(ents[n], lab, sel_all[lo:hi], truth)
            res[n]["exclusive"][float(thr)] = macro_f(ents[n], lab, excl_all[lo:hi], truth)
    for n in pops:
        summary[n] = res[n]
        for dec in ("global", "exclusive"):
            b = max(res[n][dec], key=res[n][dec].get)
            print(f"[m003] {n} {dec}: oracle-thr={b} F0.5={res[n][dec][b]:.6f}", flush=True)
    for dec in ("global", "exclusive"):
        thr = max(res["calibration_holdout"][dec], key=res["calibration_holdout"][dec].get)
        summary[f"selected_{dec}"] = {"threshold": thr, "calibration_holdout": res["calibration_holdout"][dec][thr],
                                      "dev_eval": res["dev_eval"][dec][thr]}
        print(f"[m003] SELECTED {dec}: thr={thr} calib={res['calibration_holdout'][dec][thr]:.6f} "
              f"dev={res['dev_eval'][dec][thr]:.6f}", flush=True)
        sel_all = all_p >= thr
        if dec == "exclusive":
            sel_all = target_exclusive(all_t, all_p, sel_all)
        lo, hi = offsets[names.index("dev_eval")], offsets[names.index("dev_eval") + 1]
        write_entity_file(out / f"entity_dev_{dec}.parquet", pops["dev_eval"][0], ents["dev_eval"],
                          preds["dev_eval"]["label"].astype(bool), sel_all[lo:hi], pops["dev_eval"][2])
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[m003] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
