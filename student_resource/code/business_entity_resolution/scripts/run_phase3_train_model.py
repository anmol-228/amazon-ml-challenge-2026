"""EXP-M001/M002-lite: GBDT baseline matcher + EXP-M003-lite threshold calibration.

Time-boxed for submission #1 (see mission fallback ladder): trains ONE
LightGBM binary classifier on `matcher_train` (random negatives, sampled in
run_phase3_prune_label.py at ~8 negatives per positive -- not yet the fuller
EXP-M002 collision-structure-targeted hard-negative mining, which is
POST_SUBMISSION_HIGH_PRIORITY), evaluates out-of-sample on `dev_eval` via the
real frozen evaluator (never touches `validation`), then grid-searches a
single global acceptance threshold on `calibration_holdout` (mission section
21's "accept every candidate with score >= threshold" simple policy, not full
isotonic regression -- also a time-boxed scope cut) to maximize macro F0.5.
The chosen threshold is a single confirmatory read of `calibration_holdout`
only; `dev_eval` is reported for diagnosis but is not the split threshold
selection is fit against, and `validation` is never touched at all in this
script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluation import evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FEATURES_DIR = PROJECT_ROOT / "experiments" / "phase3" / "features"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model"

FEATURE_COLUMNS = [
    "route_exact_name_address", "route_forward_word", "route_reverse_word", "route_numeric_exact_set",
    "forward_rank", "reverse_rank", "forward_score", "reverse_score",
    "n_routes_support", "best_rank",
    "name_exact", "address_exact", "name_ratio", "address_ratio", "name_jw", "address_jw",
    "name_token_jaccard", "name_len_ratio", "address_len_ratio",
    "numeric_jaccard", "numeric_conflict", "numeric_all_equal",
    "is_source2", "country_agree", "target_name_rarity",
]

THRESHOLD_GRID = np.round(np.arange(0.30, 0.96, 0.02), 2)


def load_features(split_name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{split_name}.parquet"
    df = pq.read_table(path).to_pandas()
    for col in FEATURE_COLUMNS:
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.int8)
    return df


def build_pred_sets(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, set]:
    accepted = df.loc[scores >= threshold, ["s1_entity_id", "target_entity_id"]]
    pred: dict[str, set] = {s1: set() for s1 in df["s1_entity_id"].unique()}
    for s1, tid in zip(accepted["s1_entity_id"], accepted["target_entity_id"]):
        pred[s1].add(tid)
    return pred


def real_truth_by_s1(s1_ids: set[str]) -> dict[str, frozenset]:
    """Authoritative ground truth restricted to the given S1 population,
    loaded directly from train_ground_truth.tsv (not reconstructed from a
    pruned candidate set), so a true link outside the pruned candidate
    universe still correctly counts as a miss rather than being silently
    dropped from the truth set itself."""
    from data_io import TRAIN_PATHS, load_ground_truth, parse_matched_ids

    gt = load_ground_truth(TRAIN_PATHS["ground_truth"])
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]
    truth = {row.source1_entity_id: frozenset(parse_matched_ids(row.matched_entity_ids)) for row in gt.itertuples()}
    for s1 in s1_ids:
        truth.setdefault(s1, frozenset())
    return truth


def main() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[train] loading matcher_train features...")
    train_df = load_features("matcher_train")
    print(f"[train] matcher_train: {len(train_df):,} rows, {train_df['label'].sum():,} positive "
          f"({time.time()-t0:.1f}s)")

    # small internal eval split (last 8% by row order, which is already
    # randomized by DuckDB's random()-based negative sampling) for early
    # stopping only -- never used for any reported metric.
    n = len(train_df)
    cut = int(n * 0.92)
    tr, internal_eval = train_df.iloc[:cut], train_df.iloc[cut:]

    X_tr, y_tr = tr[FEATURE_COLUMNS], tr["label"].astype(int)
    X_ev, y_ev = internal_eval[FEATURE_COLUMNS], internal_eval["label"].astype(int)

    print("[train] training LightGBM...")
    t1 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_ev, y_ev)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=50)],
    )
    print(f"[train] LightGBM trained in {time.time()-t1:.1f}s, best_iteration={model.best_iteration_}")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    print("[train] feature importances:", json.dumps(importances, indent=2))

    model.booster_.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)

    # ---- EXP-M001 out-of-sample dev_eval diagnostic ----------------------
    print("[train] loading dev_eval features...")
    dev_df = load_features("dev_eval")
    dev_scores = model.predict_proba(dev_df[FEATURE_COLUMNS])[:, 1]
    dev_truth = real_truth_by_s1(set(dev_df["s1_entity_id"].unique()))

    dev_diag = {}
    for thr in (0.5,):
        pred = build_pred_sets(dev_df, dev_scores, thr)
        result = evaluate(dev_truth, pred, keep_entity_scores=False)
        dev_diag[str(thr)] = {
            "macro_f05": result.macro_f_beta, "precision": result.mean_precision,
            "recall": result.mean_recall, "singleton_acc": result.singleton_accuracy,
            "non_singleton_f05": result.non_singleton_macro_f_beta,
        }
        print(f"[train] dev_eval @ threshold={thr}: macro_f05={result.macro_f_beta:.6f} "
              f"precision={result.mean_precision} recall={result.mean_recall}")

    # ---- EXP-M003-lite: threshold grid search on calibration_holdout -----
    print("[train] loading calibration_holdout features...")
    calib_df = load_features("calibration_holdout")
    calib_scores = model.predict_proba(calib_df[FEATURE_COLUMNS])[:, 1]
    calib_truth = real_truth_by_s1(set(calib_df["s1_entity_id"].unique()))

    grid_results = []
    best = {"threshold": None, "macro_f05": -1.0}
    for thr in THRESHOLD_GRID:
        pred = build_pred_sets(calib_df, calib_scores, float(thr))
        result = evaluate(calib_truth, pred, keep_entity_scores=False)
        grid_results.append({
            "threshold": float(thr), "macro_f05": result.macro_f_beta,
            "precision": result.mean_precision, "recall": result.mean_recall,
        })
        print(f"[calibrate] threshold={thr:.2f}: macro_f05={result.macro_f_beta:.6f} "
              f"precision={result.mean_precision} recall={result.mean_recall}")
        if result.macro_f_beta > best["macro_f05"]:
            best = {"threshold": float(thr), "macro_f05": result.macro_f_beta}

    print(f"[calibrate] BEST threshold={best['threshold']} macro_f05={best['macro_f05']:.6f}")

    # confirmatory: apply the selected threshold to dev_eval too, for reporting
    pred_dev_final = build_pred_sets(dev_df, dev_scores, best["threshold"])
    result_dev_final = evaluate(dev_truth, pred_dev_final, keep_entity_scores=False)
    print(f"[calibrate] dev_eval @ selected threshold={best['threshold']}: "
          f"macro_f05={result_dev_final.macro_f_beta:.6f}")

    summary = {
        "n_matcher_train_rows": int(len(train_df)),
        "n_matcher_train_positive": int(train_df["label"].sum()),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else int(model.n_estimators),
        "feature_importances": importances,
        "dev_eval_at_0.5": dev_diag["0.5"],
        "dev_eval_at_selected_threshold": {
            "threshold": best["threshold"], "macro_f05": result_dev_final.macro_f_beta,
            "precision": result_dev_final.mean_precision, "recall": result_dev_final.mean_recall,
            "singleton_acc": result_dev_final.singleton_accuracy,
            "non_singleton_f05": result_dev_final.non_singleton_macro_f_beta,
        },
        "calibration_holdout_grid": grid_results,
        "selected_threshold": best["threshold"],
        "calibration_holdout_macro_f05_at_selected": best["macro_f05"],
        "wall_time_s": time.time() - t0,
    }
    with open(MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(MODEL_DIR / "selected_threshold.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": best["threshold"]}, fh, indent=2)

    print(f"[train] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
