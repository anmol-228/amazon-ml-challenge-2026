"""EXP-M002: hard-negative-augmented GBDT matcher, trained on the P3
(unpruned B004) candidate architecture (see run_phase3_prune_label_v2.py /
run_phase3_pruning_check2.py for the pruning-relaxation evidence and
run_phase3_features_v2.py for the (leakage-safety-fixed) feature pipeline).

Evaluation population bug already fixed here from the start (the bug found
by independent review in the M001 script, see DEC-036): `dev_eval`/
`calibration_holdout` macro F0.5 is computed over the FULL split population
from `phase3_split_v1.tsv`, not just S1 entities with a surviving candidate
row, with missing entities scored as empty predictions.
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

from data_io import TRAIN_PATHS, load_ground_truth, parse_matched_ids  # noqa: E402
from evaluation import evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FEATURES_DIR = PROJECT_ROOT / "experiments" / "phase3" / "features_v2"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model_v2"
SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"

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
    df = pq.read_table(FEATURES_DIR / f"{split_name}.parquet").to_pandas()
    for col in FEATURE_COLUMNS:
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.int8)
    return df


def full_split_ids(split_name: str) -> set[str]:
    split_df = pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)
    return set(split_df.loc[split_df["phase3_split"] == split_name, "source1_entity_id"])


def real_truth_by_s1(s1_ids: set[str]) -> dict[str, frozenset]:
    gt = load_ground_truth(TRAIN_PATHS["ground_truth"])
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]
    truth = {row.source1_entity_id: frozenset(parse_matched_ids(row.matched_entity_ids)) for row in gt.itertuples()}
    for s1 in s1_ids:
        truth.setdefault(s1, frozenset())
    return truth


def build_pred_sets_full(full_ids: set[str], df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, set]:
    pred: dict[str, set] = {s1: set() for s1 in full_ids}
    accepted = df.loc[scores >= threshold, ["s1_entity_id", "target_entity_id"]]
    for s1, tid in zip(accepted["s1_entity_id"], accepted["target_entity_id"]):
        if s1 in pred:
            pred[s1].add(tid)
    return pred


def main() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[train_v2] loading matcher_train features...")
    train_df = load_features("matcher_train")
    print(f"[train_v2] matcher_train: {len(train_df):,} rows, {train_df['label'].sum():,} positive "
          f"({time.time()-t0:.1f}s)")

    n = len(train_df)
    cut = int(n * 0.92)
    tr, internal_eval = train_df.iloc[:cut], train_df.iloc[cut:]
    X_tr, y_tr = tr[FEATURE_COLUMNS], tr["label"].astype(int)
    X_ev, y_ev = internal_eval[FEATURE_COLUMNS], internal_eval["label"].astype(int)

    print("[train_v2] training LightGBM...")
    t1 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=700,
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
    print(f"[train_v2] LightGBM trained in {time.time()-t1:.1f}s, best_iteration={model.best_iteration_}")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    print("[train_v2] feature importances:", json.dumps(importances, indent=2))

    model.booster_.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)

    dev_full_ids = full_split_ids("dev_eval")
    calib_full_ids = full_split_ids("calibration_holdout")

    print("[train_v2] loading dev_eval features...")
    dev_df = load_features("dev_eval")
    dev_scores = model.predict_proba(dev_df[FEATURE_COLUMNS])[:, 1]
    dev_truth = real_truth_by_s1(dev_full_ids)
    n_dev_missing = len(dev_full_ids - set(dev_df["s1_entity_id"].unique()))

    pred05 = build_pred_sets_full(dev_full_ids, dev_df, dev_scores, 0.5)
    result05 = evaluate(dev_truth, pred05, keep_entity_scores=False)
    print(f"[train_v2] dev_eval @ threshold=0.5: macro_f05={result05.macro_f_beta:.6f} "
          f"precision={result05.mean_precision} recall={result05.mean_recall}")

    print("[train_v2] loading calibration_holdout features...")
    calib_df = load_features("calibration_holdout")
    calib_scores = model.predict_proba(calib_df[FEATURE_COLUMNS])[:, 1]
    calib_truth = real_truth_by_s1(calib_full_ids)
    n_calib_missing = len(calib_full_ids - set(calib_df["s1_entity_id"].unique()))

    grid_results = []
    best = {"threshold": None, "macro_f05": -1.0}
    for thr in THRESHOLD_GRID:
        pred = build_pred_sets_full(calib_full_ids, calib_df, calib_scores, float(thr))
        result = evaluate(calib_truth, pred, keep_entity_scores=False)
        grid_results.append({
            "threshold": float(thr), "macro_f05": result.macro_f_beta,
            "precision": result.mean_precision, "recall": result.mean_recall,
        })
        print(f"[calibrate_v2] threshold={thr:.2f}: macro_f05={result.macro_f_beta:.6f} "
              f"precision={result.mean_precision} recall={result.mean_recall}")
        if result.macro_f_beta > best["macro_f05"]:
            best = {"threshold": float(thr), "macro_f05": result.macro_f_beta}

    print(f"[calibrate_v2] BEST threshold={best['threshold']} macro_f05={best['macro_f05']:.6f}")

    pred_dev_final = build_pred_sets_full(dev_full_ids, dev_df, dev_scores, best["threshold"])
    result_dev_final = evaluate(dev_truth, pred_dev_final, keep_entity_scores=False)
    print(f"[calibrate_v2] dev_eval @ selected threshold={best['threshold']}: "
          f"macro_f05={result_dev_final.macro_f_beta:.6f}")

    summary = {
        "n_matcher_train_rows": int(len(train_df)),
        "n_matcher_train_positive": int(train_df["label"].sum()),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else int(model.n_estimators),
        "feature_importances": importances,
        "n_dev_eval_entities_with_zero_candidates": n_dev_missing,
        "n_calibration_holdout_entities_with_zero_candidates": n_calib_missing,
        "dev_eval_at_0.5": {
            "macro_f05": result05.macro_f_beta, "precision": result05.mean_precision,
            "recall": result05.mean_recall, "singleton_acc": result05.singleton_accuracy,
            "non_singleton_f05": result05.non_singleton_macro_f_beta,
        },
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

    print(f"[train_v2] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
