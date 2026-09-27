"""Corrected re-evaluation + recalibration (fixes a confirmed bug found by
independent review, see docs/research/ or EXPERIMENT_LOG.md entry for this
fix): `run_phase3_train_model.py` computed `dev_eval`/`calibration_holdout`
macro F0.5 over only the S1 entities that still had >=1 surviving pruned
candidate row, instead of every S1 entity actually assigned to that split in
`phase3_split_v1.tsv`. Any split entity with zero surviving candidates
(pruned away entirely, or never retrieved by any route) was silently dropped
from both the truth and prediction population, inflating the reported score
whenever such entities exist. This script does NOT retrain the model (the
training data / GBDT itself was independently confirmed correct) -- it only
recomputes evaluation/calibration using the correct, complete population.
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
FEATURES_DIR = PROJECT_ROOT / "experiments" / "phase3" / "features"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model"
SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"

with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as fh:
    FEATURE_COLUMNS = json.load(fh)

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
    """Prediction population is `full_ids` (every S1 actually in the split),
    not just S1s present in `df` (this is exactly the bug being fixed)."""
    pred: dict[str, set] = {s1: set() for s1 in full_ids}
    accepted = df.loc[scores >= threshold, ["s1_entity_id", "target_entity_id"]]
    for s1, tid in zip(accepted["s1_entity_id"], accepted["target_entity_id"]):
        if s1 in pred:
            pred[s1].add(tid)
    return pred


def main() -> None:
    t0 = time.time()
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))

    dev_full_ids = full_split_ids("dev_eval")
    calib_full_ids = full_split_ids("calibration_holdout")
    print(f"[recalibrate] full dev_eval population: {len(dev_full_ids):,}, "
          f"full calibration_holdout population: {len(calib_full_ids):,}")

    dev_df = load_features("dev_eval")
    dev_candidate_ids = set(dev_df["s1_entity_id"].unique())
    n_dev_missing = len(dev_full_ids - dev_candidate_ids)
    print(f"[recalibrate] dev_eval S1 entities with ZERO surviving pruned candidates: "
          f"{n_dev_missing:,} / {len(dev_full_ids):,} ({100*n_dev_missing/len(dev_full_ids):.2f}%)")
    dev_scores = booster.predict(dev_df[FEATURE_COLUMNS])
    dev_truth = real_truth_by_s1(dev_full_ids)

    calib_df = load_features("calibration_holdout")
    calib_candidate_ids = set(calib_df["s1_entity_id"].unique())
    n_calib_missing = len(calib_full_ids - calib_candidate_ids)
    print(f"[recalibrate] calibration_holdout S1 entities with ZERO surviving pruned candidates: "
          f"{n_calib_missing:,} / {len(calib_full_ids):,} ({100*n_calib_missing/len(calib_full_ids):.2f}%)")
    calib_scores = booster.predict(calib_df[FEATURE_COLUMNS])
    calib_truth = real_truth_by_s1(calib_full_ids)

    dev_diag = {}
    for thr in (0.5,):
        pred = build_pred_sets_full(dev_full_ids, dev_df, dev_scores, thr)
        result = evaluate(dev_truth, pred, keep_entity_scores=False)
        dev_diag[str(thr)] = {
            "macro_f05": result.macro_f_beta, "precision": result.mean_precision,
            "recall": result.mean_recall, "singleton_acc": result.singleton_accuracy,
            "non_singleton_f05": result.non_singleton_macro_f_beta,
        }
        print(f"[recalibrate] CORRECTED dev_eval @ threshold={thr}: macro_f05={result.macro_f_beta:.6f} "
              f"precision={result.mean_precision} recall={result.mean_recall}")

    grid_results = []
    best = {"threshold": None, "macro_f05": -1.0}
    for thr in THRESHOLD_GRID:
        pred = build_pred_sets_full(calib_full_ids, calib_df, calib_scores, float(thr))
        result = evaluate(calib_truth, pred, keep_entity_scores=False)
        grid_results.append({
            "threshold": float(thr), "macro_f05": result.macro_f_beta,
            "precision": result.mean_precision, "recall": result.mean_recall,
        })
        print(f"[recalibrate] CORRECTED calibration threshold={thr:.2f}: macro_f05={result.macro_f_beta:.6f} "
              f"precision={result.mean_precision} recall={result.mean_recall}")
        if result.macro_f_beta > best["macro_f05"]:
            best = {"threshold": float(thr), "macro_f05": result.macro_f_beta}

    print(f"[recalibrate] CORRECTED BEST threshold={best['threshold']} macro_f05={best['macro_f05']:.6f}")

    pred_dev_final = build_pred_sets_full(dev_full_ids, dev_df, dev_scores, best["threshold"])
    result_dev_final = evaluate(dev_truth, pred_dev_final, keep_entity_scores=False)
    print(f"[recalibrate] CORRECTED dev_eval @ selected threshold={best['threshold']}: "
          f"macro_f05={result_dev_final.macro_f_beta:.6f}")

    with open(MODEL_DIR / "training_summary.json", encoding="utf-8") as fh:
        old_summary = json.load(fh)

    summary = {
        **old_summary,
        "EVALUATION_BUG_FIX_NOTE": (
            "dev_eval/calibration_holdout metrics below are CORRECTED to use the full "
            "phase3_split_v1.tsv population (including S1 entities with zero surviving "
            "pruned candidates, scored as empty predictions), fixing a bug found by "
            "independent review where such entities were silently excluded from both "
            "truth and prediction, inflating the originally reported macro F0.5."
        ),
        "n_dev_eval_entities_with_zero_candidates": n_dev_missing,
        "n_calibration_holdout_entities_with_zero_candidates": n_calib_missing,
        "dev_eval_at_0.5_CORRECTED": dev_diag["0.5"],
        "dev_eval_at_selected_threshold_CORRECTED": {
            "threshold": best["threshold"], "macro_f05": result_dev_final.macro_f_beta,
            "precision": result_dev_final.mean_precision, "recall": result_dev_final.mean_recall,
            "singleton_acc": result_dev_final.singleton_accuracy,
            "non_singleton_f05": result_dev_final.non_singleton_macro_f_beta,
        },
        "calibration_holdout_grid_CORRECTED": grid_results,
        "selected_threshold_CORRECTED": best["threshold"],
        "calibration_holdout_macro_f05_at_selected_CORRECTED": best["macro_f05"],
        "recalibration_wall_time_s": time.time() - t0,
    }
    with open(MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(MODEL_DIR / "selected_threshold.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": best["threshold"]}, fh, indent=2)

    print(f"[recalibrate] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
