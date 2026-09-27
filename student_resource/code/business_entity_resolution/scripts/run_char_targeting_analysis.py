"""Phase 2.5 Part D -- observable char-rescue targeting rule analysis.

Question: can the S1 entities whose truth benefits from character-n-gram
retrieval be identified using signals available WITHOUT ground truth (no
oracle gating)? Candidate observable signals, all already computed by
existing routes:

  - best_forward_score: max word-TF-IDF cosine score among this S1's
    EXP-B001 forward candidates (NaN/0 if it got none at all).
  - best_reverse_score: max word-TF-IDF cosine score among this S1's
    EXP-B001R reverse-inverted candidates.
  - b004_candidate_count: how many candidates this S1 currently has from
    the full B004 union (a sparse candidate set from every route combined
    is itself an observable weak-coverage signal).

The "needs_char" LABEL used to evaluate a candidate rule is ground-truth-
derived (an S1 has >=1 missed link with word_cosine_similarity == 0, the
'wildly dissimilar' residual bucket char n-gram specifically targets) --
but that label is used ONLY to measure how well an observable rule would
have done, exactly like EXP-B002's own pilot query selection. It is never
itself the production trigger; the production trigger is the observable
signal only, per the mission's own no-oracle-gating rule.

Run: .venv/Scripts/python.exe scripts/run_char_targeting_analysis.py
Precondition: run_b004_union.py and run_residual_analysis.py have completed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, write_manifest  # noqa: E402
from blocking_metrics import candidate_count_per_s1  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B001R_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001R"
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"
RESIDUAL_DIR = PROJECT_ROOT / "experiments" / "blocking" / "RESIDUAL"
B002_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B002"
PHASE25_DIR = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_2_5"


def main() -> None:
    print("[TARGETING] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)

    print("[TARGETING] computing best_forward_score per S1 (from EXP-B001, no rerun)...")
    with ResourceMonitor() as mon_fwd:
        fwd_scores = pd.read_parquet(
            B001_DIR / "forward_word_k50.parquet", engine="pyarrow",
            columns=["s1_entity_id", "forward_score"],
        )
        best_forward = fwd_scores.groupby("s1_entity_id")["forward_score"].max()
    del fwd_scores
    print(f"[TARGETING] best_forward computed in {mon_fwd.report.wall_seconds:.1f}s, "
          f"peak {mon_fwd.report.peak_rss_mb:.0f}MB")

    print("[TARGETING] computing best_reverse_score per S1 (from EXP-B001R, no rerun)...")
    with ResourceMonitor() as mon_rev:
        rev_scores = pd.read_parquet(
            B001R_DIR / "reverse_word_r5.parquet", engine="pyarrow",
            columns=["s1_entity_id", "reverse_score"],
        )
        best_reverse = rev_scores.groupby("s1_entity_id")["reverse_score"].max()
    del rev_scores
    print(f"[TARGETING] best_reverse computed in {mon_rev.report.wall_seconds:.1f}s, "
          f"peak {mon_rev.report.peak_rss_mb:.0f}MB")

    print("[TARGETING] computing B004 candidate count per S1 (from EXP-B004, no rerun)...")
    with ResourceMonitor() as mon_b004:
        b004_df = pd.read_parquet(
            B004_DIR / "union_candidates.parquet", engine="pyarrow",
            columns=["s1_entity_id", "target_entity_id"],
        )
        b004_counts = candidate_count_per_s1(b004_df, dev_ids)
    del b004_df
    print(f"[TARGETING] B004 counts computed in {mon_b004.report.wall_seconds:.1f}s, "
          f"peak {mon_b004.report.peak_rss_mb:.0f}MB")

    print("[TARGETING] loading residual missed links (word_cosine_similarity labels)...")
    residual_df = read_candidates(RESIDUAL_DIR / "residual_missed_links.parquet")
    wildly_dissimilar = residual_df[residual_df["word_cosine_similarity"] <= 1e-9]
    needs_char_s1 = set(wildly_dissimilar["s1_entity_id"].unique())
    print(f"[TARGETING] {len(needs_char_s1):,} S1 entities have >=1 'wildly dissimilar' "
          f"missed link (the needs_char population, for MEASUREMENT only)")

    print("[TARGETING] assembling per-S1 signal table...")
    signal_df = pd.DataFrame({"s1_entity_id": dev_ids})
    signal_df["best_forward_score"] = signal_df["s1_entity_id"].map(best_forward).fillna(0.0)
    signal_df["best_reverse_score"] = signal_df["s1_entity_id"].map(best_reverse).fillna(0.0)
    signal_df["b004_candidate_count"] = signal_df["s1_entity_id"].map(b004_counts).fillna(0).astype(int)
    signal_df["best_word_score"] = signal_df[["best_forward_score", "best_reverse_score"]].max(axis=1)
    signal_df["needs_char"] = signal_df["s1_entity_id"].isin(needs_char_s1)

    n_needs_char = int(signal_df["needs_char"].sum())
    print(f"[TARGETING] {n_needs_char:,} / {len(signal_df):,} dev S1 entities need char rescue "
          f"({n_needs_char / len(signal_df):.4%})")

    # ---- try a family of simple, single-threshold observable rules ---------
    print("[TARGETING] evaluating threshold rules on best_word_score...")
    candidate_thresholds = np.percentile(
        signal_df["best_word_score"], [10, 20, 30, 40, 50, 60, 70, 80, 90]
    )
    rule_results = []
    for thresh in sorted(set(np.round(candidate_thresholds, 4).tolist()) | {0.0, 0.05, 0.1, 0.15, 0.2, 0.3}):
        flagged = signal_df["best_word_score"] <= thresh
        n_flagged = int(flagged.sum())
        if n_flagged == 0:
            continue
        n_captured = int((flagged & signal_df["needs_char"]).sum())
        rule_results.append(
            {
                "threshold_best_word_score_lte": float(thresh),
                "n_flagged": n_flagged,
                "flagged_fraction": n_flagged / len(signal_df),
                "n_needs_char_captured": n_captured,
                "needs_char_recall_of_rule": n_captured / n_needs_char if n_needs_char else 0.0,
                "precision_of_rule": n_captured / n_flagged,
            }
        )
    rule_results.sort(key=lambda r: r["threshold_best_word_score_lte"])

    for r in rule_results:
        print(f"[TARGETING] threshold<={r['threshold_best_word_score_lte']:.4f}: "
              f"flagged={r['n_flagged']:,} ({r['flagged_fraction']:.2%}) "
              f"needs_char_recall={r['needs_char_recall_of_rule']:.2%} "
              f"precision={r['precision_of_rule']:.2%}")

    # ---- also try a candidate-count-based rule ------------------------------
    print("[TARGETING] evaluating threshold rules on b004_candidate_count...")
    count_thresholds = [1, 3, 5, 10, 20, 30, 50]
    count_rule_results = []
    for thresh in count_thresholds:
        flagged = signal_df["b004_candidate_count"] <= thresh
        n_flagged = int(flagged.sum())
        if n_flagged == 0:
            continue
        n_captured = int((flagged & signal_df["needs_char"]).sum())
        count_rule_results.append(
            {
                "threshold_candidate_count_lte": thresh,
                "n_flagged": n_flagged,
                "flagged_fraction": n_flagged / len(signal_df),
                "n_needs_char_captured": n_captured,
                "needs_char_recall_of_rule": n_captured / n_needs_char if n_needs_char else 0.0,
                "precision_of_rule": n_captured / n_flagged,
            }
        )
        print(f"[TARGETING] count<={thresh}: flagged={n_flagged:,} ({n_flagged/len(signal_df):.2%}) "
              f"needs_char_recall={count_rule_results[-1]['needs_char_recall_of_rule']:.2%} "
              f"precision={count_rule_results[-1]['precision_of_rule']:.2%}")

    # ---- projected char workload for the best-value rule --------------------
    b002_results_path = B002_DIR / "b002_results.json"
    seconds_per_query = None
    if b002_results_path.exists():
        import json

        b002 = json.load(open(b002_results_path))
        n_pilot_queries = b002["n_residual_s1_entities"]
        seconds_per_query = b002["retrieval_resource"]["wall_seconds"] / n_pilot_queries

    # Pick the rule with the best recall/flagged-fraction trade-off: highest
    # needs_char_recall per unit of flagged_fraction among score-based rules.
    best_rule = max(rule_results, key=lambda r: r["needs_char_recall_of_rule"] / max(r["flagged_fraction"], 1e-9))
    if seconds_per_query:
        best_rule["projected_char_retrieval_minutes"] = (
            seconds_per_query * best_rule["n_flagged"]
        ) / 60.0

    results = {
        "n_dev_s1": len(signal_df),
        "n_needs_char_ground_truth_population": n_needs_char,
        "needs_char_fraction": n_needs_char / len(signal_df),
        "score_threshold_rules": rule_results,
        "candidate_count_threshold_rules": count_rule_results,
        "best_score_rule_by_efficiency": best_rule,
        "seconds_per_char_query_from_b002_pilot": seconds_per_query,
    }
    PHASE25_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(results, PHASE25_DIR / "char_targeting_analysis.json")
    print(f"[TARGETING] done. Best efficiency rule: {best_rule}")
    print(f"[TARGETING] Results written to {PHASE25_DIR / 'char_targeting_analysis.json'}")


if __name__ == "__main__":
    main()
