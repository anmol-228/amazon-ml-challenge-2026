"""Phase 2.5 Part A -- exact B004 oracle Amazon macro F0.5 ceiling.

Oracle prediction per S1 entity = truth ∩ B004 candidate set. This is
evaluation only: no matcher is trained, no threshold is tuned, no
`validation` touch occurs. The oracle prediction is run through the
canonical, frozen evaluator (`src/evaluation.py::evaluate`, DEC-010) itself
-- not a reimplementation -- and cross-checked against an independently
derived closed-form arithmetic identity as an integrity check:

  - truth empty (singleton)      -> oracle score is ALWAYS 1.0 (oracle
    prediction is empty by construction: truth ∩ anything-when-truth-
    is-empty = empty, so no oracle false positive is possible on a true
    singleton, regardless of candidate exposure).
  - truth non-empty, TP=0        -> oracle score is ALWAYS 0.0.
  - truth non-empty, TP>0        -> oracle precision is ALWAYS 1.0 (the
    oracle prediction IS exactly the recovered true positives, by
    construction, so it can never contain a false positive), so
    F0.5 = 1.25*recall / (0.25 + recall) exactly, where
    recall = TP / |truth|.

This closed form requires no materialized prediction dict at all and is
used here purely as an independent arithmetic check on the real evaluator
call, per the review checkpoint's own request.

Run: .venv/Scripts/python.exe scripts/run_b004_oracle_f05.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_identity import target_source_of  # noqa: E402
from blocking_io import read_candidates, write_manifest  # noqa: E402
from blocking_metrics import link_found_mask, truth_edges_dataframe  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from evaluation import evaluate  # noqa: E402
from validation_split import match_bucket  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"
PHASE25_DIR = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_2_5"


def main() -> None:
    print("[ORACLE] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}

    truth_edges = truth_edges_dataframe(truth_by_s1)
    print(f"[ORACLE] {len(dev_ids):,} dev S1 entities, {len(truth_edges):,} true links")

    print("[ORACLE] loading B004 union (column-selected) + computing found_mask...")
    union_df = pd.read_parquet(
        B004_DIR / "union_candidates.parquet", engine="pyarrow",
        columns=["s1_entity_id", "target_entity_id"],
    )
    with ResourceMonitor() as mon_mask:
        found_mask = link_found_mask(truth_edges, union_df)
    del union_df
    print(f"[ORACLE] found_mask computed in {mon_mask.report.wall_seconds:.1f}s, "
          f"peak {mon_mask.report.peak_rss_mb:.0f}MB")

    print("[ORACLE] building oracle prediction dict (truth intersect candidates, per entity)...")
    te = truth_edges.copy()
    te["_found"] = found_mask
    found_only = te[te["_found"]]
    oracle_pred_by_s1: dict[str, list[str]] = (
        found_only.groupby("s1_entity_id")["target_entity_id"].apply(list).to_dict()
    )

    print("[ORACLE] running the canonical frozen evaluator (src/evaluation.py::evaluate)...")
    with ResourceMonitor() as mon_eval:
        result = evaluate(truth_by_s1, oracle_pred_by_s1, beta=0.5, keep_entity_scores=True)
    print(f"[ORACLE] evaluator run in {mon_eval.report.wall_seconds:.1f}s, peak {mon_eval.report.peak_rss_mb:.0f}MB")

    print(f"[ORACLE] ORACLE_B004_MACRO_F0.5 = {result.macro_f_beta:.6f}")
    print(f"[ORACLE]   singleton_accuracy={result.singleton_accuracy}, "
          f"non_singleton_macro_f_beta={result.non_singleton_macro_f_beta}")

    # ---- independent closed-form arithmetic check --------------------------
    print("[ORACLE] independently verifying via closed-form arithmetic...")
    found_count_by_s1 = found_only.groupby("s1_entity_id").size().to_dict()
    truth_size_by_s1 = {s1: len(t) for s1, t in truth_by_s1.items()}
    closed_form_scores = []
    for s1 in dev_ids:
        truth_size = truth_size_by_s1.get(s1, 0)
        if truth_size == 0:
            closed_form_scores.append(1.0)
            continue
        tp = found_count_by_s1.get(s1, 0)
        if tp == 0:
            closed_form_scores.append(0.0)
            continue
        recall = tp / truth_size
        closed_form_scores.append(1.25 * recall / (0.25 + recall))
    closed_form_macro = float(np.mean(closed_form_scores))
    print(f"[ORACLE] closed-form macro F0.5 = {closed_form_macro:.6f} "
          f"(evaluator gave {result.macro_f_beta:.6f}; "
          f"match={'YES' if abs(closed_form_macro - result.macro_f_beta) < 1e-9 else 'NO -- MISMATCH'})")

    # ---- tiny synthetic sanity test -----------------------------------------
    print("[ORACLE] running tiny synthetic sanity test...")
    syn_truth = {"A": frozenset({"x", "y"}), "B": frozenset(), "C": frozenset({"z"})}
    syn_pred = {"A": ["x"], "B": [], "C": []}  # A: TP=1/2 (recall .5), B: singleton correct, C: TP=0
    syn_result = evaluate(syn_truth, syn_pred, beta=0.5, keep_entity_scores=True)
    expected_A = 1.25 * 0.5 / (0.25 + 0.5)
    expected = (expected_A + 1.0 + 0.0) / 3
    print(f"[ORACLE] synthetic: evaluator={syn_result.macro_f_beta:.6f} expected={expected:.6f} "
          f"match={'YES' if abs(syn_result.macro_f_beta - expected) < 1e-9 else 'NO -- MISMATCH'}")

    # ---- breakdowns ----------------------------------------------------------
    print("[ORACLE] computing breakdowns (country, multiplicity, source, complete-coverage)...")
    country_by_s1 = dict(zip(data.source1["entity_id"], data.source1["country"]))
    scores_df = pd.DataFrame(
        {
            "s1_entity_id": [es.entity_id for es in result.entity_scores],
            "f_beta": [es.f_beta for es in result.entity_scores],
            "truth_size": [es.truth_size for es in result.entity_scores],
            "pred_size": [es.pred_size for es in result.entity_scores],
            "tp": [es.tp for es in result.entity_scores],
        }
    )
    scores_df["country"] = scores_df["s1_entity_id"].map(country_by_s1)
    scores_df["match_bucket"] = scores_df["truth_size"].map(match_bucket)
    scores_df["complete_coverage"] = (scores_df["tp"] == scores_df["truth_size"]) & (scores_df["truth_size"] > 0)

    def target_sources_for(s1):
        t = truth_by_s1.get(s1, frozenset())
        if not t:
            return "singleton"
        srcs = {target_source_of(tid) for tid in t}
        return "both" if len(srcs) > 1 else next(iter(srcs))

    scores_df["source_involvement"] = scores_df["s1_entity_id"].map(target_sources_for)

    by_country = scores_df.groupby("country")["f_beta"].mean().to_dict()
    by_bucket = scores_df.groupby("match_bucket")["f_beta"].mean().to_dict()
    by_source = scores_df.groupby("source_involvement")["f_beta"].mean().to_dict()
    non_singleton = scores_df[scores_df["truth_size"] > 0]
    by_complete = non_singleton.groupby("complete_coverage")["f_beta"].mean().to_dict()

    results = {
        "oracle_macro_f05": result.macro_f_beta,
        "closed_form_macro_f05": closed_form_macro,
        "closed_form_matches_evaluator": abs(closed_form_macro - result.macro_f_beta) < 1e-9,
        "synthetic_sanity_test_passed": abs(syn_result.macro_f_beta - expected) < 1e-9,
        "singleton_accuracy": result.singleton_accuracy,
        "non_singleton_macro_f_beta": result.non_singleton_macro_f_beta,
        "n_entities": result.n_entities,
        "n_true_singletons": result.n_true_singletons,
        "n_true_non_singletons": result.n_true_non_singletons,
        "by_country": by_country,
        "by_match_bucket": by_bucket,
        "by_source_involvement": by_source,
        "by_complete_coverage_non_singleton": {str(k): v for k, v in by_complete.items()},
    }
    PHASE25_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(results, PHASE25_DIR / "b004_oracle_f05.json")
    print(f"[ORACLE] done. Results written to {PHASE25_DIR / 'b004_oracle_f05.json'}")


if __name__ == "__main__":
    main()
