"""Phase-3 candidate-pruning check (mission spec section 18).

Measures, on the B004 four-route `development` candidate artifact
(experiments/blocking/B004/union_candidates.parquet, 228,440,608 rows), the
effect of a deterministic upstream pruning rule intended to bound the row
count the decisive matcher must feature-engineer and score:

    keep a candidate iff:
        route_exact_name_address OR route_numeric_exact_set
        OR forward_rank <= FWD_CAP OR reverse_rank <= REV_CAP

Exact/numeric-route evidence is always kept in full (cheap, high-specificity,
never rank-capped). Only the word-TF-IDF forward/reverse routes are capped.
Reports candidate rows before/after, link recall before/after, and the exact
oracle Amazon macro F0.5 (truth intersected with the pruned candidate set,
zero false positives) via the real frozen evaluator, for each candidate cap
configuration, restricted to the same `development` population B004 itself
was built on (never touches `validation`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import load_train_data  # noqa: E402
from evaluation import evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_PATH = PROJECT_ROOT / "experiments" / "blocking" / "B004" / "union_candidates.parquet"
OUT_PATH = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_3" / "pruning_check_results.json"

CONFIGS = [
    {"name": "no_pruning", "fwd_cap": 999, "rev_cap": 999},
    {"name": "fwd20_rev5", "fwd_cap": 20, "rev_cap": 5},
    {"name": "fwd10_rev3", "fwd_cap": 10, "rev_cap": 3},
]


def main() -> None:
    t0 = time.time()
    data = load_train_data()
    dev_ids = set(data.source1.loc[data.source1["split"] == "development", "entity_id"])
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_ids}
    n_true_links = sum(len(t) for t in truth_by_s1.values())
    print(f"[pruning] loaded truth for {len(truth_by_s1):,} development entities, "
          f"{n_true_links:,} true links, in {time.time()-t0:.1f}s")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW cand AS SELECT * FROM read_parquet('{B004_PATH.as_posix()}')")
    total_rows = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    print(f"[pruning] total candidate rows: {total_rows:,}")

    results = {}
    for cfg in CONFIGS:
        t1 = time.time()
        fwd_cap, rev_cap = cfg["fwd_cap"], cfg["rev_cap"]
        where = (
            f"route_exact_name_address OR route_numeric_exact_set "
            f"OR (forward_rank IS NOT NULL AND forward_rank <= {fwd_cap}) "
            f"OR (reverse_rank IS NOT NULL AND reverse_rank <= {rev_cap})"
        )
        n_rows = con.execute(f"SELECT COUNT(*) FROM cand WHERE {where}").fetchone()[0]

        # Oracle prediction per S1: true targets that survive pruning.
        oracle_df = con.execute(
            f"""
            SELECT s1_entity_id, list(target_entity_id) AS cand_targets
            FROM cand WHERE {where}
            GROUP BY s1_entity_id
            """
        ).fetchdf()
        cand_by_s1 = dict(zip(oracle_df["s1_entity_id"], oracle_df["cand_targets"]))

        oracle_pred = {}
        for s1, truth in truth_by_s1.items():
            cands = set(cand_by_s1.get(s1, ()))
            oracle_pred[s1] = truth & cands

        # link recall: fraction of true links present in the pruned candidate set
        n_recalled = sum(len(v) for v in oracle_pred.values())
        link_recall = n_recalled / n_true_links if n_true_links else float("nan")

        result = evaluate(truth_by_s1, oracle_pred, keep_entity_scores=False)

        results[cfg["name"]] = {
            "fwd_cap": fwd_cap,
            "rev_cap": rev_cap,
            "n_rows": n_rows,
            "row_reduction_pct": 100.0 * (1 - n_rows / total_rows),
            "link_recall": link_recall,
            "oracle_macro_f05": result.macro_f_beta,
            "oracle_singleton_accuracy": result.singleton_accuracy,
            "oracle_non_singleton_f05": result.non_singleton_macro_f_beta,
            "runtime_s": time.time() - t1,
        }
        print(f"[pruning] {cfg['name']}: rows={n_rows:,} ({results[cfg['name']]['row_reduction_pct']:.1f}% cut) "
              f"link_recall={link_recall:.4f} oracle_f05={result.macro_f_beta:.6f} "
              f"took {time.time()-t1:.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"total_rows": total_rows, "n_true_links": n_true_links, "configs": results}, fh, indent=2)
    print(f"[pruning] wrote {OUT_PATH}, total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
