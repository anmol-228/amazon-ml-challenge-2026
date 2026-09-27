"""Extended pruning grid (mission spec section 7, post-M001 upgrade): P2
(fwd<=30, rev<=5) and P3 (fwd<=50, rev<=5, i.e. no forward cut at all since
B004's own forward_rank already maxes at 50), on top of the already-measured
P0 (fwd<=10, rev<=3, currently in production) and P1 (fwd<=20, rev<=5) from
run_phase3_pruning_check.py's earlier run
(experiments/blocking/PHASE_3/pruning_check_results.json). Same methodology:
oracle Amazon macro F0.5 (truth intersected with pruned candidates, zero
false positives) via the real evaluator, on the full `development` B004
union, never touching `validation` or test data.
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
OUT_PATH = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_3" / "pruning_check_results_extended.json"

CONFIGS = [
    {"name": "P2_fwd30_rev5", "fwd_cap": 30, "rev_cap": 5},
    {"name": "P3_fwd50_rev5", "fwd_cap": 50, "rev_cap": 5},
]


def main() -> None:
    t0 = time.time()
    data = load_train_data()
    dev_ids = set(data.source1.loc[data.source1["split"] == "development", "entity_id"])
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_ids}
    n_true_links = sum(len(t) for t in truth_by_s1.values())
    print(f"[pruning2] loaded truth for {len(truth_by_s1):,} entities, {n_true_links:,} true links "
          f"in {time.time()-t0:.1f}s")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW cand AS SELECT * FROM read_parquet('{B004_PATH.as_posix()}')")
    total_rows = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]

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
        oracle_df = con.execute(
            f"SELECT s1_entity_id, list(target_entity_id) AS cand_targets FROM cand WHERE {where} GROUP BY s1_entity_id"
        ).fetchdf()
        cand_by_s1 = dict(zip(oracle_df["s1_entity_id"], oracle_df["cand_targets"]))

        oracle_pred = {}
        n_recalled = 0
        for s1, truth in truth_by_s1.items():
            cands = set(cand_by_s1.get(s1, ()))
            hit = truth & cands
            oracle_pred[s1] = hit
            n_recalled += len(hit)
        link_recall = n_recalled / n_true_links if n_true_links else float("nan")

        result = evaluate(truth_by_s1, oracle_pred, keep_entity_scores=False)
        results[cfg["name"]] = {
            "fwd_cap": fwd_cap, "rev_cap": rev_cap, "n_rows": n_rows,
            "row_reduction_pct": 100.0 * (1 - n_rows / total_rows),
            "link_recall": link_recall,
            "oracle_macro_f05": result.macro_f_beta,
            "runtime_s": time.time() - t1,
        }
        print(f"[pruning2] {cfg['name']}: rows={n_rows:,} link_recall={link_recall:.4f} "
              f"oracle_f05={result.macro_f_beta:.6f} took {time.time()-t1:.1f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"total_rows": total_rows, "n_true_links": n_true_links, "configs": results}, fh, indent=2)
    print(f"[pruning2] wrote {OUT_PATH}, total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
