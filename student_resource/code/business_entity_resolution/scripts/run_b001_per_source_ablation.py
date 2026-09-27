"""EXP-B001 per-source forward retrieval ablation (review checkpoint item 4).

Question: does retrieving S1->S2 and S1->S3 as two SEPARATE top-k retrieval
universes, then unioning, materially outperform the existing global top-k
over S2+S3 combined? Global top-k risks one source's rows consuming most of
an entity's k slots, suppressing candidates from the other source -- exactly
the kind of thing that would hurt complete true-link coverage, since ~80% of
S1 entities have true matches in *both* S2 and S3 (docs/DATASET_AUDIT.md).

This is an ablation, not an architecture change: it reuses the EXACT SAME
fitted word-TF-IDF vectors EXP-B000 produced (no refit), and only compares
against the EXISTING global forward_word_k50.parquet (no rerun of the
global route). `target_ids`/`dev_s1_ids` are reconstructed directly from
`load_train_data()` rather than reading the 252MB cache manifest -- this is
safe because `run_b000_preflight.py` built the cached matrices' row order
from the exact same deterministic `load_train_data()` + concatenation
(`source2 ids + source3 ids`), so the reconstruction here is guaranteed to
match, not merely assumed to.

Metrics use the scalable join/groupby path (src/blocking_metrics.py), not
per-entity Python sets, since candidate tables here are still tens of
millions of rows.

Run: .venv/Scripts/python.exe scripts/run_b001_per_source_ablation.py
Precondition: run_b000_preflight.py has been run (cache/ populated) and
run_b001_b001r_word.py has produced forward_word_k50.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, write_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    candidate_count_per_s1,
    compute_blocking_metrics_scalable,
    link_found_mask,
    route_incremental_value_scalable,
    truth_edges_dataframe,
)
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_routes import forward_route  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
OUT_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001_PER_SOURCE"

K_GRID = [5, 10, 20]
BATCH_SIZE = 1500


def main() -> None:
    print("[per-source] loading train data + reconstructing target id/order...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    truth_edges = truth_edges_dataframe(truth_by_s1)

    s2_ids = data.source2["entity_id"].tolist()
    s3_ids = data.source3["entity_id"].tolist()
    n_s2, n_s3 = len(s2_ids), len(s3_ids)

    print("[per-source] loading cached s1_dev_matrix + target_matrix (reused, not refit)...")
    s1_dev_matrix = sparse.load_npz(CACHE_DIR / "s1_dev_word_matrix.npz").tocsr()
    target_matrix = sparse.load_npz(CACHE_DIR / "target_word_matrix.npz").tocsr()
    # Cache row order is [all S2 rows][all S3 rows], built by run_b000_preflight.py
    # from the identical data.source2/data.source3 concatenation used here.
    target_matrix_s2 = target_matrix[:n_s2]
    target_matrix_s3 = target_matrix[n_s2:]
    assert target_matrix_s2.shape[0] == n_s2
    assert target_matrix_s3.shape[0] == n_s3

    max_k = max(K_GRID)
    print(f"[per-source] retrieving S1->S2 top-{max_k}...")
    with ResourceMonitor() as mon_s2:
        fwd_s2 = forward_route(
            dev_ids, s1_dev_matrix, s2_ids, target_matrix_s2, k=max_k, batch_size=BATCH_SIZE,
            route_col="route_forward_word_s2", rank_col="forward_rank", score_col="forward_score",
        )
    print(f"[per-source] S1->S2 done in {mon_s2.report.wall_seconds:.1f}s, peak {mon_s2.report.peak_rss_mb:.0f}MB, rows={len(fwd_s2):,}")

    print(f"[per-source] retrieving S1->S3 top-{max_k}...")
    with ResourceMonitor() as mon_s3:
        fwd_s3 = forward_route(
            dev_ids, s1_dev_matrix, s3_ids, target_matrix_s3, k=max_k, batch_size=BATCH_SIZE,
            route_col="route_forward_word_s3", rank_col="forward_rank", score_col="forward_score",
        )
    print(f"[per-source] S1->S3 done in {mon_s3.report.wall_seconds:.1f}s, peak {mon_s3.report.peak_rss_mb:.0f}MB, rows={len(fwd_s3):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(fwd_s2, OUT_DIR / f"forward_s2_k{max_k}.parquet")
    write_candidates(fwd_s3, OUT_DIR / f"forward_s3_k{max_k}.parquet")

    print("[per-source] loading existing GLOBAL forward_word_k50.parquet for comparison (no rerun)...")
    global_full = read_candidates(B001_DIR / "forward_word_k50.parquet")[
        ["s1_entity_id", "target_entity_id", "forward_rank"]
    ]

    results: dict = {"k_grid": {}}

    for k in K_GRID:
        print(f"[per-source] k={k}: computing comparison...")
        sub_s2 = fwd_s2[fwd_s2["forward_rank"] <= k][["s1_entity_id", "target_entity_id"]]
        sub_s3 = fwd_s3[fwd_s3["forward_rank"] <= k][["s1_entity_id", "target_entity_id"]]
        # S2 and S3 id spaces are disjoint (S2-/S3- prefixes), so the
        # per-source union is a plain concat -- no dedup/intersection needed
        # between the two source-specific tables themselves.
        per_source_union = pd.concat([sub_s2, sub_s3], axis=0, ignore_index=True)

        global_sub = global_full[global_full["forward_rank"] <= k][["s1_entity_id", "target_entity_id"]]

        with ResourceMonitor() as mon_metrics:
            per_source_mask = link_found_mask(truth_edges, per_source_union)
            per_source_counts = candidate_count_per_s1(per_source_union, dev_ids)
            per_source_m = compute_blocking_metrics_scalable(
                dev_ids, truth_edges, per_source_mask, per_source_counts, n_s2_total=n_s2, n_s3_total=n_s3
            )

            global_mask = link_found_mask(truth_edges, global_sub)
            global_counts = candidate_count_per_s1(global_sub, dev_ids)
            global_m = compute_blocking_metrics_scalable(
                dev_ids, truth_edges, global_mask, global_counts, n_s2_total=n_s2, n_s3_total=n_s3
            )

            # unique value each direction contributes over the other
            delta_per_source_over_global = route_incremental_value_scalable(
                truth_edges, global_mask, global_mask | per_source_mask
            )
            delta_global_over_per_source = route_incremental_value_scalable(
                truth_edges, per_source_mask, per_source_mask | global_mask
            )

        results["k_grid"][str(k)] = {
            "per_source_union": per_source_m.as_dict(),
            "global": global_m.as_dict(),
            "unique_value_of_per_source_over_global": delta_per_source_over_global,
            "unique_value_of_global_over_per_source": delta_global_over_per_source,
            "metrics_resource": mon_metrics.report.as_dict(),
        }
        print(f"[per-source] k={k}: per_source_recall={per_source_m.link_recall:.4f} "
              f"global_recall={global_m.link_recall:.4f} "
              f"per_source_coverage={per_source_m.complete_true_link_coverage_non_singleton:.4f} "
              f"global_coverage={global_m.complete_true_link_coverage_non_singleton:.4f} "
              f"per_source_unique_links={delta_per_source_over_global['new_true_links_rescued']:,} "
              f"global_unique_links={delta_global_over_per_source['new_true_links_rescued']:,}")

    results["retrieval_resource"] = {
        "s1_to_s2": mon_s2.report.as_dict(),
        "s1_to_s3": mon_s3.report.as_dict(),
    }
    write_manifest(results, OUT_DIR / "per_source_ablation_results.json")
    print(f"[per-source] done. Results written to {OUT_DIR / 'per_source_ablation_results.json'}")


if __name__ == "__main__":
    main()
