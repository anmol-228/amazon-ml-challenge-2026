"""Scalable EXP-B001R r-grid metric computation from EXISTING retrieval
artifacts -- does NOT rerun retrieval.

EXP-B001 forward (k=50) and EXP-B001R reverse (r=5) retrieval already
completed successfully and are on disk at
`experiments/blocking/B001/forward_word_k50.parquet` (87,161,478 rows) and
`experiments/blocking/B001R/reverse_word_r5.parquet` (49,433,565 rows).
Only the r-grid metrics computation crashed, because it materialized a
`pandas.concat` + `drop_duplicates` union of those two tables (up to ~136M
rows) once per r value and rebuilt a per-entity dict-of-sets from it every
time. This script replaces that with the scalable join/groupby-based path
in `src/blocking_metrics.py` (`link_found_mask`, `candidate_count_per_s1`,
`union_count_per_s1`, `compute_blocking_metrics_scalable`,
`route_incremental_value_scalable`), verified equal to the original simple
implementation on synthetic data in `tests/test_blocking_infrastructure.py`.
No candidate union table is ever materialized here.

Note on the manifest: this script deliberately does NOT load
`experiments/blocking/cache/word_cache_manifest.json` (252 MB, dominated by
the two full ID-list arrays) -- `dev_s1_ids`, `n_s2`, `n_s3` are all cheaply
recoverable from `load_train_data()` directly.

Run: .venv/Scripts/python.exe scripts/fix_b001r_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    candidate_count_per_s1,
    compute_blocking_metrics_scalable,
    link_found_mask,
    route_incremental_value_scalable,
    truth_edges_dataframe,
    union_count_per_s1,
)
from blocking_resource import ResourceMonitor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B001R_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001R"

REFERENCE_FORWARD_K = 50
R_GRID = [1, 2, 3, 5]


def main() -> None:
    print("[fix-B001R] loading train data (for truth_by_s1, dev_s1_ids, n_s2/n_s3)...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    n_s2, n_s3 = len(data.source2), len(data.source3)

    truth_edges = truth_edges_dataframe(truth_by_s1)
    print(f"[fix-B001R] {len(dev_ids):,} dev S1 entities, {len(truth_edges):,} true links total.")

    print(f"[fix-B001R] loading existing forward_word_k{REFERENCE_FORWARD_K}.parquet (no retrieval rerun)...")
    fwd_full = read_candidates(B001_DIR / f"forward_word_k{REFERENCE_FORWARD_K}.parquet")
    fwd_best = fwd_full[["s1_entity_id", "target_entity_id"]]

    print("[fix-B001R] loading existing reverse_word_r5.parquet (no retrieval rerun)...")
    rev_full = read_candidates(B001R_DIR / "reverse_word_r5.parquet")

    print("[fix-B001R] computing forward-reference mask (join, not per-entity sets)...")
    with ResourceMonitor() as mon_fwd_ref:
        forward_mask = link_found_mask(truth_edges, fwd_best)
    print(f"[fix-B001R] forward reference (k={REFERENCE_FORWARD_K}) done in {mon_fwd_ref.report.wall_seconds:.1f}s, "
          f"peak {mon_fwd_ref.report.peak_rss_mb:.0f}MB")

    b001r_results: dict = {"reference_forward_k": REFERENCE_FORWARD_K, "r_grid": {}}

    for r in R_GRID:
        print(f"[fix-B001R] r={r}: computing scalable metrics...")
        with ResourceMonitor() as mon_r:
            sub = rev_full[rev_full["reverse_rank"] <= r][["s1_entity_id", "target_entity_id"]]

            # A. reverse-only metrics
            reverse_mask = link_found_mask(truth_edges, sub)
            reverse_counts = candidate_count_per_s1(sub, dev_ids)
            reverse_only = compute_blocking_metrics_scalable(
                dev_ids, truth_edges, reverse_mask, reverse_counts, n_s2_total=n_s2, n_s3_total=n_s3
            )

            # B. forward(k=50) union reverse(r) metrics -- no union table built
            union_mask = forward_mask | reverse_mask
            union_counts = union_count_per_s1(fwd_best, sub, dev_ids)
            union_m = compute_blocking_metrics_scalable(
                dev_ids, truth_edges, union_mask, union_counts, n_s2_total=n_s2, n_s3_total=n_s3
            )

            # C/D. unique true links rescued by reverse + newly-complete entities
            delta = route_incremental_value_scalable(truth_edges, forward_mask, union_mask)

        b001r_results["r_grid"][str(r)] = {
            "reverse_only": reverse_only.as_dict(),
            "forward_union_reverse": union_m.as_dict(),
            "incremental_over_forward_reference": delta,
            "resource": mon_r.report.as_dict(),
        }
        print(f"[fix-B001R] r={r}: reverse_only_recall={reverse_only.link_recall:.4f} "
              f"union_recall={union_m.link_recall:.4f} "
              f"union_complete_coverage={union_m.complete_true_link_coverage_non_singleton:.4f} "
              f"new_links_rescued={delta['new_true_links_rescued']:,} "
              f"newly_complete={delta['newly_complete_s1_entities']:,} "
              f"peak_rss={mon_r.report.peak_rss_mb:.0f}MB")

    write_manifest(b001r_results, B001R_DIR / "b001r_results.json")
    print(f"[fix-B001R] done. Results written to {B001R_DIR / 'b001r_results.json'}")


if __name__ == "__main__":
    main()
