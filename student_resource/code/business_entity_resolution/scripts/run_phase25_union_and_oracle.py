"""Phase 2.5 Part F -- union B004's routes + full-scale char retrieval into
a NEW candidate table, then measure the resulting oracle F0.5 and standard
blocking metrics for a direct before/after comparison against B004 alone.

Does NOT touch, overwrite, or delete `experiments/blocking/B004/union_candidates.parquet`
-- B004 is preserved as the prior canonical checkpoint. This writes a
separate artifact at `experiments/blocking/PHASE_2_5/union_candidates_with_char.parquet`.

Run: .venv/Scripts/python.exe scripts/run_phase25_union_and_oracle.py
Precondition: run_char_full_scale.py has completed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    candidate_count_per_s1,
    compute_blocking_metrics_scalable,
    link_found_mask,
    route_incremental_value_scalable,
    truth_edges_dataframe,
)
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_union import union_candidates_partitioned  # noqa: E402
from evaluation import evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"
B003_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B003"
PHASE25_DIR = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_2_5"
N_BUCKETS = 160


def oracle_f05(truth_by_s1: dict, truth_edges: pd.DataFrame, found_mask: np.ndarray) -> float:
    te = truth_edges.copy()
    te["_found"] = found_mask
    found_only = te[te["_found"]]
    oracle_pred_by_s1 = found_only.groupby("s1_entity_id")["target_entity_id"].apply(list).to_dict()
    result = evaluate(truth_by_s1, oracle_pred_by_s1, beta=0.5, keep_entity_scores=False)
    return result.macro_f_beta


def main() -> None:
    print("[PHASE2.5-UNION] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    truth_edges = truth_edges_dataframe(truth_by_s1)
    n_s2, n_s3 = len(data.source2), len(data.source3)

    route_paths = {
        "exact": B004_DIR / "route_inputs_tmp" / "exact_filtered.parquet",
        "forward_word": B004_DIR / "route_inputs_tmp" / "forward_filtered.parquet",
        "reverse_word": B004_DIR / "route_inputs_tmp" / "reverse_filtered.parquet",
        "numeric_exact_set": B003_DIR / "numeric_exact_set_candidates.parquet",
        "char_full": PHASE25_DIR / "char_full_candidates.parquet",
    }
    for name, p in route_paths.items():
        assert p.exists(), f"missing route input: {p}"
        print(f"[PHASE2.5-UNION]   {name}: {p}")

    output_path = PHASE25_DIR / "union_candidates_with_char.parquet"
    print(f"[PHASE2.5-UNION] running memory-safe partitioned union (n_buckets={N_BUCKETS})...")
    with ResourceMonitor() as mon_union:
        diag = union_candidates_partitioned(
            list(route_paths.values()), output_path, n_buckets=N_BUCKETS,
            tmp_dir=PHASE25_DIR / "buckets_tmp_with_char",
        )
    print(f"[PHASE2.5-UNION] union done in {mon_union.report.wall_seconds:.1f}s, "
          f"peak {mon_union.report.peak_rss_mb:.0f}MB, input_rows={diag['total_input_rows']:,}, "
          f"output_rows={diag['total_output_rows_before_cross_partition_merge']:,}")

    print("[PHASE2.5-UNION] computing standard blocking metrics on the NEW union...")
    new_union_df = pd.read_parquet(
        output_path, engine="pyarrow", columns=["s1_entity_id", "target_entity_id"]
    )
    with ResourceMonitor() as mon_metrics_new:
        new_mask = link_found_mask(truth_edges, new_union_df)
        new_counts = candidate_count_per_s1(new_union_df, dev_ids)
        m_new = compute_blocking_metrics_scalable(
            dev_ids, truth_edges, new_mask, new_counts, n_s2_total=n_s2, n_s3_total=n_s3
        )
    del new_union_df
    print(f"[PHASE2.5-UNION] NEW union: link_recall={m_new.link_recall:.4f} "
          f"complete_coverage={m_new.complete_true_link_coverage_non_singleton:.4f} "
          f"edges={m_new.total_candidate_edges:,}")

    print("[PHASE2.5-UNION] computing standard blocking metrics on the ORIGINAL B004 union (for comparison)...")
    old_union_df = pd.read_parquet(
        B004_DIR / "union_candidates.parquet", engine="pyarrow", columns=["s1_entity_id", "target_entity_id"]
    )
    with ResourceMonitor() as mon_metrics_old:
        old_mask = link_found_mask(truth_edges, old_union_df)
        old_counts = candidate_count_per_s1(old_union_df, dev_ids)
        m_old = compute_blocking_metrics_scalable(
            dev_ids, truth_edges, old_mask, old_counts, n_s2_total=n_s2, n_s3_total=n_s3
        )
    del old_union_df
    print(f"[PHASE2.5-UNION] OLD B004 union: link_recall={m_old.link_recall:.4f} "
          f"complete_coverage={m_old.complete_true_link_coverage_non_singleton:.4f} "
          f"edges={m_old.total_candidate_edges:,}")

    delta = route_incremental_value_scalable(truth_edges, old_mask, new_mask)
    print(f"[PHASE2.5-UNION] char's incremental value beyond B004: "
          f"new_links_rescued={delta['new_true_links_rescued']:,} "
          f"newly_complete={delta['newly_complete_s1_entities']:,}")

    print("[PHASE2.5-UNION] computing oracle F0.5 for OLD (B004) and NEW (B004+char) unions...")
    with ResourceMonitor() as mon_oracle:
        old_oracle = oracle_f05(truth_by_s1, truth_edges, old_mask)
        new_oracle = oracle_f05(truth_by_s1, truth_edges, new_mask)
    print(f"[PHASE2.5-UNION] ORACLE F0.5: B004={old_oracle:.6f} -> B004+char={new_oracle:.6f} "
          f"(delta={new_oracle - old_oracle:+.6f})")

    results = {
        "routes_included": list(route_paths.keys()),
        "union_diagnostics": diag,
        "union_resource": mon_union.report.as_dict(),
        "new_union_metrics": m_new.as_dict(),
        "new_union_metrics_resource": mon_metrics_new.report.as_dict(),
        "old_b004_metrics": m_old.as_dict(),
        "old_b004_metrics_resource": mon_metrics_old.report.as_dict(),
        "char_incremental_value_beyond_b004": delta,
        "oracle_f05_b004": old_oracle,
        "oracle_f05_b004_plus_char": new_oracle,
        "oracle_f05_delta": new_oracle - old_oracle,
        "oracle_resource": mon_oracle.report.as_dict(),
    }
    write_manifest(results, PHASE25_DIR / "phase25_union_and_oracle_results.json")
    print(f"[PHASE2.5-UNION] done. Results written to "
          f"{PHASE25_DIR / 'phase25_union_and_oracle_results.json'}")


if __name__ == "__main__":
    main()
