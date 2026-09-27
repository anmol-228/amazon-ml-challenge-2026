"""EXP-B005 (conditional) -- targeted numeric rare-token rescue.

Justified by the residual analysis (experiments/blocking/RESIDUAL/residual_analysis.json):
of the 663,422 true links B004 misses, 368,300 (55.5%) have NONEMPTY numeric-
token overlap between S1 and the true target -- i.e. positive, fully
observable numeric evidence exists, yet no B004 route caught them. The
likely cause: EXP-B003's Route B (rare individual numeric token) was
excluded from B004 (measured negligible standalone value at the time), and
when it did run, a memory-pressure fallback (from running immediately after
Route A's 117.7M-row generation in the same process) forced its
`max_posting_size` down to 200 -- far stricter than the measured token
collision distribution (p99=1824) would require in isolation.

This experiment re-runs Route B ALONE (not immediately after Route A, so it
is not memory-starved by a sibling route still resident) with a properly
measured cutoff, and measures its incremental value specifically over the
CURRENT B004 union (not over the word-only baseline EXP-B003 used) -- the
real question is what it adds on top of everything already included, not
what it adds on top of a weaker baseline.

No ground truth is used to decide which candidates to generate or which S1
entities receive the route -- it is applied uniformly, exactly like every
other route, per the mission's no-oracle-gating rule. Ground truth is used
only afterward, to measure the resulting incremental value.

Run: .venv/Scripts/python.exe scripts/run_b005_numeric_rescue.py
Precondition: run_b004_union.py has completed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, write_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    candidate_count_per_s1,
    compute_blocking_metrics_scalable,
    link_found_mask,
    route_incremental_value_scalable,
    truth_edges_dataframe,
    union_count_per_s1,
)
from blocking_numeric import numeric_candidates  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"
B005_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B005"

# Measured (EXP-B003/B000): token collision p50=4, p99=1824, max=763372.
# 2000 sits just above the measured p99 -- generous enough to keep almost
# every legitimately rare token, while still excluding the catastrophic
# outliers (hundreds of thousands of postings) that caused the original
# MemoryError this project hit at full-scale.
MAX_POSTING_SIZE = 2000
MIN_TOKEN_LENGTH = 2


def main() -> None:
    print("[B005] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    truth_edges = truth_edges_dataframe(truth_by_s1)
    n_s2, n_s3 = len(data.source2), len(data.source3)

    dev_s1_df = data.source1[data.source1["entity_id"].isin(dev_set)]
    s1_ids = dev_s1_df["entity_id"].tolist()
    s1_addr = dev_s1_df["business_address"].tolist()
    target_ids = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
    target_addr = list(data.source2["business_address"]) + list(data.source3["business_address"])

    print(f"[B005] generating rare-token candidates in isolation, max_posting_size={MAX_POSTING_SIZE}...")
    fallback_cutoff = MAX_POSTING_SIZE
    try:
        with ResourceMonitor() as mon_gen:
            cand_b005 = numeric_candidates(
                s1_ids, s1_addr, target_ids, target_addr,
                max_posting_size=MAX_POSTING_SIZE, min_token_length=MIN_TOKEN_LENGTH,
            )
    except MemoryError:
        fallback_cutoff = 500
        print(f"[B005] MemoryError at cutoff={MAX_POSTING_SIZE}; retrying at {fallback_cutoff}...")
        with ResourceMonitor() as mon_gen:
            cand_b005 = numeric_candidates(
                s1_ids, s1_addr, target_ids, target_addr,
                max_posting_size=fallback_cutoff, min_token_length=MIN_TOKEN_LENGTH,
            )
    cand_b005["route_numeric_rare_token_v2"] = True
    print(f"[B005] generated {len(cand_b005):,} candidate rows in {mon_gen.report.wall_seconds:.1f}s, "
          f"peak {mon_gen.report.peak_rss_mb:.0f}MB, max_posting_size_used={fallback_cutoff}")
    B005_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(cand_b005, B005_DIR / "numeric_rare_token_v2_candidates.parquet")

    print("[B005] measuring standalone metrics...")
    solo_mask = link_found_mask(truth_edges, cand_b005)
    solo_counts = candidate_count_per_s1(cand_b005, dev_ids)
    m_solo = compute_blocking_metrics_scalable(
        dev_ids, truth_edges, solo_mask, solo_counts, n_s2_total=n_s2, n_s3_total=n_s3
    )
    print(f"[B005] standalone: link_recall={m_solo.link_recall:.4f} edges={m_solo.total_candidate_edges:,}")

    print("[B005] measuring incremental value over the CURRENT B004 union...")
    b004_df = pd.read_parquet(
        B004_DIR / "union_candidates.parquet", engine="pyarrow", columns=["s1_entity_id", "target_entity_id"]
    )
    with ResourceMonitor() as mon_incr:
        b004_mask = link_found_mask(truth_edges, b004_df)
        union_mask = b004_mask | solo_mask
        union_counts = union_count_per_s1(b004_df, cand_b005, dev_ids)
        m_union = compute_blocking_metrics_scalable(
            dev_ids, truth_edges, union_mask, union_counts, n_s2_total=n_s2, n_s3_total=n_s3
        )
        delta = route_incremental_value_scalable(truth_edges, b004_mask, union_mask)
    print(f"[B005] B004 union B005: link_recall={m_union.link_recall:.4f} "
          f"complete_coverage={m_union.complete_true_link_coverage_non_singleton:.4f} "
          f"new_links_rescued={delta['new_true_links_rescued']:,} "
          f"newly_complete={delta['newly_complete_s1_entities']:,} "
          f"peak_rss={mon_incr.report.peak_rss_mb:.0f}MB")

    # ---- how much of the specific "numeric agreement but missed" residual does this close? ---
    residual_path = PROJECT_ROOT / "experiments" / "blocking" / "RESIDUAL" / "residual_missed_links.parquet"
    coverage_of_target_category = None
    if residual_path.exists():
        residual_df = read_candidates(residual_path)
        numeric_agree_residual = residual_df[residual_df["numeric_agreement"]]
        na_truth_edges = numeric_agree_residual[["s1_entity_id", "target_entity_id"]].reset_index(drop=True)
        na_found_by_b005 = link_found_mask(na_truth_edges, cand_b005)
        coverage_of_target_category = {
            "n_numeric_agreement_missed_links": len(numeric_agree_residual),
            "n_rescued_by_b005": int(na_found_by_b005.sum()),
            "rescue_rate": float(na_found_by_b005.mean()) if len(na_found_by_b005) else 0.0,
        }
        print(f"[B005] of the {coverage_of_target_category['n_numeric_agreement_missed_links']:,} "
              f"'numeric agreement but missed' residual links, B005 rescues "
              f"{coverage_of_target_category['n_rescued_by_b005']:,} "
              f"({coverage_of_target_category['rescue_rate']:.4%})")

    ACCEPT_THRESHOLD_NEW_LINKS = 20000  # a fraction of a percent of total truth is not worth the added complexity
    decision = "ACCEPTED_INTO_FINAL_ARCHITECTURE" if delta["new_true_links_rescued"] >= ACCEPT_THRESHOLD_NEW_LINKS else "REJECTED_INSUFFICIENT_VALUE"

    results = {
        "max_posting_size_used": fallback_cutoff,
        "generation_resource": mon_gen.report.as_dict(),
        "standalone_metrics": m_solo.as_dict(),
        "b004_union_b005_metrics": m_union.as_dict(),
        "incremental_over_b004": delta,
        "incremental_resource": mon_incr.report.as_dict(),
        "coverage_of_numeric_agreement_residual_category": coverage_of_target_category,
        "decision": decision,
    }
    write_manifest(results, B005_DIR / "b005_results.json")
    print(f"[B005] decision={decision}")


if __name__ == "__main__":
    main()
