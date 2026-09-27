"""EXP-B003 -- numeric-address-token blocking recall.

Two complementary routes, each with its own provenance flag, per the
Phase-2 mission spec section 19 and the review checkpoint that added
Route A:

  - `route_numeric_exact_set`: EXACT non-empty numeric-token-SET-signature
    equality (Route A). The strongest canonical numeric finding from
    Phase 1 is that 63.49% of true links agree exactly on their numeric-
    token SET, not just on any one shared token -- this route targets
    that finding directly and is far more specific (smaller collision
    groups) than single-token overlap.
  - `route_numeric_rare_token`: any single shared numeric token, with
    measured posting-size pruning (Route B, the original EXP-B003 route,
    kept as a complementary rare-token rescue mechanism for pairs that
    share a rare individual number without matching on the full set,
    e.g. a typo'd or partially-abbreviated address on one side).

Numeric disagreement is never treated as a rejection signal at the
blocking stage (mission spec section 19.1) -- both routes are purely
positive-evidence unions; conflicting numeric evidence remains a
downstream matcher feature.

Safety note (unchanged from the prior version, still load-bearing): an
unpruned candidate-GENERATION pass previously hit a real MemoryError from
a token with a 763,372-row posting list. Both routes here measure their
own collision-group-size distribution BEFORE generating candidates and
pick a conservative cutoff from that measurement, never a guess.

Metrics use the scalable join/groupby path (src/blocking_metrics.py),
verified equal to the simple per-entity-set path on synthetic data
(tests/test_blocking_infrastructure.py), since candidate tables at this
scale are too large to safely build Python-level per-entity sets from
directly (this is exactly what crashed EXP-B001R's r-grid metrics and an
earlier version of this script's own metrics step).

Run: .venv/Scripts/python.exe scripts/run_b003_numeric.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

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
from blocking_numeric import (  # noqa: E402
    build_numeric_postings,
    build_signature_postings,
    exact_numeric_signature_candidates,
    numeric_candidates,
    numeric_collision_group_sizes,
)
from blocking_resource import ResourceMonitor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B001R_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001R"
B003_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B003"

REFERENCE_FORWARD_K = 50
REFERENCE_REVERSE_R = 5

# Candidate-load acceptance bound: if p99 overall candidate load exceeds
# this, a GBDT-scale matcher would face an unreasonable per-entity
# comparison burden. Used only to decide whether to report a widened pass;
# never silently applied without measurement.
P99_LOAD_ACCEPTANCE_BOUND = 2000


def _measure_collision(postings: dict) -> dict:
    sizes = numeric_collision_group_sizes(postings)
    return {
        "vocab_size": len(postings),
        "p50": float(np.percentile(sizes.values, 50)),
        "p95": float(np.percentile(sizes.values, 95)),
        "p99": float(np.percentile(sizes.values, 99)),
        "max": float(sizes.values.max()),
        "top_20_groups": {str(k): int(v) for k, v in sizes.head(20).to_dict().items()},
    }


def main() -> None:
    print("[B003] loading train data...")
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

    results: dict = {}

    # ---- Route A: exact numeric-set signature ------------------------------
    print("[B003] Route A: measuring signature collision-group sizes...")
    sig_postings = build_signature_postings(target_ids, target_addr)
    sig_collision = _measure_collision(sig_postings)
    print(f"[B003] Route A signature collisions: p50={sig_collision['p50']:.0f} "
          f"p99={sig_collision['p99']:.0f} max={sig_collision['max']:.0f} "
          f"vocab={sig_collision['vocab_size']:,}")
    results["route_a_signature_collision"] = sig_collision

    # An unconditional, measurement-informed safety ceiling, not a
    # conditional one keyed off p99: p99 can look perfectly tame (65 here)
    # while a single pathological signature's group (max=59,180, observed
    # directly on this data) is large enough to exhaust memory during
    # candidate GENERATION itself once multiplied across every S1 sharing
    # it -- the same failure mode Route B's generation already guards
    # against unconditionally. The tail, not the typical case, is the risk.
    sig_cutoff = min(P99_LOAD_ACCEPTANCE_BOUND, max(50, int(sig_collision["p99"] * 10)))
    print(f"[B003] Route A: pruning signatures with a group size above {sig_cutoff} "
          f"(measured p99={sig_collision['p99']:.0f}, max={sig_collision['max']:.0f})")

    print("[B003] Route A: generating exact-signature candidates...")
    fallback_sig_cutoff = sig_cutoff
    try:
        with ResourceMonitor() as mon_a:
            cand_a = exact_numeric_signature_candidates(
                s1_ids, s1_addr, target_ids, target_addr, max_group_size=fallback_sig_cutoff
            )
    except MemoryError:
        fallback_sig_cutoff = 50
        print(f"[B003] Route A: MemoryError at cutoff={sig_cutoff}; retrying at a much "
              f"stricter cutoff={fallback_sig_cutoff}...")
        with ResourceMonitor() as mon_a:
            cand_a = exact_numeric_signature_candidates(
                s1_ids, s1_addr, target_ids, target_addr, max_group_size=fallback_sig_cutoff
            )
    cand_a["route_numeric_exact_set"] = True
    results["route_a_max_group_size_used"] = fallback_sig_cutoff
    print(f"[B003] Route A: {len(cand_a):,} candidate rows, {mon_a.report.wall_seconds:.1f}s, "
          f"peak {mon_a.report.peak_rss_mb:.0f}MB")
    write_candidates(cand_a[["s1_entity_id", "target_entity_id", "route_numeric_exact_set"]], B003_DIR / "numeric_exact_set_candidates.parquet")

    mask_a = link_found_mask(truth_edges, cand_a)
    counts_a = candidate_count_per_s1(cand_a, dev_ids)
    metrics_a = compute_blocking_metrics_scalable(dev_ids, truth_edges, mask_a, counts_a, n_s2_total=n_s2, n_s3_total=n_s3)
    results["route_a_standalone"] = metrics_a.as_dict()
    print(f"[B003] Route A standalone: link_recall={metrics_a.link_recall:.4f} "
          f"edges={metrics_a.total_candidate_edges:,} p99_load={metrics_a.overall_candidate_load['p99']:.0f}")

    # ---- Route B: rare individual numeric token -----------------------------
    print("[B003] Route B: measuring individual-token collision-group sizes...")
    tok_postings = build_numeric_postings(target_ids, target_addr)
    tok_collision = _measure_collision(tok_postings)
    print(f"[B003] Route B token collisions: p50={tok_collision['p50']:.0f} "
          f"p99={tok_collision['p99']:.0f} max={tok_collision['max']:.0f} "
          f"vocab={tok_collision['vocab_size']:,}")
    results["route_b_token_collision"] = tok_collision

    initial_cutoff = 1000
    print(f"[B003] Route B: generating rare-token candidates, pre-pruned at max_posting_size={initial_cutoff}...")
    fallback_cutoff = initial_cutoff
    try:
        with ResourceMonitor() as mon_b:
            cand_b = numeric_candidates(
                s1_ids, s1_addr, target_ids, target_addr,
                max_posting_size=initial_cutoff, min_token_length=2,
            )
    except MemoryError:
        fallback_cutoff = 200
        print(f"[B003] Route B: MemoryError at cutoff={initial_cutoff}; retrying at {fallback_cutoff}...")
        with ResourceMonitor() as mon_b:
            cand_b = numeric_candidates(
                s1_ids, s1_addr, target_ids, target_addr,
                max_posting_size=fallback_cutoff, min_token_length=2,
            )
    cand_b["route_numeric_rare_token"] = True
    print(f"[B003] Route B: {len(cand_b):,} candidate rows, {mon_b.report.wall_seconds:.1f}s, "
          f"peak {mon_b.report.peak_rss_mb:.0f}MB, max_posting_size={fallback_cutoff}")
    write_candidates(cand_b[["s1_entity_id", "target_entity_id", "route_numeric_rare_token"]], B003_DIR / "numeric_rare_token_candidates.parquet")

    mask_b = link_found_mask(truth_edges, cand_b)
    counts_b = candidate_count_per_s1(cand_b, dev_ids)
    metrics_b = compute_blocking_metrics_scalable(dev_ids, truth_edges, mask_b, counts_b, n_s2_total=n_s2, n_s3_total=n_s3)
    results["route_b_standalone"] = metrics_b.as_dict()
    results["route_b_max_posting_size_used"] = fallback_cutoff
    print(f"[B003] Route B standalone: link_recall={metrics_b.link_recall:.4f} "
          f"edges={metrics_b.total_candidate_edges:,} p99_load={metrics_b.overall_candidate_load['p99']:.0f}")

    # ---- Union of A and B ----------------------------------------------------
    print("[B003] computing A union B metrics (scalable, no table concat)...")
    union_ab_mask = mask_a | mask_b
    union_ab_counts = union_count_per_s1(cand_a, cand_b, dev_ids)
    metrics_union_ab = compute_blocking_metrics_scalable(
        dev_ids, truth_edges, union_ab_mask, union_ab_counts, n_s2_total=n_s2, n_s3_total=n_s3
    )
    results["route_a_union_b"] = metrics_union_ab.as_dict()

    delta_a_over_b = route_incremental_value_scalable(truth_edges, mask_b, union_ab_mask)
    delta_b_over_a = route_incremental_value_scalable(truth_edges, mask_a, union_ab_mask)
    results["unique_value_of_a_over_b"] = delta_a_over_b
    results["unique_value_of_b_over_a"] = delta_b_over_a
    print(f"[B003] A union B: link_recall={metrics_union_ab.link_recall:.4f} "
          f"A-unique-links={delta_a_over_b['new_true_links_rescued']:,} "
          f"B-unique-links={delta_b_over_a['new_true_links_rescued']:,}")

    # ---- Incremental value beyond word forward(k=50) union reverse(r=5) ----
    print("[B003] computing incremental value beyond existing word forward+reverse union...")
    fwd_best = read_candidates(B001_DIR / f"forward_word_k{REFERENCE_FORWARD_K}.parquet")[
        ["s1_entity_id", "target_entity_id"]
    ]
    rev_full = read_candidates(B001R_DIR / f"reverse_word_r{REFERENCE_REVERSE_R}.parquet")
    rev_best = rev_full[["s1_entity_id", "target_entity_id"]]

    word_mask = link_found_mask(truth_edges, fwd_best) | link_found_mask(truth_edges, rev_best)
    delta_a_beyond_word = route_incremental_value_scalable(truth_edges, word_mask, word_mask | mask_a)
    delta_b_beyond_word = route_incremental_value_scalable(truth_edges, word_mask, word_mask | mask_b)
    delta_ab_beyond_word = route_incremental_value_scalable(truth_edges, word_mask, word_mask | union_ab_mask)
    results["route_a_incremental_beyond_word_forward_reverse"] = delta_a_beyond_word
    results["route_b_incremental_beyond_word_forward_reverse"] = delta_b_beyond_word
    results["route_a_union_b_incremental_beyond_word_forward_reverse"] = delta_ab_beyond_word
    print(f"[B003] beyond word forward+reverse: A adds {delta_a_beyond_word['new_true_links_rescued']:,} links, "
          f"B adds {delta_b_beyond_word['new_true_links_rescued']:,} links, "
          f"A union B adds {delta_ab_beyond_word['new_true_links_rescued']:,} links "
          f"({delta_ab_beyond_word['newly_complete_s1_entities']:,} newly-complete entities)")

    results["status"] = "COMPLETE"
    write_manifest(results, B003_DIR / "b003_results.json")
    print(f"[B003] done. Results written to {B003_DIR / 'b003_results.json'}")


if __name__ == "__main__":
    main()
