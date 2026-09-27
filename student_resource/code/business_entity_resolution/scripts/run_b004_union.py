"""EXP-B004 -- multi-route union benchmark, the Phase-2 headline experiment.

Operating points selected from MEASURED evidence (review checkpoint item 6),
not placeholders:

  - FORWARD_K = 50: the best measured forward operating point (B001's own
    grid top; already fully retrieved, no rerun cost). Per-source retrieval
    was tested (EXP ablation) and, at matched candidate-edge budget (not
    matched nominal k), showed NO material advantage over global top-k --
    global k=10 (17.49M edges) reached recall=0.7774/coverage=0.5609 vs.
    per-source k=5 (also 17.49M edges) at recall=0.7751/coverage=0.5548,
    global very slightly ahead. Per the mission's own decision rule ("if
    per-source produces no material advantage, keep global"), global top-k
    is retained.
  - REVERSE_R = 5: the best measured reverse operating point (B001R's own
    grid top); forward(k=50) union reverse(r=5) reaches recall=0.8775,
    coverage=0.7252, rescuing 188,431 links / 83,490 newly-complete
    entities beyond forward alone.
  - Numeric Route A (exact numeric-set signature) INCLUDED: measured
    +84,519 true links / +35,405 newly-complete entities beyond the word
    union, a real (if modest, ~1.4% of total truth) positive contribution.
  - Numeric Route B (rare individual token) EXCLUDED from this canonical
    union: measured incremental value beyond word+RouteA was negligible
    (order of a few thousand links) for a disproportionate edge cost (up
    to 18-185M rows depending on run-to-run memory pressure) -- recorded as
    DEFERRED evidence, not deleted.
  - Character n-gram (B002) EXCLUDED from this canonical union: the pilot
    showed a genuinely strong 74.4% rescue rate of EXP-B001's own residual
    (202,061 / 271,507 links), but that pilot was deliberately scoped to
    the ~207K-entity residual population, not the full 1.76M-entity
    development set -- a true full-`development`-scale run would need a
    full corpus refit and a full ~10.32M-row target pool, which the
    pilot's own naive linear extrapolation (200.8 minutes) does not
    account for and almost certainly understates. Recommended instead as
    a conditional, OBSERVABLE-trigger-gated EXP-B005 rescue applied only
    to whatever bounded residual population the analysis below identifies
    (never ground-truth-gated, per the mission's own no-oracle-gating
    rule) -- this keeps its cost close to what the pilot itself measured
    (bounded population, not unrestricted scale) rather than the
    unbounded full-development cost.
  - Exact route: only `route_exact_name_address` (50,437 rows) is used.
    The name-only relationship (12,677,375 rows on its own) was already
    established as far too coarse a signal to trust as a production
    candidate source (see EXACT_DIR/exact_results.json).

Memory-safety note: the four included routes total ~254M raw candidate
rows before dedup (exact 50,437 + forward 87,161,478 + reverse 49,433,565
+ numeric-exact-set 117,706,510). This is exactly the scale the review
checkpoint's item 7 anticipated as unsafe for a single in-memory
`pandas.concat` + `groupby`; this script uses
`union_candidates_partitioned` (src/blocking_union.py) instead, which
processes one route's file at a time and never holds the full union in
memory.

Run: .venv/Scripts/python.exe scripts/run_b004_union.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

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
from blocking_union import union_candidates_partitioned  # noqa: E402
from validation_split import match_bucket  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"
EXACT_DIR = PROJECT_ROOT / "experiments" / "blocking" / "EXACT"
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B001R_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001R"
B003_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B003"
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"

FORWARD_K = 50
REVERSE_R = 5
N_BUCKETS = 128


def _filtered_exact_path(tmp_dir: Path) -> Path:
    """Only route_exact_name_address rows -- see module docstring."""
    df = read_candidates(EXACT_DIR / "exact_candidates.parquet")
    df = df[df["route_exact_name_address"]][
        ["s1_entity_id", "target_entity_id", "route_exact_name_address"]
    ].copy()
    path = tmp_dir / "exact_filtered.parquet"
    write_candidates(df, path)
    return path


def _filtered_forward_path(tmp_dir: Path) -> Path:
    df = read_candidates(B001_DIR / f"forward_word_k{FORWARD_K}.parquet")
    path = tmp_dir / "forward_filtered.parquet"
    write_candidates(df, path)
    return path


def _filtered_reverse_path(tmp_dir: Path) -> Path:
    df = read_candidates(B001R_DIR / "reverse_word_r5.parquet")
    df = df[df["reverse_rank"] <= REVERSE_R]
    path = tmp_dir / "reverse_filtered.parquet"
    write_candidates(df, path)
    return path


def slice_metrics(truth_edges, cand_by_s1_counts, mask, n_s2, n_s3, subset_ids, label):
    if not subset_ids:
        return None
    subset_set = set(subset_ids)
    te_mask = truth_edges["s1_entity_id"].isin(subset_set).to_numpy()
    sub_truth_edges = truth_edges[te_mask].reset_index(drop=True)
    sub_found = mask[te_mask]
    sub_counts = cand_by_s1_counts.reindex(subset_ids, fill_value=0)
    m = compute_blocking_metrics_scalable(subset_ids, sub_truth_edges, sub_found, sub_counts, n_s2_total=n_s2, n_s3_total=n_s3)
    return {"slice": label, "n_entities": len(subset_ids), **m.as_dict()}


def main() -> None:
    print("[B004] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    truth_edges = truth_edges_dataframe(truth_by_s1)
    n_s2, n_s3 = len(data.source2), len(data.source3)

    tmp_dir = B004_DIR / "route_inputs_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("[B004] preparing per-route input files (filtered to selected operating points)...")
    route_paths = {
        "exact": _filtered_exact_path(tmp_dir),
        "forward_word": _filtered_forward_path(tmp_dir),
        "reverse_word": _filtered_reverse_path(tmp_dir),
        "numeric_exact_set": B003_DIR / "numeric_exact_set_candidates.parquet",
    }
    for name, p in route_paths.items():
        print(f"[B004]   {name}: {p}")

    output_path = B004_DIR / "union_candidates.parquet"

    print(f"[B004] running memory-safe partitioned union (n_buckets={N_BUCKETS})...")
    with ResourceMonitor() as mon_union:
        diag = union_candidates_partitioned(
            list(route_paths.values()), output_path, n_buckets=N_BUCKETS, tmp_dir=B004_DIR / "buckets_tmp"
        )
    print(f"[B004] union done in {mon_union.report.wall_seconds:.1f}s, peak {mon_union.report.peak_rss_mb:.0f}MB, "
          f"input_rows={diag['total_input_rows']:,}")

    union_df = read_candidates(output_path)
    print(f"[B004] final unioned candidate table: {len(union_df):,} rows")

    print("[B004] computing authoritative metrics (scalable path)...")
    union_mask = link_found_mask(truth_edges, union_df)
    union_counts = candidate_count_per_s1(union_df, dev_ids)
    m_union = compute_blocking_metrics_scalable(
        dev_ids, truth_edges, union_mask, union_counts, n_s2_total=n_s2, n_s3_total=n_s3
    )
    print(f"[B004] UNION: link_recall={m_union.link_recall:.4f} "
          f"complete_coverage={m_union.complete_true_link_coverage_non_singleton:.4f} "
          f"singleton_exposure={m_union.singleton_candidate_exposure_rate:.4f} "
          f"edges={m_union.total_candidate_edges:,} global_RR={m_union.global_reduction_ratio:.6f}")

    # ---- per-route standalone + incremental value (reusing route files) ---
    print("[B004] computing per-route standalone + incremental value...")
    incremental: dict = {}
    running_mask = None
    for name, path in route_paths.items():
        route_df = read_candidates(path)
        solo_mask = link_found_mask(truth_edges, route_df)
        solo_counts = candidate_count_per_s1(route_df, dev_ids)
        m_solo = compute_blocking_metrics_scalable(
            dev_ids, truth_edges, solo_mask, solo_counts, n_s2_total=n_s2, n_s3_total=n_s3
        )

        before_mask = running_mask if running_mask is not None else (solo_mask & False)
        running_mask = before_mask | solo_mask if running_mask is not None else solo_mask
        delta = route_incremental_value_scalable(truth_edges, before_mask, running_mask)

        incremental[name] = {
            "standalone_metrics": m_solo.as_dict(),
            "incremental_new_true_links_rescued": delta["new_true_links_rescued"],
            "incremental_newly_complete_s1_entities": delta["newly_complete_s1_entities"],
            "candidate_edges_in_route": int(len(route_df)),
        }
        print(f"[B004] route={name}: standalone_recall={m_solo.link_recall:.4f} "
              f"incremental_new_links={delta['new_true_links_rescued']:,} "
              f"incremental_newly_complete={delta['newly_complete_s1_entities']:,}")
        del route_df

    # ---- diagnostic slices --------------------------------------------------
    print("[B004] computing diagnostic slices...")
    country_by_s1 = dict(zip(data.source1["entity_id"], data.source1["country"]))
    addr_by_s1 = dict(zip(data.source1["entity_id"], data.source1["business_address"]))

    slices = []
    countries = sorted(set(country_by_s1.get(s1, "") for s1 in dev_ids))
    for country in countries:
        subset = [s1 for s1 in dev_ids if country_by_s1.get(s1) == country]
        r = slice_metrics(truth_edges, union_counts, union_mask, n_s2, n_s3, subset, f"country={country}")
        if r:
            slices.append(r)

    for bucket in ("0", "1", "2+"):
        subset = [s1 for s1 in dev_ids if match_bucket(len(truth_by_s1.get(s1, frozenset()))) == bucket]
        r = slice_metrics(truth_edges, union_counts, union_mask, n_s2, n_s3, subset, f"match_bucket={bucket}")
        if r:
            slices.append(r)

    for has_addr, label in ((True, "address_present"), (False, "address_missing")):
        subset = [s1 for s1 in dev_ids if (bool((addr_by_s1.get(s1, "") or "").strip())) == has_addr]
        r = slice_metrics(truth_edges, union_counts, union_mask, n_s2, n_s3, subset, label)
        if r:
            slices.append(r)

    results = {
        "selected_operating_point": {
            "forward_k": FORWARD_K, "reverse_r": REVERSE_R,
            "routes_included": list(route_paths.keys()),
            "per_source_retrieval": "REJECTED (no material advantage at matched budget)",
            "numeric_rare_token_route": "DEFERRED (negligible incremental value beyond exact-set route)",
            "char_ngram_route": "DEFERRED from full-scale union; recommended as targeted EXP-B005 rescue",
        },
        "union_metrics": m_union.as_dict(),
        "route_incremental_value": incremental,
        "diagnostic_slices": slices,
        "union_resource": mon_union.report.as_dict(),
        "union_diagnostics": diag,
    }
    write_manifest(results, B004_DIR / "b004_results.json")
    print(f"[B004] done. Full results written to {B004_DIR / 'b004_results.json'}")


if __name__ == "__main__":
    main()
