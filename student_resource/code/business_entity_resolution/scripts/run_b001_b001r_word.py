"""EXP-B001 (forward S1->target) and EXP-B001R (reverse target->S1) word-TF-IDF
blocking, sharing the single vectorizer/matrices EXP-B000 fit and cached.

Both directions use the exact same fitted representation (DEC-020 / mission
spec section 17.1), so this script loads the B000 cache once and runs both
directions from it -- one retrieval pass at the grid's max k/r, with smaller
grid points derived by rank-filtering the same pass (no redundant re-runs,
per the mission spec's time-discipline requirement, section 28).

Ordering note (load-bearing, found by direct diagnosis after two real
crashes): retrieval runs FIRST, immediately after loading the small cached
matrices, and ONLY AFTER both retrieval passes are done and written to disk
does this script load the full train data and build the (multi-GB) name-
token dictionaries used for the diagnostic (B)/(C)/(D) breakdowns. An
isolated test proved 200 real retrieval batches succeed cleanly with flat
~700MB RSS in a fresh process -- the earlier crashes (repeated
`std::bad_alloc` from scipy's sparse matmul, persisting even at an 8-row
sub-batch) only occurred when retrieval ran *after* the full train data and
name-token dictionaries were already resident, i.e. process-level heap
fragmentation from that unrelated, much larger allocation activity, not a
true memory shortage or a batch-size problem. Retrieval must therefore run
while the process heap is still clean.

Run: .venv/Scripts/python.exe scripts/run_b001_b001r_word.py
Precondition: scripts/run_b000_preflight.py has been run (populates
experiments/blocking/cache/).
"""

from __future__ import annotations

import gc
import pickle
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scipy import sparse  # noqa: E402

from blocking_data import load_train_data  # noqa: E402
from blocking_io import read_manifest, write_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    build_id_sets_by_s1,
    classify_true_links_by_name_overlap,
    compute_blocking_metrics,
)
from blocking_normalization import name_token_set  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_routes import forward_route, reverse_route  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B001R_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001R"

K_GRID = [5, 10, 20, 50]
R_GRID = [1, 2, 3, 5]
FORWARD_BATCH_SIZE = 1500
REVERSE_BATCH_SIZE = 1500
MIN_SCORE = 0.0


def load_cache():
    # Pickle load is safe here: word_vectorizer.pkl is produced by
    # run_b000_preflight.py in this same local pipeline (experiments/blocking/cache/),
    # never an externally-supplied or downloaded file.
    with open(CACHE_DIR / "word_vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)
    s1_dev_matrix = sparse.load_npz(CACHE_DIR / "s1_dev_word_matrix.npz").tocsr()
    target_matrix = sparse.load_npz(CACHE_DIR / "target_word_matrix.npz").tocsr()
    manifest = read_manifest(CACHE_DIR / "word_cache_manifest.json")
    return vectorizer, s1_dev_matrix, target_matrix, manifest


def _fast_union(df_a, df_b):
    import pandas as pd

    cols = ["s1_entity_id", "target_entity_id"]
    return pd.concat([df_a[cols], df_b[cols]], axis=0, ignore_index=True).drop_duplicates()


def main() -> None:
    print("[B001/B001R] loading cache (small: vectorizer + two matrices only)...")
    vectorizer, s1_dev_matrix, target_matrix, manifest = load_cache()
    dev_s1_ids: list[str] = manifest["dev_s1_ids"]
    target_ids: list[str] = manifest["target_ids"]
    n_s2 = manifest["n_s2"]
    n_s3 = manifest["n_s3"]

    max_k = max(K_GRID)
    max_r = max(R_GRID)

    # ---- EXP-B001: forward, one pass at max_k (clean-heap phase) ----------
    print(f"[B001] forward retrieval at k={max_k} over {len(dev_s1_ids):,} dev S1 queries...")
    with ResourceMonitor() as mon_fwd:
        fwd_df = forward_route(dev_s1_ids, s1_dev_matrix, target_ids, target_matrix, k=max_k, batch_size=FORWARD_BATCH_SIZE, min_score=MIN_SCORE)
    print(f"[B001] forward retrieval done in {mon_fwd.report.wall_seconds:.1f}s, peak {mon_fwd.report.peak_rss_mb:.0f}MB, rows={len(fwd_df):,}")

    B001_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(fwd_df, B001_DIR / f"forward_word_k{max_k}.parquet")

    # ---- EXP-B001R: reverse, one pass at max_r (still clean-heap phase) ---
    print(f"[B001R] reverse retrieval at r={max_r} over {len(target_ids):,} target queries...")
    with ResourceMonitor() as mon_rev:
        rev_df = reverse_route(dev_s1_ids, s1_dev_matrix, target_ids, target_matrix, r=max_r, batch_size=REVERSE_BATCH_SIZE, min_score=MIN_SCORE)
    print(f"[B001R] reverse retrieval done in {mon_rev.report.wall_seconds:.1f}s, peak {mon_rev.report.peak_rss_mb:.0f}MB, rows={len(rev_df):,}")

    B001R_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(rev_df, B001R_DIR / f"reverse_word_r{max_r}.parquet")

    # Both retrieval passes are safely on disk now. Free everything
    # retrieval-specific before the much heavier train-data/diagnostic-dict
    # phase begins, so that phase starts from a clean, low-fragmentation
    # heap too (defensive, not just cosmetic -- see module docstring).
    del vectorizer, s1_dev_matrix, target_matrix
    gc.collect()

    # ---- Diagnostics phase: full train data + name-token maps -------------
    print("[B001/B001R] loading full train data for grid metrics + diagnostics...")
    data = load_train_data()
    dev_id_set = set(dev_s1_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_id_set}

    print("[B001/B001R] building name-token maps for overlap diagnostics...")
    s1_name_by_id = dict(zip(data.source1["entity_id"], data.source1["business_name"]))
    target_name_by_id = dict(
        zip(list(data.source2["entity_id"]) + list(data.source3["entity_id"]),
            list(data.source2["business_name"]) + list(data.source3["business_name"]))
    )
    s1_name_tokens = {eid: name_token_set(name) for eid, name in s1_name_by_id.items() if eid in dev_id_set}
    target_name_tokens = {eid: name_token_set(name) for eid, name in target_name_by_id.items()}
    nonzero_truth, zero_truth = classify_true_links_by_name_overlap(truth_by_s1, s1_name_tokens, target_name_tokens)

    # ---- EXP-B001 k-grid metrics --------------------------------------------
    b001_results = {"resource": mon_fwd.report.as_dict(), "k_grid": {}}
    for k in K_GRID:
        sub = fwd_df[fwd_df["forward_rank"] <= k]
        cand_by_s1 = build_id_sets_by_s1(sub)
        overall = compute_blocking_metrics(dev_s1_ids, truth_by_s1, cand_by_s1, n_s2_total=n_s2, n_s3_total=n_s3)
        nonzero_m = compute_blocking_metrics(dev_s1_ids, nonzero_truth, cand_by_s1)
        zero_m = compute_blocking_metrics(dev_s1_ids, zero_truth, cand_by_s1)
        n_zero_links = zero_m.n_true_links
        n_zero_rescued = int(round(zero_m.link_recall * n_zero_links)) if n_zero_links else 0
        b001_results["k_grid"][str(k)] = {
            "overall": overall.as_dict(),
            "diagnostic_B_nonzero_name_overlap_recall": nonzero_m.link_recall,
            "diagnostic_C_zero_name_overlap_recall": zero_m.link_recall,
            "diagnostic_D_address_rescued_count": n_zero_rescued,
            "diagnostic_D_address_rescued_pct_of_zero_overlap": zero_m.link_recall,
            "n_true_links_nonzero_overlap": nonzero_m.n_true_links,
            "n_true_links_zero_overlap": n_zero_links,
        }
        print(f"[B001] k={k}: link_recall={overall.link_recall:.4f} complete_coverage={overall.complete_true_link_coverage_non_singleton:.4f} singleton_exposure={overall.singleton_candidate_exposure_rate:.4f}")

    write_manifest(b001_results, B001_DIR / "b001_results.json")

    # ---- EXP-B001R r-grid metrics (reverse alone + forward-union-reverse) --
    fwd_best = fwd_df[fwd_df["forward_rank"] <= max_k]
    b001r_results = {"resource": mon_rev.report.as_dict(), "r_grid": {}}
    for r in R_GRID:
        sub = rev_df[rev_df["reverse_rank"] <= r]
        rev_cand_by_s1 = build_id_sets_by_s1(sub)
        reverse_only = compute_blocking_metrics(dev_s1_ids, truth_by_s1, rev_cand_by_s1, n_s2_total=n_s2, n_s3_total=n_s3)

        union_df = _fast_union(fwd_best, sub)
        union_cand_by_s1 = build_id_sets_by_s1(union_df)
        union_m = compute_blocking_metrics(dev_s1_ids, truth_by_s1, union_cand_by_s1, n_s2_total=n_s2, n_s3_total=n_s3)

        b001r_results["r_grid"][str(r)] = {
            "reverse_only": reverse_only.as_dict(),
            "forward_union_reverse": union_m.as_dict(),
        }
        print(f"[B001R] r={r}: reverse_only_recall={reverse_only.link_recall:.4f} union_recall={union_m.link_recall:.4f} union_complete_coverage={union_m.complete_true_link_coverage_non_singleton:.4f}")

    write_manifest(b001r_results, B001R_DIR / "b001r_results.json")
    print("[B001/B001R] done.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"[B001/B001R] total wall time: {time.time() - start:.1f}s")
