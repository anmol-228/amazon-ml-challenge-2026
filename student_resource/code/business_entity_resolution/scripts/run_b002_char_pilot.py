"""EXP-B002 -- character n-gram / TF-IDF blocking, pilot-first (mission spec
section 18).

Question actually asked (round-3 corrected): does character-trigram TF-IDF
retrieval recover EXP-B001's own residual -- true links with zero business-
name token overlap that EXP-B001's joint (name+address) vector *also* failed
to retrieve -- not the raw ~10% zero-overlap population, most of which
EXP-B001 already rescues via address content?

Pilot design: the query set is exactly the dev-S1 entities that have >=1
such residual true link (an evidence-driven pilot, not an arbitrary sample).
The target pool is every one of those residual true targets (guaranteed
positives) plus a large random background sample of other targets, so
recall is measured honestly (true targets really are there) while keeping
the vectorizer/retrieval cost pilot-scale rather than full 10.3M-row scale.

Precondition: run_b000_preflight.py and run_b001_b001r_word.py have run.
Run: .venv/Scripts/python.exe scripts/run_b002_char_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, write_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    build_id_sets_by_s1,
    classify_true_links_by_name_overlap,
    compute_blocking_metrics,
    zero_overlap_residual_links,
)
from blocking_normalization import joint_text, name_token_set  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_routes import forward_route  # noqa: E402
from blocking_vectorization import VectorizerConfig, fit_vectorizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"
B002_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B002"

BEST_FORWARD_K = 50  # the largest grid point -- the most generous forward baseline
CHAR_K = 50
BACKGROUND_TARGET_N = 400000
SEED = 42


def main() -> None:
    print("[B002] loading train data + B001 forward candidates...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}

    fwd_df = read_candidates(B001_DIR / f"forward_word_k{BEST_FORWARD_K}.parquet")
    b001_cand_by_s1 = build_id_sets_by_s1(fwd_df)
    del fwd_df  # the 87M-row table is no longer needed once its per-entity summary exists
    gc.collect()

    # Only target ids that are an actual true-link target for some dev S1
    # entity are ever consulted by classify_true_links_by_name_overlap --
    # restricting to that set (a few million) instead of building name-
    # token sets for the full ~10.32M target universe was the dominant
    # memory cost the first time this script ran (peaked at ~25.9GB,
    # uncomfortably close to this machine's ~31.6GB ceiling).
    relevant_target_ids = {t for targets in truth_by_s1.values() for t in targets}

    s1_name_by_id = dict(zip(data.source1["entity_id"], data.source1["business_name"]))
    s1_name_tokens = {eid: name_token_set(name) for eid, name in s1_name_by_id.items() if eid in dev_set}
    del s1_name_by_id

    target_id_arr = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
    target_name_arr = list(data.source2["business_name"]) + list(data.source3["business_name"])
    target_name_tokens = {
        eid: name_token_set(name)
        for eid, name in zip(target_id_arr, target_name_arr)
        if eid in relevant_target_ids
    }
    _, zero_truth = classify_true_links_by_name_overlap(truth_by_s1, s1_name_tokens, target_name_tokens)
    del s1_name_tokens, target_name_tokens, relevant_target_ids
    gc.collect()

    residual = zero_overlap_residual_links(zero_truth, b001_cand_by_s1)
    n_residual_links = sum(len(v) for v in residual.values())
    print(f"[B002] EXP-B001 residual (zero name-overlap, not address-rescued): "
          f"{len(residual):,} S1 entities, {n_residual_links:,} true links")

    if n_residual_links == 0:
        write_manifest(
            {"status": "SKIPPED", "reason": "no residual true links after EXP-B001", "n_residual_links": 0},
            B002_DIR / "b002_results.json",
        )
        print("[B002] no residual -- nothing for character n-grams to rescue. SKIPPED.")
        return

    pilot_s1_ids = list(residual.keys())
    residual_target_ids = sorted({t for targets in residual.values() for t in targets})

    # `all_target_ids` (the full ~10.32M-id population) is needed only to
    # draw the background sample from -- it is dropped immediately after,
    # never used to build a full-scale name/address lookup dict. Those
    # lookups are built below restricted to `target_pool_ids` (bounded to
    # ~BACKGROUND_TARGET_N + the guaranteed positives), not all 10.32M
    # targets, for the same memory reason as the token-map restriction above.
    all_target_ids = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
    rng = np.random.RandomState(SEED)
    background_ids = rng.choice(all_target_ids, size=min(BACKGROUND_TARGET_N, len(all_target_ids)), replace=False)
    target_pool_ids = sorted(set(residual_target_ids) | set(background_ids.tolist()))
    target_pool_set = set(target_pool_ids)
    del all_target_ids, background_ids
    print(f"[B002] pilot target pool: {len(target_pool_ids):,} ids "
          f"({len(residual_target_ids):,} guaranteed positives + background)")

    target_universe = pd.concat(
        [
            data.source2[["entity_id", "business_name", "business_address"]],
            data.source3[["entity_id", "business_name", "business_address"]],
        ],
        ignore_index=True,
    )
    target_universe = target_universe[target_universe["entity_id"].isin(target_pool_set)]
    name_by_target = dict(zip(target_universe["entity_id"], target_universe["business_name"]))
    addr_by_target = dict(zip(target_universe["entity_id"], target_universe["business_address"]))
    del target_universe

    pilot_s1_df = data.source1[data.source1["entity_id"].isin(set(pilot_s1_ids))]
    pilot_s1_texts = [
        joint_text(n, a) for n, a in zip(pilot_s1_df["business_name"], pilot_s1_df["business_address"])
    ]
    pilot_s1_ids_ordered = pilot_s1_df["entity_id"].tolist()

    target_texts = [joint_text(name_by_target[t], addr_by_target[t]) for t in target_pool_ids]

    char_config = VectorizerConfig(
        representation="char_ngram", analyzer="char_wb", ngram_range=(3, 3),
        min_df=2, max_df=0.05, fit_id="b002_pilot_char_v1",
        corpus_scope="b002_residual_pilot",
    )
    corpus = pilot_s1_texts + target_texts
    print(f"[B002] fitting char-3gram vectorizer on pilot corpus ({len(corpus):,} docs)...")
    with ResourceMonitor() as mon_fit:
        fitted_char = fit_vectorizer(corpus, char_config)
        s1_matrix = fitted_char.vectorizer.transform(pilot_s1_texts).tocsr()
        target_matrix = fitted_char.vectorizer.transform(target_texts).tocsr()

    print(f"[B002] char retrieval: k={CHAR_K} over {len(pilot_s1_ids_ordered):,} pilot queries "
          f"x {len(target_pool_ids):,} targets...")
    with ResourceMonitor() as mon_retrieve:
        char_df = forward_route(
            pilot_s1_ids_ordered, s1_matrix, target_pool_ids, target_matrix, k=CHAR_K,
            batch_size=500, route_col="route_char", rank_col="char_rank", score_col="char_score",
        )

    char_cand_by_s1 = build_id_sets_by_s1(char_df)

    residual_frozensets = {s1: v for s1, v in residual.items()}
    n_rescued = 0
    for s1, missing in residual_frozensets.items():
        n_rescued += len(missing & char_cand_by_s1.get(s1, frozenset()))
    residual_rescue_rate = n_rescued / n_residual_links

    union_after_char = {}
    for s1 in pilot_s1_ids_ordered:
        union_after_char[s1] = b001_cand_by_s1.get(s1, frozenset()) | char_cand_by_s1.get(s1, frozenset())
    m_with_char = compute_blocking_metrics(pilot_s1_ids_ordered, {s1: truth_by_s1.get(s1, frozenset()) for s1 in pilot_s1_ids_ordered}, union_after_char)
    m_without_char = compute_blocking_metrics(pilot_s1_ids_ordered, {s1: truth_by_s1.get(s1, frozenset()) for s1 in pilot_s1_ids_ordered}, {s1: b001_cand_by_s1.get(s1, frozenset()) for s1 in pilot_s1_ids_ordered})

    B002_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(char_df, B002_DIR / "char_pilot_candidates.parquet")

    results = {
        "n_residual_s1_entities": len(residual),
        "n_residual_true_links": n_residual_links,
        "n_residual_links_rescued_by_char": n_rescued,
        "residual_rescue_rate": residual_rescue_rate,
        "target_pool_size": len(target_pool_ids),
        "fit_resource": mon_fit.report.as_dict(),
        "retrieval_resource": mon_retrieve.report.as_dict(),
        "char_provenance": fitted_char.provenance(),
        "pilot_link_recall_with_char": m_with_char.link_recall,
        "pilot_link_recall_without_char": m_without_char.link_recall,
        "pilot_candidate_load_with_char": m_with_char.overall_candidate_load,
    }

    seconds_per_query = mon_retrieve.report.wall_seconds / max(1, len(pilot_s1_ids_ordered))
    n_full_dev = len(dev_ids)
    results["estimated_full_scale_retrieval_minutes"] = (seconds_per_query * n_full_dev) / 60.0

    if residual_rescue_rate >= 0.15:
        status = "SELECTED_CONDITIONAL_FULL_SCALE_RECOMMENDED"
    elif residual_rescue_rate >= 0.03:
        status = "CONDITIONAL_RESIDUAL_ONLY"
    else:
        status = "DEFERRED_REJECTED"
    results["status"] = status

    write_manifest(results, B002_DIR / "b002_results.json")
    print(f"[B002] residual_rescue_rate={residual_rescue_rate:.4f} status={status}")
    print(f"[B002] estimated full-scale forward retrieval: {results['estimated_full_scale_retrieval_minutes']:.1f} min")


if __name__ == "__main__":
    main()
