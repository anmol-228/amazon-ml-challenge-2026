"""Residual blocking-miss analysis (mission spec section 21), scalable.

After the best EXP-B004 union, categorize the true links that STILL fail to
enter the candidate set, using only observable characteristics -- this is
post-hoc diagnostic analysis of an already-fixed candidate set, never a
production trigger (no oracle gating).

Rewritten for scale (the review checkpoint's own lesson from B001R/B003):
the missed-link set itself is bounded (~660K rows out of 6.1M true links,
once B004's 89% recall is subtracted), so per-pair Python-level analysis of
JUST that residual is cheap and safe. What must NOT happen is building a
per-entity Python set/frozenset structure from the 228M-row B004 union
table -- `link_found_mask` (the scalable join-based path) is used instead
to identify exactly which truth edges are missing, without ever
materializing per-entity candidate sets for the full union.

As a bonus over the earlier per-text-feature-only version, this script also
scores each missed pair's actual word-TF-IDF cosine similarity using the
already-cached B000 vectors (a cheap, bounded set of sparse dot products
over only the ~660K residual pairs) to distinguish a "near miss" (positive
similarity, likely just outside forward k=50 or reverse r=5) from a
"wildly dissimilar" pair no larger k/r would have caught.

Run: .venv/Scripts/python.exe scripts/run_residual_analysis.py
Precondition: run_b004_union.py has produced experiments/blocking/B004/union_candidates.parquet.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_identity import target_source_of  # noqa: E402
from blocking_io import read_candidates, read_manifest, write_manifest  # noqa: E402
from blocking_metrics import link_found_mask, truth_edges_dataframe  # noqa: E402
from blocking_normalization import name_token_set, numeric_token_set  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from validation_split import match_bucket  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B004_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B004"
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"
RESIDUAL_DIR = PROJECT_ROOT / "experiments" / "blocking" / "RESIDUAL"


def main() -> None:
    print("[RESIDUAL] loading train data + B004 union...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}
    truth_edges = truth_edges_dataframe(truth_by_s1)
    n_true_links = len(truth_edges)

    union_df = read_candidates(B004_DIR / "union_candidates.parquet")
    with ResourceMonitor() as mon_mask:
        found_mask = link_found_mask(truth_edges, union_df)
    print(f"[RESIDUAL] found_mask computed in {mon_mask.report.wall_seconds:.1f}s, "
          f"peak {mon_mask.report.peak_rss_mb:.0f}MB")
    del union_df

    missed = truth_edges[~found_mask].reset_index(drop=True)
    n_missed = len(missed)
    print(f"[RESIDUAL] {n_missed:,} / {n_true_links:,} true links missed "
          f"({n_missed / n_true_links:.4%})")

    # ---- observable text/metadata features, computed only for the missed set --
    print("[RESIDUAL] building lookup maps for text/metadata features (missed-set only)...")
    name_by_s1 = dict(zip(data.source1["entity_id"], data.source1["business_name"]))
    addr_by_s1 = dict(zip(data.source1["entity_id"], data.source1["business_address"]))
    country_by_s1 = dict(zip(data.source1["entity_id"], data.source1["country"]))

    missed_s1_set = set(missed["s1_entity_id"])
    missed_target_set = set(missed["target_entity_id"])

    target_name_by_id: dict[str, str] = {}
    target_addr_by_id: dict[str, str] = {}
    for src_df in (data.source2, data.source3):
        sub = src_df[src_df["entity_id"].isin(missed_target_set)]
        target_name_by_id.update(zip(sub["entity_id"], sub["business_name"]))
        target_addr_by_id.update(zip(sub["entity_id"], sub["business_address"]))

    match_bucket_by_s1 = {s1: match_bucket(len(truth_by_s1.get(s1, frozenset()))) for s1 in missed_s1_set}

    print("[RESIDUAL] classifying each missed link by observable features...")
    categories = {
        "zero_name_overlap": 0,
        "zero_address_overlap": 0,
        "s1_address_missing": 0,
        "target_address_missing": 0,
        "no_numeric_evidence_either_side": 0,
        "numeric_disagreement": 0,
        "numeric_agreement_but_still_missed": 0,
        "short_s1_name_lte_2_tokens": 0,
    }
    rows = []
    for s1, target in zip(missed["s1_entity_id"], missed["target_entity_id"]):
        s1_name = name_by_s1.get(s1, "") or ""
        s1_addr = addr_by_s1.get(s1, "") or ""
        t_name = target_name_by_id.get(target, "") or ""
        t_addr = target_addr_by_id.get(target, "") or ""

        s1_name_tok = name_token_set(s1_name)
        t_name_tok = name_token_set(t_name)
        s1_addr_tok = name_token_set(s1_addr)
        t_addr_tok = name_token_set(t_addr)
        s1_num = numeric_token_set(s1_addr)
        t_num = numeric_token_set(t_addr)

        zero_name = len(s1_name_tok & t_name_tok) == 0
        zero_addr = len(s1_addr_tok & t_addr_tok) == 0
        s1_addr_missing = s1_addr.strip() == ""
        t_addr_missing = t_addr.strip() == ""
        no_numeric = len(s1_num) == 0 and len(t_num) == 0
        numeric_conflict = len(s1_num) > 0 and len(t_num) > 0 and len(s1_num & t_num) == 0
        numeric_agree = len(s1_num) > 0 and len(t_num) > 0 and len(s1_num & t_num) > 0
        short_name = len(s1_name_tok) <= 2

        if zero_name:
            categories["zero_name_overlap"] += 1
        if zero_addr:
            categories["zero_address_overlap"] += 1
        if s1_addr_missing:
            categories["s1_address_missing"] += 1
        if t_addr_missing:
            categories["target_address_missing"] += 1
        if no_numeric:
            categories["no_numeric_evidence_either_side"] += 1
        if numeric_conflict:
            categories["numeric_disagreement"] += 1
        if numeric_agree:
            categories["numeric_agreement_but_still_missed"] += 1
        if short_name:
            categories["short_s1_name_lte_2_tokens"] += 1

        rows.append(
            {
                "s1_entity_id": s1,
                "target_entity_id": target,
                "target_source": target_source_of(target),
                "zero_name_overlap": zero_name,
                "zero_address_overlap": zero_addr,
                "s1_address_missing": s1_addr_missing,
                "target_address_missing": t_addr_missing,
                "no_numeric_evidence": no_numeric,
                "numeric_disagreement": numeric_conflict,
                "numeric_agreement": numeric_agree,
                "short_s1_name": short_name,
                "country": country_by_s1.get(s1, ""),
                "match_bucket": match_bucket_by_s1.get(s1, ""),
            }
        )

    missed_df = pd.DataFrame(rows)

    # ---- near-miss scoring via cached word vectors (bounded, missed-set only) --
    print("[RESIDUAL] scoring missed pairs against cached word vectors (near-miss check)...")
    # Pickle load is safe here: word_vectorizer.pkl is produced by
    # run_b000_preflight.py in this same local pipeline, never an
    # externally-supplied file. Loaded only to confirm cache integrity;
    # the sparse matrices below (not the vectorizer object) are what we
    # actually score the residual pairs with.
    with open(CACHE_DIR / "word_vectorizer.pkl", "rb") as fh:
        pickle.load(fh)
    s1_dev_matrix = sparse.load_npz(CACHE_DIR / "s1_dev_word_matrix.npz").tocsr()
    target_matrix = sparse.load_npz(CACHE_DIR / "target_word_matrix.npz").tocsr()
    manifest = read_manifest(CACHE_DIR / "word_cache_manifest.json")
    dev_pos = {eid: i for i, eid in enumerate(manifest["dev_s1_ids"])}
    target_pos = {eid: i for i, eid in enumerate(manifest["target_ids"])}

    scores = np.zeros(len(missed_df), dtype="float32")
    for i, (s1, target) in enumerate(zip(missed_df["s1_entity_id"], missed_df["target_entity_id"])):
        si = dev_pos.get(s1)
        ti = target_pos.get(target)
        if si is None or ti is None:
            continue
        scores[i] = float(s1_dev_matrix[si].dot(target_matrix[ti].T)[0, 0])
    missed_df["word_cosine_similarity"] = scores
    near_miss_mask = scores > 0.0
    categories["near_miss_nonzero_word_similarity"] = int(near_miss_mask.sum())
    categories["wildly_dissimilar_zero_word_similarity"] = int((~near_miss_mask).sum())

    # ---- sibling completeness: is this S1 entity partially or wholly missed? ---
    print("[RESIDUAL] computing sibling completeness within multi-match entities...")
    truth_size_by_s1 = {s1: len(t) for s1, t in truth_by_s1.items()}
    missed_count_by_s1 = missed_df.groupby("s1_entity_id").size()
    n_fully_missed_entities = int((missed_count_by_s1 == missed_count_by_s1.index.map(lambda s1: truth_size_by_s1.get(s1, 0))).sum())
    n_partially_missed_entities = len(missed_count_by_s1) - n_fully_missed_entities

    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    missed_df.to_parquet(RESIDUAL_DIR / "residual_missed_links.parquet", index=False)

    results = {
        "n_true_links_total": n_true_links,
        "n_missed_links": n_missed,
        "missed_link_rate": n_missed / n_true_links,
        "n_s1_entities_with_any_missed_link": missed_df["s1_entity_id"].nunique(),
        "n_s1_entities_fully_missed": n_fully_missed_entities,
        "n_s1_entities_partially_missed": n_partially_missed_entities,
        "category_counts": categories,
        "category_rates_of_missed": {k: (v / n_missed if n_missed else 0.0) for k, v in categories.items()},
        "missed_by_country": missed_df["country"].value_counts().to_dict(),
        "missed_by_target_source": missed_df["target_source"].value_counts().to_dict(),
        "missed_by_match_bucket": missed_df["match_bucket"].value_counts().to_dict(),
        "word_cosine_similarity_stats": {
            "mean": float(scores.mean()),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
            "max": float(scores.max()),
        },
    }
    write_manifest(results, RESIDUAL_DIR / "residual_analysis.json")
    print(f"[RESIDUAL] missed {n_missed:,} / {n_true_links:,} true links ({results['missed_link_rate']:.4%})")
    for k, v in categories.items():
        pct = (v / n_missed * 100) if n_missed else 0.0
        print(f"  {k}: {v:,} ({pct:.1f}% of missed)")
    print(f"  fully-missed S1 entities: {n_fully_missed_entities:,}, partially-missed: {n_partially_missed_entities:,}")


if __name__ == "__main__":
    main()
