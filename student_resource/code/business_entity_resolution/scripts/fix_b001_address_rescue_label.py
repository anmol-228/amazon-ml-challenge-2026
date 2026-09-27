"""Review checkpoint item 5: correct the EXP-B001 zero-name-overlap
diagnostic label.

The existing `b001_results.json` calls every zero-name-token-overlap true
link that the JOINT (name+address) word-TF-IDF forward route recovers
"address_rescued". That label overclaims: because the joint vectorizer
shares ONE vocabulary across the whole corpus, a token string can be a
"name token" for one record and an "address token" for another, so a
retrieval hit on a zero-name-overlap pair could ALSO come from a cross-
field match (S1's address tokens vs. the target's NAME tokens, or vice
versa) rather than genuine address-to-address overlap. This script does
NOT rerun retrieval and does NOT change any recall/coverage number --
it reads the existing `forward_word_k50.parquet` and, for each k in the
grid, explicitly re-derives which zero-name-overlap recovered links have a
verified nonempty normalized ADDRESS-token intersection (real
address-rescue) versus which do not (renamed `joint_lexical_rescued_other`
per the review's option B, applied together with option A's verification).

Run: .venv/Scripts/python.exe scripts/fix_b001_address_rescue_label.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import read_candidates, read_manifest, write_manifest  # noqa: E402
from blocking_metrics import classify_true_links_by_name_overlap  # noqa: E402
from blocking_normalization import name_token_set  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
B001_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B001"

K_GRID = [5, 10, 20, 50]


def main() -> None:
    print("[fix-address-label] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}

    print("[fix-address-label] building name AND address token maps...")
    s1_name_by_id = dict(zip(data.source1["entity_id"], data.source1["business_name"]))
    s1_addr_by_id = dict(zip(data.source1["entity_id"], data.source1["business_address"]))
    target_name_by_id = dict(
        zip(list(data.source2["entity_id"]) + list(data.source3["entity_id"]),
            list(data.source2["business_name"]) + list(data.source3["business_name"]))
    )
    target_addr_by_id = dict(
        zip(list(data.source2["entity_id"]) + list(data.source3["entity_id"]),
            list(data.source2["business_address"]) + list(data.source3["business_address"]))
    )

    s1_name_tokens = {eid: name_token_set(name) for eid, name in s1_name_by_id.items() if eid in dev_set}
    target_name_tokens = {eid: name_token_set(name) for eid, name in target_name_by_id.items()}
    s1_addr_tokens = {eid: name_token_set(addr) for eid, addr in s1_addr_by_id.items() if eid in dev_set}
    target_addr_tokens = {eid: name_token_set(addr) for eid, addr in target_addr_by_id.items()}

    _, zero_truth = classify_true_links_by_name_overlap(truth_by_s1, s1_name_tokens, target_name_tokens)
    n_zero_links = sum(len(v) for v in zero_truth.values())
    print(f"[fix-address-label] {n_zero_links:,} true links have zero name-token overlap (across {len(zero_truth):,} S1 entities)")

    print("[fix-address-label] loading existing forward_word_k50.parquet (no retrieval rerun)...")
    fwd_full = read_candidates(B001_DIR / "forward_word_k50.parquet")

    results_path = B001_DIR / "b001_results.json"
    b001_results = read_manifest(results_path)

    zero_overlap_s1_set = set(zero_truth.keys())
    # Pre-filter to only the S1 entities that actually have a zero-name-
    # overlap true link -- this diagnostic never needs the other entities'
    # candidates, so there is no reason to group the full 87M-row table.
    fwd_relevant = fwd_full[fwd_full["s1_entity_id"].isin(zero_overlap_s1_set)]

    for k in K_GRID:
        sub = fwd_relevant[fwd_relevant["forward_rank"] <= k]
        cand_by_s1: dict[str, set[str]] = {}
        for s1, tid in zip(sub["s1_entity_id"].to_numpy(), sub["target_entity_id"].to_numpy()):
            cand_by_s1.setdefault(s1, set()).add(tid)

        n_verified_address_rescued = 0
        n_joint_lexical_rescued_other = 0
        for s1, targets in zero_truth.items():
            cand = cand_by_s1.get(s1, set())
            found = targets & cand
            if not found:
                continue
            s1_addr_tok = s1_addr_tokens.get(s1, frozenset())
            for t in found:
                t_addr_tok = target_addr_tokens.get(t, frozenset())
                if s1_addr_tok & t_addr_tok:
                    n_verified_address_rescued += 1
                else:
                    n_joint_lexical_rescued_other += 1

        n_total_rescued = n_verified_address_rescued + n_joint_lexical_rescued_other
        entry = b001_results["k_grid"][str(k)]
        # Preserve the original (now-labeled-as-approximate) field so the
        # historical number is not silently lost, but stop presenting it as
        # authoritative; add the verified breakdown alongside it.
        entry["diagnostic_D_LEGACY_label_was_overclaimed_see_verified_breakdown"] = entry.get(
            "diagnostic_D_address_rescued_count"
        )
        entry["diagnostic_D_verified_address_rescued_count"] = n_verified_address_rescued
        entry["diagnostic_D_joint_lexical_rescued_other_count"] = n_joint_lexical_rescued_other
        entry["diagnostic_D_total_zero_name_overlap_rescued_count"] = n_total_rescued
        entry["diagnostic_D_verified_address_rescued_pct_of_zero_overlap"] = (
            n_verified_address_rescued / n_zero_links if n_zero_links else 0.0
        )
        entry["diagnostic_D_joint_lexical_rescued_other_pct_of_zero_overlap"] = (
            n_joint_lexical_rescued_other / n_zero_links if n_zero_links else 0.0
        )
        print(f"[fix-address-label] k={k}: verified_address_rescued={n_verified_address_rescued:,} "
              f"joint_lexical_rescued_other={n_joint_lexical_rescued_other:,} "
              f"(total zero-overlap rescued={n_total_rescued:,} / {n_zero_links:,})")

    write_manifest(b001_results, results_path)
    print(f"[fix-address-label] done. Updated {results_path} in place (link_recall/coverage/etc unchanged).")


if __name__ == "__main__":
    main()
