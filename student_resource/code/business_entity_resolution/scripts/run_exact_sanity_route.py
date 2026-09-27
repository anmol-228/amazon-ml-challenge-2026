"""Deterministic exact-match sanity route (mission spec section 12).

Cheap, high-specificity normalized-field-equality candidates -- no domain
dictionaries, no country-specific mappings, no promotion to a final match
decision. Computed once, standalone, independent of the TF-IDF caches so it
can run at any point in the sequence (in practice, run early since it is
the cheapest route and its own collision-size check is a useful sanity
signal before the heavier routes are trusted).

Run: .venv/Scripts/python.exe scripts/run_exact_sanity_route.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_exact import exact_candidates  # noqa: E402
from blocking_io import write_candidates, write_manifest  # noqa: E402
from blocking_metrics import build_id_sets_by_s1, compute_blocking_metrics  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXACT_DIR = PROJECT_ROOT / "experiments" / "blocking" / "EXACT"


def main() -> None:
    print("[EXACT] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)
    truth_by_s1 = {s1: t for s1, t in data.truth_by_s1.items() if s1 in dev_set}

    dev_s1_df = data.source1[data.source1["entity_id"].isin(dev_set)]
    target_ids = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
    target_names = list(data.source2["business_name"]) + list(data.source3["business_name"])
    target_addrs = list(data.source2["business_address"]) + list(data.source3["business_address"])
    n_s2, n_s3 = len(data.source2), len(data.source3)

    print(f"[EXACT] computing exact candidates for {len(dev_s1_df):,} dev S1 rows "
          f"against {len(target_ids):,} target rows...")
    with ResourceMonitor() as mon:
        cand_df = exact_candidates(
            s1_ids=dev_s1_df["entity_id"].tolist(),
            s1_names=dev_s1_df["business_name"].tolist(),
            s1_addresses=dev_s1_df["business_address"].tolist(),
            target_ids=target_ids,
            target_names=target_names,
            target_addresses=target_addrs,
        )

    cand_by_s1 = build_id_sets_by_s1(cand_df)
    m_all = compute_blocking_metrics(dev_ids, truth_by_s1, cand_by_s1, n_s2_total=n_s2, n_s3_total=n_s3)

    name_addr_only = cand_df[cand_df["route_exact_name_address"]]
    cand_by_s1_na = build_id_sets_by_s1(name_addr_only)
    m_name_addr = compute_blocking_metrics(dev_ids, truth_by_s1, cand_by_s1_na, n_s2_total=n_s2, n_s3_total=n_s3)

    EXACT_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(cand_df, EXACT_DIR / "exact_candidates.parquet")
    write_manifest(
        {
            "resource": mon.report.as_dict(),
            "metrics_name_and_address_or_name_alone_union": m_all.as_dict(),
            "metrics_name_address_exact_only": m_name_addr.as_dict(),
            "n_rows": len(cand_df),
        },
        EXACT_DIR / "exact_results.json",
    )
    print(f"[EXACT] done. name+address exact link_recall={m_name_addr.link_recall:.4f}, "
          f"n_rows={len(cand_df):,}, resource={mon.report.as_dict()}")


if __name__ == "__main__":
    main()
