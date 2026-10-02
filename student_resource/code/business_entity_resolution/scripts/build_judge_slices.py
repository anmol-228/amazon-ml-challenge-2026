"""Build country slices for Stage-2 held-out comparisons from documented inputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE_ROOT / "src"))
from data_io import TRAIN_PATHS, load_source_table  # noqa: E402


def build_slices(split: pd.DataFrame, source1: pd.DataFrame) -> pa.Table:
    """Return dev_eval IDs in frozen split order with their original countries.

    Consumers align by entity ID; no labels, predictions or model files are used.
    """
    if split["source1_entity_id"].isna().any() or not split["source1_entity_id"].is_unique:
        raise ValueError("Split entity IDs must be non-null and unique")
    if source1["entity_id"].isna().any() or not source1["entity_id"].is_unique:
        raise ValueError("Source-1 entity IDs must be non-null and unique")
    ids = split.loc[split["phase3_split"] == "dev_eval", "source1_entity_id"].tolist()
    if not ids:
        raise ValueError("Split contains no dev_eval entities")
    countries = source1.set_index("entity_id")["country"].reindex(ids)
    if countries.isna().any() or (countries == "").any():
        raise ValueError("Every dev_eval entity must have a Source-1 country")
    return pa.table({
        "s1_entity_id": pa.array(ids, type=pa.string()),
        "country": pa.array(countries.tolist(), type=pa.string()),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, default=CODE_ROOT / "artifacts/splits/phase3_split_v1.tsv")
    ap.add_argument("--source1", type=Path, default=TRAIN_PATHS["source1"])
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/judge/slices.parquet")
    args = ap.parse_args()
    split = pd.read_csv(args.split, sep="\t", dtype=str, keep_default_na=False)
    source1 = load_source_table(args.source1, usecols=["entity_id", "country"])
    table = build_slices(split, source1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve any retained artifact instead of silently replacing it.
    with args.output.open("xb") as output:
        pq.write_table(table, output, compression="snappy")
    print(f"Wrote {table.num_rows} country slices to {args.output}")


if __name__ == "__main__":
    main()
