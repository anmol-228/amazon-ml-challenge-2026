"""Candidate artifact I/O: columnar, chunkable, memory-efficient storage.

Parquet is used (already available via the project's pinned `pyarrow`
dependency) instead of any giant in-memory Python object structure. Route
provenance/config lives once in a JSON manifest alongside the parquet file,
never duplicated per candidate row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_candidates(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def read_candidates(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)


def read_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
