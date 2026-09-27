"""Raw dataset loading for the Business Entity Resolution task.

All loaders read TSV files as plain strings with no NA inference, so that
"missing" is unambiguous: it means an empty string, never a pandas NaN caused
by a value that merely looks like a null token (e.g. a business literally
named "NA"). This module never mutates or writes into the source TSV files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STUDENT_RESOURCE_DIR = PROJECT_ROOT / "student_resource"
DATASET_DIR = STUDENT_RESOURCE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

TRAIN_PATHS = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

TEST_PATHS = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}


def load_source_table(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a source1/source2/source3 TSV as an all-string DataFrame.

    keep_default_na/na_filter are disabled so an empty cell reads as ``""``,
    never NaN, matching how the raw files actually represent missingness.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="c",
        usecols=list(usecols) if usecols is not None else None,
    )


def load_ground_truth(path: Path | None = None) -> pd.DataFrame:
    """Load train_ground_truth.tsv as an all-string DataFrame."""
    path = path or TRAIN_PATHS["ground_truth"]
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="c",
    )


def parse_matched_ids(raw: str) -> list[str]:
    """Split a matched_entity_ids cell into a list of IDs.

    An empty string (true singleton) parses to an empty list. IDs are
    comma-separated with no quoting or internal whitespace expected.
    """
    if raw == "":
        return []
    return raw.split(",")


def iter_chunks(path: Path, usecols: Iterable[str] | None = None, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream a source table in chunks for operations that don't need the
    whole file resident at once (e.g. a single-column scan)."""
    yield from pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="c",
        usecols=list(usecols) if usecols is not None else None,
        chunksize=chunksize,
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
