"""Integrity checks against the actual frozen experiments/splits/validation_v1.tsv
artifact (not synthetic data). Skipped automatically if the raw dataset or the
frozen split file is not present in this checkout, so the suite stays runnable
without the (large, gitignored) dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_io import TRAIN_PATHS  # noqa: E402
from validation_split import manifest_sha256  # noqa: E402

import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "validation_v1.tsv"

pytestmark = pytest.mark.skipif(
    not (SPLIT_PATH.exists() and TRAIN_PATHS["source1"].exists()),
    reason="frozen split file or raw train_source1.tsv not present in this checkout",
)


def _load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)


def test_every_train_s1_appears_exactly_once():
    split = _load_split()
    s1_ids = pd.read_csv(TRAIN_PATHS["source1"], sep="\t", dtype=str, usecols=["entity_id"])["entity_id"]
    assert split["source1_entity_id"].is_unique
    assert set(split["source1_entity_id"]) == set(s1_ids)


def test_split_values_are_only_development_or_validation():
    split = _load_split()
    assert set(split["split"].unique()) <= {"development", "validation"}


def test_manifest_hash_matches_frozen_value():
    split = _load_split()
    assert manifest_sha256(split) == "f9772dd4fe8b515657c620b0952f933902ae2dd77ab736887162bdea0a2fc7e0"


def test_split_proportions_close_to_80_20():
    split = _load_split()
    val_frac = (split["split"] == "validation").mean()
    assert abs(val_frac - 0.20) < 0.001
