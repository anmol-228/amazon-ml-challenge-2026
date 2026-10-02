"""Small fixtures for the published Stage-2 country-slice dependency."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_judge_slices.py"
sys.path.insert(0, str(SCRIPT.parent))
from build_judge_slices import build_slices


@pytest.fixture
def inputs():
    split = pd.DataFrame({"source1_entity_id": ["B", "A", "C"],
                          "phase3_split": ["dev_eval", "matcher_train", "dev_eval"]})
    source = pd.DataFrame({"entity_id": ["A", "C", "B"], "country": ["US", "India", "US"]})
    return split, source


def test_schema_values_and_deterministic_order(inputs):
    table = build_slices(*inputs)
    assert table.schema == pa.schema([("s1_entity_id", pa.string()), ("country", pa.string())])
    assert table.to_pydict() == {"s1_entity_id": ["B", "C"], "country": ["US", "India"]}
    assert table.equals(build_slices(*inputs))
    assert table.equals(build_slices(inputs[0], inputs[1].iloc[::-1]))


@pytest.mark.parametrize("problem", ["duplicate_split", "duplicate_source", "missing_country", "missing_entity", "empty_dev"])
def test_invalid_inputs_fail(inputs, problem):
    split, source = (x.copy() for x in inputs)
    if problem == "duplicate_split":
        split.loc[2, "source1_entity_id"] = "B"
    elif problem == "duplicate_source":
        source.loc[1, "entity_id"] = "B"
    elif problem == "missing_country":
        source.loc[2, "country"] = ""
    elif problem == "missing_entity":
        source = source[source.entity_id != "B"]
    else:
        split["phase3_split"] = "matcher_train"
    with pytest.raises(ValueError):
        build_slices(split, source)


def test_command_from_unrelated_directory_and_no_overwrite(inputs, tmp_path):
    split, source = inputs
    split_path, source_path = tmp_path / "split.tsv", tmp_path / "source.tsv"
    split.to_csv(split_path, sep="\t", index=False)
    source.to_csv(source_path, sep="\t", index=False)
    command = [sys.executable, "-B", str(SCRIPT), "--split", str(split_path), "--source1", str(source_path)]
    outputs = [tmp_path / "one/slices.parquet", tmp_path / "two/slices.parquet"]
    for output in outputs:
        subprocess.run(command + ["--output", str(output)], cwd=tmp_path, check=True, capture_output=True)
        assert pq.read_table(output).equals(build_slices(*inputs))
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    before = hashlib.sha256(outputs[0].read_bytes()).digest()
    rerun = subprocess.run(command + ["--output", str(outputs[0])], cwd=tmp_path, capture_output=True)
    assert rerun.returncode != 0
    assert hashlib.sha256(outputs[0].read_bytes()).digest() == before
