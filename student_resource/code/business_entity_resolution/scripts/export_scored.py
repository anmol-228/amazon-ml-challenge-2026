"""Write the two output TSVs from a test run's scored.parquet (produced by
run_m003_test.py or stage-2 inference). Usage:
  python -u export_scored.py experiments/m003/test_runs/<run>/scored.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_io import TEST_PATHS, load_source_table  # noqa: E402
from export_outputs import export  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    scored = Path(sys.argv[1])
    if not scored.is_absolute():
        scored = ROOT / scored
    tb = pq.read_table(scored, columns=["s1_entity_id", "target_entity_id", "selected"])
    s1_order = load_source_table(TEST_PATHS["source1"], usecols=["entity_id"])["entity_id"].tolist()
    targets = (load_source_table(TEST_PATHS["source2"], usecols=["entity_id"])["entity_id"].tolist()
               + load_source_table(TEST_PATHS["source3"], usecols=["entity_id"])["entity_id"].tolist())
    meta = {"scored_parquet": str(scored)}
    run_summary = scored.parent / "summary_in.json"
    if run_summary.exists():
        meta.update(json.load(open(run_summary, encoding="utf-8")))
    summary = export(tb["s1_entity_id"], tb["target_entity_id"], tb["selected"].to_numpy(), s1_order, targets,
                     ROOT / "student_resource" / "output", meta)
    with open(scored.parent / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
