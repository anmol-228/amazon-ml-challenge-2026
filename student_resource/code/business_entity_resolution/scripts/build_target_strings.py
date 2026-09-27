"""Cache v2-normalized target (Source-2/3) name and address strings aligned to a feature dir's t_ids.parquet
(used by the EXP-S2C corroboration features). Usage: python build_target_strings.py dev|test --feat FEATDIR"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_io import TEST_PATHS, TRAIN_PATHS, load_source_table  # noqa: E402
from stage2c import target_strings  # noqa: E402
from train_m003 import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--feat", required=True)
    args = ap.parse_args()
    t0 = time.time()
    paths = TRAIN_PATHS if args.mode == "dev" else TEST_PATHS
    fdir = ROOT / "experiments" / "m003" / args.feat
    ids = pq.read_table(fdir / "t_ids.parquet")["entity_id"].to_pylist()
    tab = pd.concat([load_source_table(paths[k], usecols=["entity_id", "business_name", "business_address"])
                     for k in ("source2", "source3")], ignore_index=True)
    names, addrs = target_strings(ids, tab)
    pq.write_table(pa.table({"name": pa.array(names.tolist(), pa.string()), "addr": pa.array(addrs.tolist(), pa.string())}),
                   fdir / "target_strings_v2.parquet", compression="zstd")
    print(f"[tstr] {args.mode}: {len(ids):,} targets ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
