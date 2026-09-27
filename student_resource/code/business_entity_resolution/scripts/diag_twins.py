"""DIAG-TWIN: are R2 stage-2 address-missing FNs exact same-name twins (unrecoverable from the
target alone) or distinguishable? dev_eval only, post-hoc diagnostic (no production use).

For each dev_eval true candidate row with t_addr_missing, compare the true owner's v2-normalized
name with every competing candidate S1 for the same target (competitors from ALL splits of the
train-corpus candidate table). Buckets: TP / FN at the promoted threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalization_v2 import normalize_name_v2  # noqa: E402
from train_m003 import ROOT  # noqa: E402

M = ROOT / "experiments" / "m003"
FEAT = M / "feat_dev_r2"
THR = 0.69


def main() -> None:
    names = ["matcher_train", "calibration_holdout", "dev_eval"]
    tabs = {n: pq.read_table(FEAT / f"features_{n}.parquet", columns=["s1_idx", "t_idx"]).to_pandas() for n in names}
    allrows = pd.concat([tabs[n] for n in names], ignore_index=True)
    dev = pq.read_table(FEAT / "features_dev_eval.parquet",
                        columns=["s1_idx", "t_idx", "label", "t_addr_missing", "name_exact", "is_source2"]).to_pandas()
    dev["p2"] = np.load(M / "m004_r2" / "pred2_dev_eval.npy")
    pos = dev[dev["label"].astype(bool) & (dev["t_addr_missing"] > 0)].copy()
    pos["bucket"] = np.where(pos["p2"] >= THR, "TP", "FN")
    s1_ids = pq.read_table(FEAT / "s1_ids.parquet")["entity_id"].to_pylist()
    src = pd.read_csv(ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv", sep="\t", dtype=str,
                      usecols=["entity_id", "business_name"]).set_index("entity_id")["business_name"]
    nm = pd.Series(s1_ids).map(src).fillna("").map(normalize_name_v2).to_numpy(dtype=object)
    comp = allrows[allrows["t_idx"].isin(pos["t_idx"])]
    m = pos[["s1_idx", "t_idx", "bucket", "p2"]].merge(comp, on="t_idx", suffixes=("", "_c"))
    m = m[m["s1_idx_c"] != m["s1_idx"]]
    m["twin"] = nm[m["s1_idx"].to_numpy()] == nm[m["s1_idx_c"].to_numpy()]
    g = m.groupby(["s1_idx", "t_idx"]).agg(n_comp=("twin", "size"), n_twin=("twin", "sum")).reset_index()
    pos = pos.merge(g, on=["s1_idx", "t_idx"], how="left").fillna({"n_comp": 0, "n_twin": 0})
    out = {}
    for b, d in pos.groupby("bucket"):
        out[b] = {"n": int(len(d)), "has_exact_twin": float((d["n_twin"] > 0).mean()),
                  "mean_twins": float(d["n_twin"].mean()), "mean_competitors": float(d["n_comp"].mean()),
                  "name_exact_true_owner": float((d["name_exact"] > 0).mean()),
                  "source2_share": float(d["is_source2"].mean()),
                  "p2_ge_0.4": float((d["p2"] >= 0.4).mean()),
                  "no_twin_and_p2_ge_0.3": int(((d["n_twin"] == 0) & (d["p2"] >= 0.3)).sum()),
                  "no_twin": int((d["n_twin"] == 0).sum())}
    print(json.dumps(out, indent=1))
    (M / "diag_twins.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
