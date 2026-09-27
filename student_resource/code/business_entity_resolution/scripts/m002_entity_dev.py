"""Per-entity dev_eval outcomes of M002 (model_v2 on P3 candidates, threshold
from model_v2/selected_threshold.json), in the blind-judge format.

Output: experiments/m003/baselines/entity_dev_m002.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_io import load_ground_truth  # noqa: E402
from fast_eval import per_entity_f  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "experiments" / "phase3" / "model_v2"
FEAT = ROOT / "experiments" / "phase3" / "features_v2" / "dev_eval.parquet"
SPLIT = ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"
OUT = ROOT / "experiments" / "m003" / "baselines"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = json.load(open(MODEL / "feature_columns.json", encoding="utf-8"))
    thr = json.load(open(MODEL / "selected_threshold.json", encoding="utf-8"))["threshold"]
    booster = lgb.Booster(model_file=str(MODEL / "lgbm_model.txt"))
    split = pd.read_csv(SPLIT, sep="\t", dtype=str)
    ids = split.loc[split["phase3_split"] == "dev_eval", "source1_entity_id"].tolist()
    id_arr = pa.array(ids, pa.string())
    gt = load_ground_truth()
    sizes = dict(zip(gt["source1_entity_id"], [0 if m == "" else m.count(",") + 1 for m in gt["matched_entity_ids"]]))
    truth = np.array([sizes.get(i, 0) for i in ids], dtype=np.int64)
    pf = pq.ParquetFile(FEAT)
    E, L, SEL = [], [], []
    for i in range(pf.metadata.num_row_groups):
        tb = pf.read_row_group(i, columns=["s1_entity_id", "label"] + cols)
        df = tb.select(cols).to_pandas()
        for c in cols:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int8)
        p = booster.predict(df[cols])
        e = pc.index_in(tb["s1_entity_id"], value_set=id_arr).to_numpy(zero_copy_only=False)
        assert not np.isnan(e.astype(float)).any()
        E.append(e.astype(np.int64))
        L.append(tb["label"].to_numpy(zero_copy_only=False).astype(bool))
        SEL.append(p >= thr)
    ent, lab, sel = np.concatenate(E), np.concatenate(L), np.concatenate(SEL)
    n = len(ids)
    f = per_entity_f(ent, lab, sel, truth)
    pd.DataFrame({"s1_entity_id": ids, "f": f, "tp": np.bincount(ent[sel & lab], minlength=n),
                  "pred": np.bincount(ent[sel], minlength=n), "truth": truth}).to_parquet(OUT / "entity_dev_m002.parquet")
    print(f"M002 dev_eval macro F0.5 @ {thr}: {f.mean():.6f} (reported 0.894594)")


if __name__ == "__main__":
    main()
