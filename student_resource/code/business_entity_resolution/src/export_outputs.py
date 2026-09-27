"""Write matching_results.tsv / candidate_pairs.tsv from a scored pair table.

Input: arrays of Source-1 ids, target ids and a boolean `selected` per scored
candidate row. Output rows follow the test Source-1 file order; ids within a
cell are de-duplicated and sorted (byte order); candidate cells hold every
scored pair, matching cells the selected subset (so matches are always a subset
of candidates). Files are written to temporary names and atomically renamed;
OUTPUT_COMPLETE.json is written only after both files are in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(s1_col: pa.ChunkedArray, t_col: pa.ChunkedArray, selected: np.ndarray,
           s1_order: list[str], target_ids: list[str], out_dir: Path, meta: dict) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "OUTPUT_COMPLETE.json"
    if marker.exists():
        marker.unlink()
    s1_arr = pa.array(s1_order, pa.string())
    t_sorted = np.array(sorted(set(target_ids)), dtype=object)
    t_arr = pa.array(t_sorted.tolist(), pa.string())
    si = pc.index_in(s1_col, value_set=s1_arr).to_numpy(zero_copy_only=False)
    ti = pc.index_in(t_col, value_set=t_arr).to_numpy(zero_copy_only=False)
    if np.isnan(si.astype(float)).any() or np.isnan(ti.astype(float)).any():
        raise SystemExit("scored ids not found in the test tables")
    si, ti = si.astype(np.int64), ti.astype(np.int64)
    key = si * len(t_sorted) + ti
    order = np.argsort(key, kind="stable")
    key, si, ti, sel = key[order], si[order], ti[order], selected[order].astype(np.int8)
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(first)
    sel = np.maximum.reduceat(sel, starts).astype(bool) if len(starts) else sel.astype(bool)
    si, ti = si[first], ti[first]
    bounds = np.searchsorted(si, np.arange(len(s1_order) + 1))
    for fname, col, mask in (("candidate_pairs.tsv", "candidate_entity_ids", None),
                             ("matching_results.tsv", "matched_entity_ids", sel)):
        tmp = out_dir / (fname + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            buf = []
            for i, sid in enumerate(s1_order):
                lo, hi = bounds[i], bounds[i + 1]
                tt = ti[lo:hi] if mask is None else ti[lo:hi][mask[lo:hi]]
                buf.append(sid + "\t" + ",".join(t_sorted[tt]) + "\n")
                if len(buf) >= 50_000:
                    fh.write("".join(buf))
                    buf = []
            fh.write("".join(buf))
        os.replace(tmp, out_dir / fname)
    summary = dict(meta)
    summary.update({
        "n_test_s1": len(s1_order),
        "n_distinct_candidate_pairs": int(len(si)),
        "n_s1_with_candidates": int((np.diff(bounds) > 0).sum()),
        "n_selected_pairs": int(sel.sum()),
        "n_s1_with_predicted_match": int(len(np.unique(si[sel]))),
        "matching_results_sha256": sha256_file(out_dir / "matching_results.tsv"),
        "candidate_pairs_sha256": sha256_file(out_dir / "candidate_pairs.tsv"),
        "export_s": time.time() - t0,
    })
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
