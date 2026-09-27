"""Blind paired comparison of two runs on the same S1 population.

Inputs: two parquet files with columns s1_entity_id, f (per-entity F0.5),
tp, pred, truth. Run identities are hidden behind RUN_A / RUN_B (the caller
shuffles the order with a seed and records the mapping separately).

Verdict vs pre-registered minimum delta (B - A):
  PROMOTE      mean >= min_delta and CI95 lower > 0
  REJECT       CI95 upper < min_delta or mean <= 0
  INCONCLUSIVE otherwise

Usage: python blind_judge.py A.parquet B.parquet --min-delta 0.01 [--slices slices.parquet]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--min-delta", type=float, required=True)
    ap.add_argument("--slices", default=None, help="parquet: s1_entity_id + slice columns")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    A = pd.read_parquet(args.a).set_index("s1_entity_id")
    B = pd.read_parquet(args.b).set_index("s1_entity_id")
    assert A.index.equals(B.index) or set(A.index) == set(B.index), "populations differ"
    B = B.loc[A.index]
    d = (B["f"] - A["f"]).to_numpy()
    rng = np.random.default_rng(12345)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = d.mean()
    out = {
        "n_entities": int(n), "RUN_A_macro": float(A["f"].mean()), "RUN_B_macro": float(B["f"].mean()),
        "mean_delta": float(mean), "ci95": [float(lo), float(hi)],
        "entities_improved": int((d > 1e-12).sum()), "entities_harmed": int((d < -1e-12).sum()),
        "fp_change": int(((B["pred"] - B["tp"]) - (A["pred"] - A["tp"])).sum()),
        "fn_change": int(((B["truth"] - B["tp"]) - (A["truth"] - A["tp"])).sum()),
        "singleton_acc_A": float(A.loc[A["truth"] == 0, "f"].mean()),
        "singleton_acc_B": float(B.loc[B["truth"] == 0, "f"].mean()),
    }
    if args.slices:
        S = pd.read_parquet(args.slices).set_index("s1_entity_id").loc[A.index]
        out["slices"] = {}
        for col in S.columns:
            out["slices"][col] = {str(k): {"A": float(A["f"][S[col] == k].mean()), "B": float(B["f"][S[col] == k].mean()),
                                           "n": int((S[col] == k).sum())} for k in S[col].unique()}
    # verdict_B_over_A: is RUN_B better than RUN_A by the pre-registered margin?
    if mean >= args.min_delta and lo > 0:
        verdict = "PROMOTE"
    elif hi < args.min_delta or mean <= 0:
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE"
    out["delta_definition"] = "RUN_B minus RUN_A"
    out["verdict_B_over_A"] = verdict
    # symmetric statement: which run is better, and does that run clear the margin over the other?
    winner = "RUN_B" if mean > 0 else "RUN_A"
    w_lo, w_hi = (lo, hi) if mean > 0 else (-hi, -lo)
    out["better_run"] = winner
    out["better_run_margin_ci95"] = [float(w_lo), float(w_hi)]
    out["better_run_clears_min_delta"] = bool(abs(mean) >= args.min_delta and w_lo > 0)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
