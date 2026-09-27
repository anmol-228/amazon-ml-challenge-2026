"""Fit the final decoder: per-entity adaptive thresholds on stage-2 probabilities (calibration only, no training).

Candidate rule families (label-free conditioning, no country identity):
  rank       top candidate of the Source-1 entity vs the others
  source     target record from Source 2 vs Source 3
  conf_size  min(#candidates of the entity with p >= 0.90, 2)
  pred_size  min(#candidates of the entity with p >= 0.71, 2)
  cand_size  #candidates of the entity: < 10, 10-29, >= 30
  rank_source, rank_conf_size, source_conf_size (products of the above)
  geometric  exp((1-a) log p2 + a log p1) with one global threshold (a in .05/.10/.20/.35; also for rank, source)
For each family the per-group thresholds (grid 0.40-0.95) are fitted by exact coordinate ascent of macro F0.5 on
calibration_holdout. The family with the best calibration score is frozen BEFORE dev_eval is loaded; dev_eval is
then scored once against the global-threshold baseline with a paired bootstrap over entities (2000 resamples).

Usage: python -u scripts/fit_adaptive_decoder.py --feat feat_dev_r3idf --stage1 m003_r3mix --stage2 s2x_r3mix --out decoder_c1
Writes experiments/m003/<out>/{calibration_rules.json, frozen_rule.json, result.json, entity_dev.parquet}.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_eval import per_entity_f  # noqa: E402
from train_m003 import ROOT, entity_codes, population  # noqa: E402

M = ROOT / "experiments" / "m003"
GRID = np.round(np.arange(.40, .951, .01), 2)
START = time.monotonic()


def log(*args) -> None:
    print(f"[{time.monotonic() - START:.1f}s]", *args, flush=True)


def counts_f(tp, pred, truth):
    den = pred + .25 * truth
    return np.divide(1.25 * tp, den, out=((pred == 0) & (truth == 0)).astype(float), where=den > 0)


def structure(ent, p, n, source, total_count=None):
    """Per-row conditioning codes; all label-free and country-independent."""
    idx = np.arange(len(ent))
    order = np.lexsort((idx, -p, ent))
    first = np.r_[True, ent[order][1:] != ent[order][:-1]]
    top = np.zeros(len(ent), dtype=bool)
    top[order[first]] = True
    confident = np.bincount(ent[p >= .90], minlength=n)
    baseline_size = np.bincount(ent[p >= .71], minlength=n)
    if total_count is None:
        total_count = np.bincount(ent, minlength=n)
    return dict(rank=(~top).astype(np.int8), source=source.astype(np.int8),
                conf_size=np.minimum(confident[ent], 2).astype(np.int8),
                pred_size=np.minimum(baseline_size[ent], 2).astype(np.int8),
                cand_size=np.searchsorted([10, 30], total_count[ent], side="right").astype(np.int8))


def load_split(split, feat, stage1, stage2):
    t = pq.read_table(feat / f"features_{split}.parquet", columns=["s1_idx", "t_idx", "label"])
    p = np.load(M / stage2 / f"pred2_{split}.npy", mmap_mode="r")
    p1 = np.load(M / stage1 / f"pred_{split}.npy", mmap_mode="r")
    assert t.num_rows == len(p) == len(p1)
    ids, pop_idx, truth = population(split, feat)
    ent = entity_codes(t["s1_idx"].to_numpy(), pop_idx).astype(np.int32)
    assert ent.min() >= 0
    total_count = np.bincount(ent, minlength=len(ids))
    # every rule on the grid needs p2 >= .40**(1/.65) > .24 (alpha <= .35), so rows below .10 are never selectable
    keep = p >= .10
    tid = t["t_idx"].to_numpy()[keep]
    label = t["label"].to_numpy()[keep].astype(bool)
    ent = ent[keep]
    p, p1 = np.asarray(p[keep], dtype=np.float64), np.asarray(p1[keep], dtype=np.float64)
    del t
    target = pq.read_table(feat / "t_ids.parquet", columns=["entity_id"])["entity_id"]
    src = pc.starts_with(target, "S3-").to_numpy(zero_copy_only=False)[tid]
    data = dict(ids=ids, ent=ent, label=label, p=p, p1=p1, truth=truth)
    data["structure"] = structure(ent, p, len(ids), src, total_count)
    log("loaded", split, "entities", len(ids), "retained candidate rows", len(ent))
    return data


def group_codes(data, family):
    s = data["structure"]
    if family in s:
        return s[family]
    if family == "rank_source":
        return 2 * s["rank"] + s["source"]
    if family == "rank_conf_size":
        return 3 * s["rank"] + s["conf_size"]
    if family == "source_conf_size":
        return 3 * s["source"] + s["conf_size"]
    if family == "geometric":
        return np.zeros(len(data["ent"]), dtype=np.int8)
    raise ValueError(family)


def scores(data, alpha):
    if alpha == 0:
        return data["p"]
    return np.exp((1 - alpha) * np.log(np.maximum(data["p"], 1e-30)) + alpha * np.log(np.maximum(data["p1"], 1e-30)))


def selection(data, rule):
    return scores(data, rule["alpha"]) >= np.asarray(rule["thresholds"])[group_codes(data, rule["family"])]


def curve(data, q, group, target_group, thresholds):
    """Exact macro F0.5 for every grid threshold of one group, the other groups held fixed (event sweep)."""
    ent, lab, truth = data["ent"], data["label"], data["truth"]
    n = len(truth)
    others = (group != target_group) & (q >= thresholds[group])
    bp = np.bincount(ent[others], minlength=n)
    bt = np.bincount(ent[others & lab], minlength=n)
    base = counts_f(bt, bp, truth).sum()
    rows = np.flatnonzero(group == target_group)
    order = rows[np.lexsort((rows, -q[rows], ent[rows]))]
    e = ent[order]
    if len(e) == 0:
        return np.full(len(GRID), base / n)
    starts = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1]
    ends = np.r_[starts[1:], len(e)]
    cum = np.cumsum(lab[order], dtype=np.int64)
    before = np.r_[0, cum[starts[1:] - 1]]
    tp_added = cum - np.repeat(before, ends - starts)
    pred_added = np.arange(len(e)) - np.repeat(starts, ends - starts) + 1
    tp, pred = bt[e] + tp_added, bp[e] + pred_added
    delta = counts_f(tp, pred, truth[e]) - counts_f(tp - lab[order], pred - 1, truth[e])
    bins = np.searchsorted(GRID, q[order], side="right")  # grid values <= row score, ties included
    bybin = np.bincount(bins, weights=delta, minlength=len(GRID) + 1)
    return (base + np.cumsum(bybin[::-1])[::-1][1:]) / n


def fit_rule(data, family, base_thr, alpha=0.):
    group, q = group_codes(data, family), scores(data, alpha)
    thresholds = np.full(int(group.max()) + 1, base_thr)
    best = None
    for start in [base_thr, .60]:  # two fixed starts, deterministic tie-breaking towards the global threshold
        thresholds[:] = start
        for _ in range(5):
            old = thresholds.copy()
            for k in range(len(thresholds)):
                values = curve(data, q, group, k, thresholds)
                choices = np.flatnonzero(values >= values.max() - 1e-13)
                thresholds[k] = GRID[choices[np.argmin(np.abs(GRID[choices] - base_thr))]]
            if np.array_equal(old, thresholds):
                break
        rule = dict(family=family, alpha=alpha, thresholds=thresholds.tolist())
        rule["calibration"] = float(per_entity_f(data["ent"], data["label"], selection(data, rule), data["truth"]).mean())
        if best is None or rule["calibration"] > best["calibration"]:
            best = rule
    log("fit", best)
    return best


def bootstrap(d):
    rng = np.random.default_rng(12345)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return np.percentile(means, [2.5, 97.5]).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="feat_dev_r3idf")
    ap.add_argument("--stage1", default="m003_r3mix", help="dir with pred_{split}.npy (stage-1 probabilities)")
    ap.add_argument("--stage2", default="s2x_r3mix", help="dir with pred2_{split}.npy and summary.json")
    ap.add_argument("--out", default="decoder_c1")
    args = ap.parse_args()
    feat, out = M / args.feat, M / args.out
    out.mkdir(parents=True, exist_ok=False)
    base_thr = float(json.load(open(M / args.stage2 / "summary.json", encoding="utf-8"))["summary"]["threshold"])

    cal = load_split("calibration_holdout", feat, args.stage1, args.stage2)
    baseline = float(per_entity_f(cal["ent"], cal["label"], cal["p"] >= base_thr, cal["truth"]).mean())
    g, thr = group_codes(cal, "rank"), np.array([base_thr, base_thr])
    cv = curve(cal, cal["p"], g, 0, thr)
    for x in [.4, base_thr, .95]:  # event-sweep objective == official per-entity implementation
        thr[0] = x
        official = per_entity_f(cal["ent"], cal["label"], cal["p"] >= thr[g], cal["truth"]).mean()
        assert abs(cv[np.argmin(abs(GRID - x))] - official) < 1e-10
    families = ["rank", "source", "conf_size", "pred_size", "cand_size", "rank_source", "rank_conf_size",
                "source_conf_size"]
    rules = [fit_rule(cal, f, base_thr) for f in families]
    for alpha in [.05, .10, .20, .35]:
        for family in ["geometric", "rank", "source"]:
            rules.append(fit_rule(cal, family, base_thr, alpha))
    chosen = max(rules, key=lambda r: (r["calibration"], -len(r["thresholds"]), -r["alpha"]))
    json.dump({"baseline": baseline, "global_threshold": base_thr, "rules": rules, "chosen": chosen,
               "selection_policy": "maximum calibration macro F0.5; ties prefer fewer parameters"},
              open(out / "calibration_rules.json", "w", encoding="utf-8"), indent=2)
    json.dump(chosen, open(out / "frozen_rule.json", "w", encoding="utf-8"), indent=2)
    log("FROZEN BEFORE DEV", chosen)
    del cal
    gc.collect()

    dev = load_split("dev_eval", feat, args.stage1, args.stage2)
    e, y, truth = dev["ent"], dev["label"], dev["truth"]
    sel, base_sel = selection(dev, chosen), dev["p"] >= base_thr
    f, base_f = per_entity_f(e, y, sel, truth), per_entity_f(e, y, base_sel, truth)
    d = f - base_f
    result = dict(rule=chosen, dev=float(f.mean()), baseline_dev=float(base_f.mean()), delta=float(d.mean()),
                  ci95=bootstrap(d), bootstrap_reps=2000, bootstrap_seed=12345,
                  fp_change=int(np.sum(sel & ~y) - np.sum(base_sel & ~y)),
                  fn_change=int(np.sum(base_sel & y) - np.sum(sel & y)), changed_entities=int(np.count_nonzero(d)))
    pd.DataFrame(dict(s1_entity_id=dev["ids"], f=f, baseline_f=base_f, delta=d,
                      tp=np.bincount(e[sel & y], minlength=len(f)), pred=np.bincount(e[sel], minlength=len(f)),
                      truth=truth)).to_parquet(out / "entity_dev.parquet")
    json.dump(result, open(out / "result.json", "w", encoding="utf-8"), indent=2)
    log("RESULT", result)


if __name__ == "__main__":
    main()
