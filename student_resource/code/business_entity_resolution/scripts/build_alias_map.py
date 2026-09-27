"""Build the script->Latin name-token alias map used by normalization v2.

Source: ground-truth links of matcher_train Source-1 entities ONLY
(experiments/splits/phase3_split_v1.tsv); dev_eval / calibration_holdout / test
never contribute. For each link whose target name contains Indic-script
characters, the whitespace tokens of the target name (punctuation removed) are
aligned positionally with the word tokens of the Latin Source-1 name when both
have the same token count. A script token is mapped to its most frequent aligned
Latin token when that token has support >= 2 and share >= 0.6.

Usage: python build_alias_map.py [--out PATH]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_io import TRAIN_PATHS, load_ground_truth, load_source_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SPLIT_PATH = ROOT / "experiments" / "splits" / "phase3_split_v1.tsv"
SCRIPT_RE = re.compile("[" + chr(0x0900) + "-" + chr(0x0DFF) + "]")
STRIP_RE = re.compile(r"[^\w" + chr(0x0900) + "-" + chr(0x0DFF) + "]")


def script_tokens(s: str) -> list[str]:
    return [t for t in (STRIP_RE.sub("", x) for x in unicodedata.normalize("NFKC", s).casefold().split()) if t]


def latin_tokens(s: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", s).casefold())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src" / "alias_map_matcher_train.json"))
    args = ap.parse_args()
    import pandas as pd

    split = pd.read_csv(SPLIT_PATH, sep="\t", dtype=str)
    train_s1 = set(split.loc[split["phase3_split"] == "matcher_train", "source1_entity_id"])
    s1 = load_source_table(TRAIN_PATHS["source1"], usecols=["entity_id", "business_name"])
    s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
    t_name = {}
    for key in ("source2", "source3"):
        t = load_source_table(TRAIN_PATHS[key], usecols=["entity_id", "business_name"])
        t_name.update(zip(t["entity_id"], t["business_name"]))
    gt = load_ground_truth()
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_links = 0
    for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s not in train_s1 or not m:
            continue
        for tid in m.split(","):
            tn = t_name[tid]
            if not SCRIPT_RE.search(tn):
                continue
            n_links += 1
            a, b = script_tokens(tn), latin_tokens(s1_name[s])
            if len(a) == len(b):
                for x, y in zip(a, b):
                    counts[x][y] += 1
    mapping = {}
    for x, c in counts.items():
        y, k = c.most_common(1)[0]
        if k >= 2 and k / sum(c.values()) >= 0.6:
            mapping[x] = y
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, sort_keys=True, indent=0)
    print(f"script-name links {n_links:,}; aligned tokens {len(counts):,}; mapped {len(mapping):,} -> {args.out}")


if __name__ == "__main__":
    main()
