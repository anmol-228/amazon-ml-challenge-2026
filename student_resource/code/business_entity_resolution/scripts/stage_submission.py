"""Stage student_resource/output/{matching_results,candidate_pairs}.tsv as an
immutable submissions/STAGED_submission_<NNN>/ package.

Refuses to overwrite an existing package. Refuses to stage unless the official
validator passes AND an independent streaming check confirms: every test S1
exactly once, no duplicate ids within a cell, every match contained in that
S1's candidate cell, all ids carry an S2-/S3- prefix.

Usage: python -u stage_submission.py NNN --desc "..." --meta path.json [--meta path.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SR = ROOT / "student_resource"
OUT = SR / "output"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def independent_check(matching: Path, candidate: Path, test_s1: Path) -> dict:
    with open(test_s1, encoding="utf-8") as fh:
        next(fh)
        required = [line.split("\t", 1)[0] for line in fh if line.strip()]
    req_set = set(required)
    stats = {"required": len(required), "rows": 0, "nonempty": 0, "n_matches": 0, "n_candidates": 0,
             "subset_violations": 0, "dup_in_cell": 0, "bad_prefix": 0, "order_mismatch": 0}
    seen = set()
    with open(matching, encoding="utf-8") as fm, open(candidate, encoding="utf-8") as fc:
        hm, hc = next(fm).rstrip("\n").split("\t"), next(fc).rstrip("\n").split("\t")
        assert hm == ["source1_entity_id", "matched_entity_ids"], hm
        assert hc == ["source1_entity_id", "candidate_entity_ids"], hc
        for lm, lc in zip(fm, fc):
            sm, mm = lm.rstrip("\n").split("\t")
            sc, cc = lc.rstrip("\n").split("\t")
            if sm != sc:
                stats["order_mismatch"] += 1
                continue
            stats["rows"] += 1
            if sm in seen or sm not in req_set:
                raise SystemExit(f"duplicate or unknown S1 {sm}")
            seen.add(sm)
            m = mm.split(",") if mm else []
            c = cc.split(",") if cc else []
            stats["n_matches"] += len(m)
            stats["n_candidates"] += len(c)
            stats["nonempty"] += bool(m)
            if len(set(m)) != len(m) or len(set(c)) != len(c):
                stats["dup_in_cell"] += 1
            if not set(m) <= set(c):
                stats["subset_violations"] += 1
            stats["bad_prefix"] += sum(not (x.startswith("S2-") or x.startswith("S3-")) for x in m)
    stats["missing_s1"] = len(req_set - seen)
    stats["ok"] = (stats["rows"] == len(required) and stats["missing_s1"] == 0 and stats["subset_violations"] == 0
                   and stats["dup_in_cell"] == 0 and stats["bad_prefix"] == 0 and stats["order_mismatch"] == 0)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n")
    ap.add_argument("--desc", required=True)
    ap.add_argument("--meta", action="append", default=[])
    args = ap.parse_args()
    dest = ROOT / "submissions" / f"STAGED_submission_{args.n}"
    if dest.exists():
        raise SystemExit(f"{dest} exists; staged packages are immutable")
    matching, candidate = OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv"

    res = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                          "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"],
                         cwd=str(SR), capture_output=True, text=True)
    print(res.stdout, res.stderr)
    if res.returncode != 0:
        raise SystemExit("official validator FAILED; not staging")
    res_ids = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                              "--test-dir", "dataset/test", "--check-ids"], cwd=str(SR), capture_output=True, text=True)
    print(res_ids.stdout, res_ids.stderr)
    if res_ids.returncode != 0:
        raise SystemExit("official validator with --check-ids FAILED; not staging")
    chk = independent_check(matching, candidate, SR / "dataset" / "test" / "test_source1.tsv")
    print(json.dumps(chk, indent=2))
    if not chk["ok"]:
        raise SystemExit("independent check FAILED; not staging")

    hashes = {"matching_results.tsv": sha256_file(matching), "candidate_pairs.tsv": sha256_file(candidate)}
    dest.mkdir(parents=True)
    shutil.copy2(matching, dest / "matching_results.tsv")
    shutil.copy2(candidate, dest / "candidate_pairs.tsv")
    for name, h in hashes.items():
        if sha256_file(dest / name) != h:
            raise SystemExit(f"copy hash mismatch for {name}")
    meta = {}
    for m in args.meta:
        with open(m, encoding="utf-8") as fh:
            meta[Path(m).name] = json.load(fh)
    manifest = {"submission_local_id": f"SUB-{args.n}", "staged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "description": args.desc, "sha256": hashes, "independent_check": chk, "meta": meta}
    with open(dest / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(dest / "validation.txt", "w", encoding="utf-8") as fh:
        fh.write(res.stdout + res.stderr + "\n--- matching_results with --check-ids ---\n" + res_ids.stdout + res_ids.stderr)
    print(f"STAGED {dest}\n" + json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
