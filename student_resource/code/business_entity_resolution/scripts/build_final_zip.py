"""Assemble the final competition ZIP from an immutable staged submission (read-only use).

<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv   (copied from the staged dir, hashes checked
                                                             against its manifest.json)
  code/business_entity_resolution/{src/, scripts/, tests/, README.md, requirements.txt}
  Documentation_template.md                                  (the filled methodology document)

Usage: python build_final_zip.py --staged submissions/STAGED_submission_00X --team TEAM
                                 --doc experiments/docs_draft/Documentation_final.md --out submissions/final_zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
CODE_PARTS = ["src", "scripts", "tests", "artifacts", "README.md", "requirements.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    staged = (ROOT / args.staged).resolve()
    manifest = json.load(open(staged / "manifest.json", encoding="utf-8"))
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        got = sha256(staged / name)
        assert got == manifest["sha256"][name], (name, got)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{args.team}_submission.zip"
    assert not zpath.exists(), f"{zpath} exists; refusing to overwrite"
    listing = []
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(staged / name, f"output/{name}")
            listing.append(f"output/{name}")
        for part in CODE_PARTS:
            p = CODE / part
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc":
                    continue
                arc = "code/business_entity_resolution/" + f.relative_to(CODE).as_posix()
                z.write(f, arc)
                listing.append(arc)
        z.write(ROOT / args.doc, "Documentation_template.md")
        listing.append("Documentation_template.md")
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            h = hashlib.sha256(z.read(f"output/{name}")).hexdigest()
            assert h == manifest["sha256"][name], name
    info = {"zip": str(zpath), "zip_sha256": sha256(zpath), "staged": str(staged), "n_files": len(listing),
            "output_sha256": manifest["sha256"]}
    (out_dir / f"{args.team}_submission.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
