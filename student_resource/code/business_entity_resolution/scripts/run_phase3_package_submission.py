"""Assemble the submission-#1 staging package after test_pipeline output
exists and the official validator passes. Does NOT create the immutable
submissions/DAY<n>_SUB<nn>/ snapshot (per submissions/README.md, that only
happens after an actual portal upload) -- this only prepares a clean staging
copy plus config/metrics/manifest for the human to review before uploading.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = PROJECT_ROOT / "student_resource" / "code" / "business_entity_resolution"
STUDENT_RESOURCE_DIR = PROJECT_ROOT / "student_resource"
OUTPUT_DIR = STUDENT_RESOURCE_DIR / "output"
TEST_DIR = PROJECT_ROOT / "experiments" / "phase3" / "test"
MODEL_DIR = PROJECT_ROOT / "experiments" / "phase3" / "model"
STAGING_DIR = PROJECT_ROOT / "submissions" / "STAGED_submission_001"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    for p in (matching_path, candidate_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist yet. Run run_phase3_test_pipeline.py first.")
            sys.exit(1)

    print("Running official validator...")
    result = subprocess.run(
        [sys.executable, "utils/validate_submission.py",
         "--matching", "output/matching_results.tsv",
         "--candidate", "output/candidate_pairs.tsv",
         "--test-dir", "dataset/test"],
        cwd=str(PROJECT_ROOT / "student_resource"),
        capture_output=True, text=True,
    )
    print(result.stdout)
    print(result.stderr)
    validator_pass = result.returncode == 0
    print(f"Validator exit code: {result.returncode} ({'PASS' if validator_pass else 'FAIL'})")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged_matching = STAGING_DIR / "matching_results.tsv"
    staged_candidate = STAGING_DIR / "candidate_pairs.tsv"
    shutil.copy2(matching_path, staged_matching)
    shutil.copy2(candidate_path, staged_candidate)

    with open(MODEL_DIR / "training_summary.json", encoding="utf-8") as fh:
        training_summary = json.load(fh)
    with open(MODEL_DIR / "selected_threshold.json", encoding="utf-8") as fh:
        threshold = json.load(fh)["threshold"]
    with open(TEST_DIR / "test_pipeline_summary.json", encoding="utf-8") as fh:
        test_summary = json.load(fh)
    with open(PROJECT_ROOT / "experiments" / "phase3" / "pruning_labeling_metadata.json", encoding="utf-8") as fh:
        pruning_meta = json.load(fh)
    with open(PROJECT_ROOT / "experiments" / "blocking" / "PHASE_3" / "pruning_check_results.json", encoding="utf-8") as fh:
        oracle_meta = json.load(fh)

    manifest = {
        "submission_local_id": "SUB-001",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_architecture": {
            "base": "B004 four-route union (DEC-027): exact(name+address), forward "
                    "word-TF-IDF k=50, reverse word-TF-IDF r=5, numeric-exact-signature "
                    "(max_group_size=650)",
            "upstream_pruning": pruning_meta,
            "development_oracle_after_pruning": oracle_meta["configs"]["fwd10_rev3"],
            "development_oracle_unpruned_B004": oracle_meta["configs"]["no_pruning"],
        },
        "model": {
            "family": "LightGBM binary GBDT",
            "feature_count": len(training_summary["feature_importances"]),
            "matcher_train_rows": training_summary["n_matcher_train_rows"],
            "matcher_train_positives": training_summary["n_matcher_train_positive"],
            "best_iteration": training_summary["best_iteration"],
        },
        "calibration": {
            "selected_threshold": threshold,
            "calibration_holdout_macro_f05_CORRECTED": training_summary.get(
                "calibration_holdout_macro_f05_at_selected_CORRECTED"),
            "dev_eval_macro_f05_at_selected_CORRECTED": training_summary.get(
                "dev_eval_at_selected_threshold_CORRECTED", {}).get("macro_f05"),
        },
        "test_processing": test_summary,
        "files": {
            "matching_results.tsv": {"sha256": sha256_file(staged_matching), "rows": test_summary["n_test_s1"]},
            "candidate_pairs.tsv": {"sha256": sha256_file(staged_candidate)},
        },
        "official_validator": {
            "command": "python3 utils/validate_submission.py --matching output/matching_results.tsv "
                       "--candidate output/candidate_pairs.tsv --test-dir dataset/test",
            "exit_code": result.returncode,
            "status": "PASS" if validator_pass else "FAIL",
        },
        "compliance": {
            "no_external_lookup": True,
            "no_geocoding": True,
            "no_test_adaptation": True,
            "test_derived_statistics": "none -- target_name_rarity uses a frozen name-collision "
                                        "count built from TRAIN Source-2/3 only (train_name_rarity_counts()), "
                                        "looked up by normalized name string, not computed from test's own "
                                        "population; no test-derived IDF/thresholds anywhere",
            "no_hidden_labels": True,
            "no_commit_or_push": True,
        },
    }
    with open(STAGING_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(STAGING_DIR / "validation.txt", "w", encoding="utf-8") as fh:
        fh.write(result.stdout + "\n" + result.stderr)

    print(json.dumps(manifest, indent=2))
    print(f"\nStaged at: {STAGING_DIR}")
    print(f"matching_results.tsv sha256: {manifest['files']['matching_results.tsv']['sha256']}")
    print(f"candidate_pairs.tsv sha256: {manifest['files']['candidate_pairs.tsv']['sha256']}")
    print(f"VALIDATOR: {'PASS' if validator_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
