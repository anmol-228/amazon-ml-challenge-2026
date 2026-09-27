# Challenge Rules Checklist — Operational Checks

Convenience checklist for before/during submission work. Not a replacement for
`docs\OFFICIAL_REQUIREMENTS.md` or the source PDFs.

## Format
- [ ] Every output file is genuinely TAB-separated, not comma-separated (`sep="\t"` on
      write and read).
- [ ] `matching_results.tsv` has exactly the header `source1_entity_id\tmatched_entity_ids`.
- [ ] `candidate_pairs.tsv` has exactly the header `source1_entity_id\tcandidate_entity_ids`.
- [ ] ID lists are comma-separated with no quoting.

## Coverage and IDs
- [ ] Every test Source-1 entity appears exactly once in `matching_results.tsv`.
- [ ] No duplicate `source1_entity_id` rows.
- [ ] Matched/candidate IDs use only `S2-`/`S3-` prefixes — no `S1-` self-matches.
- [ ] Matched/candidate IDs actually exist in the test set (spot-check or use
      `--check-ids` on the validator).
- [ ] No duplicate IDs within a single row's ID list.

## Singletons
- [ ] Singleton entities (no true match) are predicted with an **empty**
      `matched_entity_ids`, not omitted from the file.

## Candidate/match subset invariant
- [ ] Every ID appearing in `matching_results.tsv` also appears in the corresponding row
      of `candidate_pairs.tsv`.
- [ ] `candidate_pairs.tsv` reflects the *final* candidate set immediately before
      inference — not an intermediate/raw blocking pass that gets filtered later.

## Validator
- [ ] Run `python3 utils/validate_submission.py --matching output/matching_results.tsv
      --candidate output/candidate_pairs.tsv --test-dir dataset/test` and confirm `PASS`
      before every submission.
- [ ] Remember: the validator checks structure/IDs/rules — it does **not** compute your
      F_0.5 score. Score your own held-out validation split separately.

## Model
- [ ] Final model is MIT or Apache-2.0 licensed.
- [ ] Final model is ≤ 8 billion parameters.

## Fair play
- [ ] No external business/entity identity lookup, commercial ER API, government registry
      lookup, or geocoding/enrichment of supplied records was used anywhere in the
      pipeline.

## Submission tracking
- [ ] Preserve an immutable snapshot of every real submission under `submissions\`.
- [ ] Log every real submission in `SUBMISSION_LOG.md` immediately after making it.
- [ ] Track the daily submission count — **maximum 5 per day**, 3 challenge days.

## Portal / access
- [ ] Attempt the challenge from a single desktop/laptop only — no mobile device.
- [ ] Do not use simultaneous logins from multiple machines/sessions.
- [ ] Leaderboard submission is a human-approved action; this local workspace's tooling
      does not access or automate the Unstop portal.
