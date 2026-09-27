# Final Deliverable Checklist — Amazon ML Challenge 2026

Tracks what the final submission package needs, per both official documents. Nothing
below is marked complete — this is Phase 0; no modeling work has started.

## Leaderboard
- [ ] `matching_results.tsv` uploaded to the Portal (human-approved action only).

## Final package (`<team_name>_submission.zip`)
- [ ] `output/matching_results.tsv`
- [ ] `output/candidate_pairs.tsv`
- [ ] `code/business_entity_resolution/src/` — all source code
- [ ] `code/business_entity_resolution/README.md` — reproduction instructions
      (data → blocking → matching → output)
- [ ] `code/business_entity_resolution/requirements.txt` — pinned dependencies/environment
- [ ] `Documentation_template.md` filled in (methodology write-up)

## General guideline artefacts (best-solution deliverables, per [GUIDE])
- [ ] 1–2 page document explaining ML approach, models, experiments, conclusion
- [ ] Source code for experiments/training/inference, with comments on functions
- [ ] Maintained submission version history (`SUBMISSION_LOG.md` + `submissions\`)

> Note: item above and the `Documentation_template.md` deliverable may be two separate
> documents — see "Official-document differences" in `OFFICIAL_REQUIREMENTS.md`. Do not
> assume one satisfies the other without later clarification.

## Methodology content (must be described somewhere in the deliverables)
- [ ] Methodology used
- [ ] Candidate generation / blocking strategy
- [ ] Model architecture and feature engineering
- [ ] Any other relevant information about the approach

## Compliance
- [ ] Final model is MIT/Apache-2.0 licensed and ≤ 8B parameters
- [ ] No external entity/business lookup or enrichment anywhere in the pipeline
- [ ] `output/candidate_pairs.tsv` is the true final candidate set (matches ⊆ candidates)
- [ ] Passed `utils/validate_submission.py` before final submission
