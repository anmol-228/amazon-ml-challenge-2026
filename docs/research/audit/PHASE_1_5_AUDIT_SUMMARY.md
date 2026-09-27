# Phase 1.5 Audit Summary

A bounded, read-only independent technical audit was run against the four Phase 1.5 research
artifacts (`RESEARCH_DOSSIER.md`, `TECHNIQUE_EVIDENCE_MATRIX.md`, `LICENSE_AND_TOOLING_MATRIX.md`,
`RESEARCH_TO_EXPERIMENT_PLAN.md`) against canonical project truth (`docs/DATASET_AUDIT.md`,
`PROJECT_STATE.md`, `DECISIONS.md`, `docs/OFFICIAL_REQUIREMENTS.md`), before Phase 2. The
full finding-by-finding mapping of what was found and how each was repaired is preserved in
`RESEARCH_DOSSIER.md` §20 (see that section for complete detail); this file is a compact,
tool-neutral summary of the same review.

## Issues Identified, Severity, and Repair

| Severity | Issue identified | Repair performed |
|---|---|---|
| CRITICAL | Calibration/threshold leakage was unspecified — the plan risked fitting calibration and threshold selection on in-sample predictions. | A matcher-training/calibration-holdout split (or equivalent k-fold cross-fitting) was added as a binding cross-cutting policy in `RESEARCH_TO_EXPERIMENT_PLAN.md`. |
| CRITICAL | `candidate_pairs.tsv`'s required final-set provenance was not operationalized against the official requirement. | An explicit provenance-contract policy was added, naming the exact boundary between upstream candidate-pruning and the decisive matching stage. |
| MATERIAL | Entity-level threshold policy was under-specified relative to a "per-entity" framing elsewhere in the research. | The calibration experiment now compares a global threshold against at least one entity-adaptive policy explicitly. |
| MATERIAL | The frozen validation partition was scheduled for repeated, adaptive-looking reuse across multiple experiments. | Reserved for exactly one confirmatory touch, at a single, later, mature-pipeline comparison point. |
| MATERIAL | The hard-negative-mining rationale overstated what referential-integrity checking proves about label completeness. | Corrected wording plus a more conservative negative-mining procedure that avoids the population most exposed if the completeness assumption is wrong. |
| MATERIAL | The blocking routes' complementarity was stated as an established fact rather than a hypothesis to be measured. | Reframed explicitly as the union experiment's own hypothesis, to be confirmed or refuted by that experiment, not assumed beforehand. |
| MATERIAL | An address-retrieval ablation was framed as optional rather than a required protection for a specific at-risk population. | Made a required ablation, not optional. |
| MATERIAL | Key blocking-diagnostic metrics were named informally without exact definitions. | Formal metric definitions were added, including singleton-specific handling. |
| MATERIAL | A France-transfer mitigation was proposed but the one proxy test actually capable of measuring transfer cost was never scheduled. | The leave-one-country-out proxy experiment was added to the schedule. |
| MATERIAL | Country-feature handling for the unseen test-only category was ambiguous between two different encodings. | Resolved in favor of an agreement-style feature by default, with a raw-identity encoding requiring an explicit ablation if tried at all. |
| MATERIAL | A structural rejection of graph-based entity resolution was broader than the evidence supporting it. | A narrow, non-graph, deterministic conflict-resolution check was carved out and scheduled separately, without reversing the broader rejection. |
| MATERIAL | Scale-feasibility claims were stated more confidently than available precedent supported. | Softened to explicit, per-experiment peak-RAM/disk/candidate-row scale-budget reporting requirements. |
| MATERIAL | The concrete feature-engineering inventory did not clearly guarantee the feature families the research had prioritized. | Made explicit as a named feature inventory. |
| MATERIAL | Singleton-specific risk diagnostics were folded into post-matching accuracy rather than treated as a first-class blocking-stage metric. | Added as a named part of the candidate-load metric definition. |
| MINOR | One early experiment's frozen-validation touch was low-information for its cost. | Removed; that experiment is now development-only. |
| MINOR | Some "robust to an unseen language/country" wording was stronger than what was actually established. | Tightened to "language-agnostic mechanism, transfer unverified" throughout. |
| MINOR | A compliance-status label risked being read as full submission compliance rather than component-level license compatibility. | A scope clarification was added. |
| NEEDS VERIFICATION | Pretrained-model permissibility under the fair-play rule. | Left open, correctly — not literature-resolvable; flagged as review-required and not depended on by any priority-0 technique. |
| NEEDS VERIFICATION | Exact model/checkpoint compliance at adoption time. | Left open — noted as a future adoption-time step, not resolvable in advance. |
| NEEDS VERIFICATION | Numeric-token blocking's real-world selectivity (collision risk). | Left open — identified as exactly what a later collision-group-analysis experiment exists to measure. |
| NEEDS VERIFICATION | Whether the frozen validation split is a reusable selection benchmark or a one-shot holdout. | Resolved by the "touched exactly once" policy above — no longer an open item. |

## Final Status

All CRITICAL and MATERIAL findings were repaired in the research documents before Phase 1.5
was closed out. All MINOR findings were addressed. All NEEDS VERIFICATION items remain
correctly open pending information this project cannot resolve on its own (organizer
clarification, or a later experiment's own measurement) — none of them block any priority-0
technique. No implementation, experiment, or data access occurred during this review or its
repair. Phase 2 remained not started throughout.
