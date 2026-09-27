# Official Requirements — Amazon ML Challenge 2026

Sourced strictly from three places, kept separate by source. Where the two official PDFs
disagree, both are recorded verbatim rather than reconciled — see
"Official-document differences / items to preserve" at the end.

**Sources:**
- **[GUIDE]** = Guidelines and Key Instructions PDF
  (`_source\6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`)
- **[PROB]** = Problem Statement PDF
  (`_source\6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`)
- **[PKG]** = Files inside `student_resource\` (README.md, Documentation_template.md,
  validate_submission.py)

---

## CHALLENGE WINDOW — [GUIDE]
25 September 2026, 12:00 AM IST → 27 September 2026, 11:59 PM IST. Dataset access on day
1; build/submit through day 3.

## SUBMISSION LIMIT — [GUIDE]
Maximum 5 submissions per day, over 3 days; the submit button is disabled after the 5th
that day. Teams must maintain version history of all submissions — shortlisting is based
on submitted solutions, and the final source code may be requested later.

## PROBLEM — [PROB] / [PKG]
Business Entity Resolution across 3 independent, noisy data sources with no shared
identifiers.

## DATA — [PROB] / [PKG]
- Source 1 = deduplicated reference source.
- Source 2 and Source 3 = match targets.
- Each Source-1 entity may match zero, one, or many Source-2/3 records.
- Columns: `entity_id`, `business_name`, `business_address`, `country`.
- Ground truth (`train_ground_truth.tsv`): `source1_entity_id`, `matched_entity_ids`.

## COUNTRIES — [PROB] / [PKG]
Train = `US`, `India`. Test additionally includes `France` (not present in training).
`country` must be treated as an **open set** — do not hard-code, filter, or one-hot to
`{US, India}` only; every test entity, France included, must appear in the submission.

## OUTPUT — [PROB] / [PKG]
Two TSV files in `output/`:
- `matching_results.tsv` — final matches; the **only** file scored on the live leaderboard.
- `candidate_pairs.tsv` — the exact, final candidate set fed into the matching model at
  inference time (not an earlier raw blocking pass later filtered further). Not scored,
  but used to audit blocking quality and pipeline integrity. Every matched ID must appear
  in the candidate set.

## SCORING — [PROB] / [PKG]
Macro-averaged F_0.5 (β = 0.5): `F_0.5 = (1.25 × P × R) / (0.25 × P + R)`, computed per
Source-1 entity then averaged over all Source-1 entities. Singletons included — an entity
with no true matches scores 1.0 for an empty prediction, 0.0 for any false match.
Precision-heavy: false merges cost more than missed matches.

## FINAL PACKAGE — [PROB] / [PKG]
Single zip `<team_name>_submission.zip`:
```
output/
  matching_results.tsv
  candidate_pairs.tsv
code/
  business_entity_resolution/
    src/
    README.md            (exact reproduction instructions, data → blocking → matching → output)
    requirements.txt     (pinned dependencies/environment)
Documentation_template.md  (filled-in methodology write-up; .md or a .pdf export of it)
```
Used by Amazon to reproduce results, audit blocking, and check fair-play/model-license
compliance; top teams' packages are reviewed before final rankings are confirmed.

## MODEL RESTRICTION — [PROB]
Final model must be MIT or Apache 2.0 licensed, and ≤ 8 billion parameters.

## FAIR PLAY — [PROB] / [GUIDE]
No external business/entity identity lookup or enrichment: no commercial entity-resolution
APIs, no government business-registration lookups, no geocoding APIs, no internet
enrichment of the supplied entities. All approaches/methodologies/code will be reviewed;
evidence of external lookup causes immediate disqualification. General technical research
(algorithms, libraries, papers, ML methods) is a separate matter and not restricted.
[GUIDE] additionally prohibits any cheating/plagiarism/multi-ID registration.

## GENERAL GUIDELINE ARTEFACTS — [GUIDE]
For the best solution submitted by the team:
- A 1–2 page document explaining ML approach, models used, experiments, and conclusion.
- Source code for experiments, training, and inference, with comments describing
  functions.
- Maintained version history of all submissions.

## TOP-100 / METHODOLOGY REQUIREMENTS — [GUIDE] / [PROB]
The methodology document (or, per [GUIDE], the Top-100 documents/details) must record:
- Methodology used.
- Candidate generation / blocking strategy.
- Model architecture and feature engineering.
- Any other relevant information about the approach.
[PROB] specifies the template for this is `Documentation_template.md`, with **no page
limit** — prioritise clarity and technical depth over brevity.

---

## Official-document differences / items to preserve

Do **not** silently resolve these — both are recorded as stated, and neither is deleted in
favor of the other absent later official clarification.

### A. Methodology document length

- **[GUIDE]:** "1–2-page document explaining the ML approach, ML models used, experiments
  and conclusion" — required for the **best solution submitted by the team**.
- **[PROB] / [PKG]:** `Documentation_template.md` "has no page limit... prioritise clarity
  and technical depth over brevity," and must cover methodology, candidate
  generation/blocking, model architecture/feature engineering, and any other relevant
  information.

These may be two distinct deliverables (a short summary vs. the full template) rather than
one document under two descriptions — treat them as potentially separate expectations
until Amazon clarifies.

### B. Leaderboard wording

- **[GUIDE]:** "There will be two leaderboards - Private and Public. Evaluation and
  shortlisting will be based on performance across both leaderboards."
- **[PROB]:** "Public Leaderboard... provide real-time feedback... Private Leaderboard...
  Final Rankings: The final decision will be based on the private leaderboard." Predictions
  for the full test set are submitted in both cases; the split is applied during scoring.

Do not assume which governs Top-100 shortlisting vs. final ranking — both statements are
preserved as given.
