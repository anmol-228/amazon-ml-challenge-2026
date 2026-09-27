# Problem Summary — Business Entity Resolution (Convenience Notes)

**This is a convenience summary, not a replacement for the authoritative PDFs in
`_source\` or `docs\OFFICIAL_REQUIREMENTS.md`.** When in doubt, check the source.

Given business records from three noisy, independently-collected sources, find every
Source-2 and Source-3 record that refers to the same real-world business as each Source-1
(deduplicated reference) entity. A match can be zero, one, or many records.

**Fields per record:** `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`,
`business_address`, `country`.

**Noise to expect:** name abbreviations/legal-suffix variants/DBA names/punctuation/typos/
word-order changes; address abbreviations/transliteration/missing components/landmark
references/numbering-format differences/component reordering.

**Countries:** train has US + India; test adds France. Treat `country` as open-set — never
hard-code to just US/India.

**Two output files, both TSV, in `output/`:**
- `matching_results.tsv` — scored on the leaderboard.
- `candidate_pairs.tsv` — your exact final candidate set before inference (not an earlier
  raw blocking pass); every match must appear here.

**Metric:** macro-averaged F_0.5 per Source-1 entity (precision weighted 2× recall),
singletons included and worth full credit when correctly left empty.

**Constraints:** MIT/Apache-2.0 model, ≤8B parameters; no external entity/business lookup
or enrichment of the supplied records (general algorithm/library research is fine); output
must validate against `utils\validate_submission.py` before submitting.

**Final package:** a zip with `output/`, `code/business_entity_resolution/{src,README.md,
requirements.txt}`, and the filled-in `Documentation_template.md`.
