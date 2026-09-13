# Public edition / 公开版说明

Prepared on 2026-09-13 from the Group029 submission archive.

## Preserved

- Transformation Python files are byte-identical to the submission source.
- Both notebooks retain all original cell source text; only stored outputs and nonessential metadata were removed.
- The source-to-target mapping and EDA report are retained.
- All six original test modules are included; a separate runner selects the 18 tests that do not need private records.

## Excluded from publication

- AI declarations and conversation records, member signatures, student numbers and private contact information.
- Original JSON/XML records, generated record-level CSVs, notebook outputs, private runtime paths and submission archives.
- Teaching slides, assignment specification and marking rubric.

## Verification

- Notebook structure validated; code-cell sources compared with the submission.
- All 18 public tests passed locally.
- The public edition's unchanged pipeline was rerun in an isolated local directory with authorised source data: all six regenerated CSVs matched the submission byte for byte, and all 46 integration/unit tests passed. No private input or regenerated output was copied into this repository.
- Known member identifiers, personal contact details, local home paths and private chat URLs were scanned in the selected text files and PDF text/metadata.
- No PDF embedded attachments were found. The report is an aggregate student-authored report, not a copy of the signed declaration.
- Privacy review is scoped to the published files; it does not establish that raw source records are safe to publish. They remain excluded.

The public tests are runnable without the course data. Full integration and EDA execution need the authorised original inputs described in the README. Re-running the project produces local outputs, which must not be committed.
