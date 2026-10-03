# Release validation

Validation date: 2026-10-03.

## Completed

- **14 unittest checks passed** on Python 3.11.0 / PyTorch 2.2.2, using a small
  synthetic CPU encoder. No pretrained models or external datasets were used.
- Consolidated single-edit updates matched the extracted original SD1.4 and
  SDXL MSE functions within absolute tolerance `1e-7` over a five-step synthetic
  optimization, including overlap between source and target tokens.
- Joint SD1.4 updates matched the original joint MSE function under the same
  small-fixture comparison.
- Verified unchanged non-source embedding rows and non-embedding parameters,
  restored gradient flags, stopping behavior, and invalid input rejection.
- Verified SDXL's second tokenizer is selected, and the seed and guidance scale
  are forwarded to both base and refiner calls, using call stubs.
- Exercised sequential, joint and independent save/reload workflows with small
  encoder fixtures. Confirmed independent rows do not retain previous edits and
  generation receives the reloaded weights. This fixture does not validate
  Diffusers' actual serialization implementation.
- Python AST parsing and the Slurm shell syntax check passed.
- The initial code release was scanned for weights, generated images and original
  machine-specific absolute paths. Source provenance hashes are included.

## Historical local data check (tables not publicly distributed)

- Added 13 original CSVs, with 380 total rows including overlapping variants.
- Every CSV is byte-identical to the original copied source. SHA-256 hashes,
  sizes, row counts and columns are recorded in `data/SOURCES.json`.
- All rows in all 13 tables pass the released `read_edits` parser; prompt fields
  also pass the existing `prompts_for` helper. No missing `old`/`new` values or
  extra CSV fields were found.
- README quick-start commands reference included CSV paths. The small SD1.4
  example uses its original one-row table; SDXL selects the first EE row.
- No optimizer or generation code changed in the data addition. The 14 CPU
  checks above describe the code validation already performed; the data checks
  do not load a tokenizer or a pretrained model.

## Not performed

- Installing and testing the proposed Diffusers/Transformers dependency ranges
  in a clean environment.
- Loading pretrained SD1.4/SDXL models or generating real images on a GPU.
- Benchmarking memory, running the Slurm template, or reproducing paper metrics.
- Validating the experimental gender-specific SDXL reference as a supported
  inference workflow.

The successful CPU tests establish behavior on the supplied fixtures; they do
not establish full-model equivalence or paper-result reproducibility. Confirm
the final paper hyperparameters and citation separately before a paper release.

Run the included tests with:

```bash
python -m unittest discover -s tests -v
```

## Public-release documentation update

Paper metadata and BibTeX were checked against ACL Anthology. A synthetic example now drives the quick starts. Third-party experiment tables remain local and are excluded from Git and the release archive. Checksums were regenerated for public files only. This update does not add GPU validation or reproduce paper results.
