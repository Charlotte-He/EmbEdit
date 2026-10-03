# Release scope

This repository releases the SD1.4 and SDXL MSE-based embedding optimizer and its editing/generation workflow accompanying the EMNLP 2025 EmbEdit paper. It supports independent and sequential edits, SD1.4 joint edits, encoder save/reload, and an original example CSV.

It does not include FLUX, the probing experiment, adaptive gender/racial balancing, baseline methods or the paper's metric evaluation. The gender-specific source excerpt in `reference/` is archival and is not a supported CLI workflow. An explicit destination edit such as a gender-conditioned phrase is not equivalent to the paper's adaptive balancing objective.

Independent mode corresponds conceptually to resetting embeddings per edit; sequential mode applies cumulative edits. These labels do not establish equivalence to the complete paper protocol. Generation order, tokenizer corrections and other differences are recorded in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Validation limits

Historical CPU fixture checks are documented in [VALIDATION.md](VALIDATION.md). The dependency environment and real pretrained SD1.4/SDXL editing, saving, reloading and image generation still need validation. No paper metrics have been independently reproduced for this release. Quick-start hyperparameters are implementation examples, not verified settings for every reported table.

## Before claiming full reproduction

Record exact model revisions and environment versions; map each experiment to data versions, preprocessing, row order, seeds and commands; provide metric scripts; validate a real-model save/reload workflow; and compare the resulting metrics against the paper. Baseline implementations may remain upstream dependencies with attribution rather than being copied into this repository.
