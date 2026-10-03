# Implementation and provenance

The release is restricted to MSE implementations for SD1.4 and SDXL.
`reference/sources.json` records original relative filenames, original source
SHA-256 hashes, and the functions extracted. Reference function bodies are kept
unchanged; only the standalone import preamble is supplied for comparison.
Third-party editing CSVs and their local provenance inventory are excluded from the public release. External data sources are listed in `data/README.md`.

| Original source | Released behavior |
| --- | --- |
| `changeemb/scripts/pipeline.py::change_emd` | SD1.4 Adam/MSE, source-token gradient mask, default threshold factor 0.35 |
| `changeemb-pipeline-xl/apply_emb_changing.py::change_emd` | SDXL Adam/MSE, default threshold factor 0.30, applied to both encoders |
| `changeemb/scripts/pipeline_multiple.py::multi_change_emd` | SD1.4 joint MSE editing, fixed targets, default factor 0.20 |
| `changeemb-pipeline-xl/apply_emb_changing_gender_mse.py::change_emd` | Preserved as a reference-only gender-specific experiment |

The loss is the mean squared error over the last hidden-state tensor returned
by the text encoder. Only token embedding rows corresponding to `old` receive
nonzero gradients. For the single-pair variants, target features are recomputed
without gradients each iteration; this matters when `new` contains tokens also
present in `old`. A fresh Adam optimizer is created per single edit.

The original stopping check uses the loss calculated **before** the current
optimizer step. This ordering is retained. Single-pair MSE variants begin
checking after the second update; the joint variant can stop after the first.
The iteration cap is 100 by default. No Fréchet stopping implementation is
included.

## Packaging changes

- Remove machine-specific paths, unused imports, commented baseline loaders and
  import-time model loading. Read paths and parameters from the CLI.
- Freeze non-embedding parameters during editing; restore the caller's gradient
  flags afterward. This avoids accumulating gradients in unchanged parameters.
- Use the encoder's actual device and position limit. Reject empty or overlong
  edit texts. Retain float32 during Adam updates.
- Use `tokenizer_2` with SDXL's second encoder. Original SDXL code passed
  `tokenizer` to both encoders; this is recorded as a corrected interface issue.
- Forward the generator and guidance scale to both SDXL generation stages. The
  old helper accepted these arguments but did not use them.
- Restore per-row embeddings in independent mode. Sequential mode edits all
  selected rows first, then generates with the final state. Original SD1.4 code
  also contained interleaved generation; use an explicit experiment driver if
  that exact historical output sequence is required.
- Preserve the SD1.4 pipeline's default safety checker instead of the source
  script's bypass lambda. Thus filtered-image behavior can differ from the old
  script; numerical/image equivalence is not claimed.
- Save text encoders and metadata instead of duplicating the full UNet/VAE.
  Generate via `--load-edit` to reload those encoders into the same base model.
- Keep prompt text in JSON metadata and use numbered image filenames, avoiding
  accidental path creation from prompt text.

## Gender-specific SDXL reference

The gender-specific file is not interchangeable with the main hidden-state MSE
method: it uses pooled outputs, two gender-conditioned targets and an adaptive
weighted loss. Its stopping condition is `bias_con >= 0.*stopping_con` after
`i > 1`, so finite nonnegative values normally end the run after three updates.
Its wrapper edited only the first encoder. It also assumes `new` starts with
`female ` or `male `. These behaviors are preserved in the source excerpt for
author review; this variant is not advertised as a supported CLI mode.

Adaptive gender-search scripts, old LDM samplers, SD2.1, FLUX, ReFACT, UCE,
Fréchet variants and paper-specific evaluation scripts are outside this focused
release. The original project remains the source of those experiments.

## Publication and release scope

Feng He, Chao Zhang and Zhixue Zhao. EMNLP 2025. [Minimal, Local, and Robust: Embedding-Only Edits for Implicit Bias in T2I Models](https://aclanthology.org/2025.emnlp-main.777/).

This release implements the SD1.4/SDXL MSE workflow only. See [release scope](RELEASE_SCOPE.md) for omitted experiments and outstanding validation. The recorded interface and workflow changes above remain relevant when comparing with historical experiments.
