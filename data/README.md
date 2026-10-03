# External research data

The public release includes only `examples/edits.csv`, an original format example.
Third-party experiment CSVs are not redistributed. Any existing local tables and
`SOURCES.json` are preserved locally and ignored by Git.

## Sources cited by the paper

- **TIMED / TIME**: Hadas Orgad, Bahjat Kawar and Yonatan Belinkov (2023), *Editing Implicit Assumptions in Text-to-Image Diffusion Models*. [Official repository](https://github.com/bahjat-kawar/time-diffusion) and [project page](https://time-diffusion.github.io/).
- **RoAD / ReFACT**: Dana Arad, Hadas Orgad and Yonatan Belinkov (2024), *ReFACT: Updating Text-to-Image Models by Editing the Text Encoder*. [Official project page](https://technion-cs-nlp.github.io/ReFACT/) and [paper](https://arxiv.org/abs/2306.00738).
- **COCO**, for separate image-quality evaluation: [official website](https://cocodataset.org/).

Acquire files through the upstream projects and consult their data terms and
citation instructions. These links identify the paper's data sources; they do
not establish the exact version or provenance of every locally named CSV.

## Using your data

Prepare a UTF-8 CSV with `old` and `new` columns, then pass its path through
`--csv`. See [the schema](../docs/DATA.md). This CLI edits and generates images;
it does not compute efficacy, generality, specificity, FID or CLIP Score.

## Paper splits and preprocessing

Sections 5.1 and Appendix B.1 of the [EmbEdit paper](https://aclanthology.org/2025.emnlp-main.777.pdf)
describe data and modifications to TIMED. The local TIMED variants differ, and
local RoAD tables do not contain the complete 91-entry collection described in
the paper. A verified mapping from original versions to final splits, row order,
filtered objects and modified prompts is not included in this focused release.
Do not treat the example or an arbitrary upstream split as the paper benchmark.
