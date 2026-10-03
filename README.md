# EmbEdit — Stable Diffusion 1.4 and SDXL

MSE-based text-embedding editing for **Stable Diffusion v1.4** and **SDXL 1.0**.
This repository contains editing and image-generation code, an original minimal CSV example, installation instructions and source provenance. Third-party experiment datasets, model weights and generated images are not distributed in this public release.



[Paper](https://aclanthology.org/2025.emnlp-main.777/) · [PDF](https://aclanthology.org/2025.emnlp-main.777.pdf) · [Release scope](docs/RELEASE_SCOPE.md)

This release provides the **MSE-based embedding editing and generation workflow for SD1.4 and SDXL**. It includes independent, sequential and SD1.4 joint editing. FLUX, adaptive gender balancing, probing, baseline implementations and paper-specific metric evaluation are outside this release. The quick starts demonstrate the released workflow; they are not commands for reproducing the paper tables.

## Installation

Use Python 3.10 or later (3.11 recommended) and a CUDA-capable GPU for full models.
Install a CUDA-compatible PyTorch build following the
[PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m embedit.cli --help
```

The dependency bounds are a proposed compatible environment, not a recovered
lockfile of the original experiments. Full pipelines use float32, matching the
editing scripts. SDXL with the refiner requires substantially more GPU memory
than SD1.4; choose an appropriate GPU before running. No model is downloaded by
`--help` or the included unit tests.

## Example and external data

Use the original format example at `examples/edits.csv`, or provide your own CSV with `old` and `new` columns. Optional generation fields are documented in [docs/DATA.md](docs/DATA.md).

The paper uses TIMED from TIME and RoAD from ReFACT. Obtain research data from the original authors; see [data acquisition](data/README.md). Local experiment CSVs remain excluded from Git. The example is a format demonstration, not a paper benchmark.

## Stable Diffusion v1.4

```bash
python -m embedit.cli \
  --model sd14 --csv examples/edits.csv --output outputs/sd14_mse \
  --mode sequential --learning-rate 0.001 --iterations 100 --mse-factor 0.35 \
  --seed 0 --num-seeds 1
```

## SDXL

Both text encoders are edited using their respective tokenizers. The optional
refiner shares the edited second encoder and VAE with the base pipeline.

```bash
python -m embedit.cli \
  --model sdxl --csv examples/edits.csv --output outputs/sdxl_mse \
  --mode sequential --learning-rate 0.001 --iterations 100 --mse-factor 0.30 \
  --begin 0 --end 1 --refiner --seed 0 --num-seeds 1
```

Omit `--refiner` for base-only generation. The base/refiner denoising split is
0.8. Default model IDs are `CompVis/stable-diffusion-v1-4`,
`stabilityai/stable-diffusion-xl-base-1.0`, and
`stabilityai/stable-diffusion-xl-refiner-1.0`. Supply `--model-id` and optionally
`--refiner-id` for local Diffusers-format model directories; add
`--local-files-only` to prohibit downloads.

## Editing modes and output

| Mode | Behavior | Models |
| --- | --- | --- |
| `sequential` | Apply CSV rows in order; generate with the final cumulative edit | SD1.4, SDXL |
| `independent` | Restore original embedding weights before each row; save/generate per row | SD1.4, SDXL |
| `joint` | Optimize all selected pairs together with fixed target features | SD1.4 |

`--begin` is inclusive and `--end` is exclusive; by default all rows are used.
`--skip-generation` saves edited encoders without generating images. Single-pair
editing recomputes the detached target features every step. Joint editing keeps
the targets fixed and defaults to `--mse-factor 0.2`, as in its source script.

Sequential/joint runs save `edited/text_encoder/`, and for SDXL also
`edited/text_encoder_2/`. Independent runs save one `edit_XXXX/` directory per
row. Each saved edit has `edit.json` with the model ID and optimization history.
`run.json` records arguments and dependency versions. Images have numeric file
names, with prompts and seeds in `images/prompts.json`.

Reload saved encoders for generation:

```bash
python -m embedit.cli \
  --model sd14 --load-edit outputs/sd14_mse/edited \
  --output outputs/sd14_reload --prompt "a photo of a rose" --seed 0
```

Use `--model sdxl --refiner` when reloading an SDXL edit for base/refiner generation.
Each command requires a new output directory to avoid replacing previous runs.
Saved encoders are model-derived artifacts and retain the base model's licensing
requirements; they are not included in this repository.

## Slurm

Activate the installed Python environment and submit from the repository root:

```bash
sbatch scripts/slurm_edit.sh \
  --model sd14 --csv examples/edits.csv --output outputs/slurm_sd14 \
  --skip-generation
```

Adapt the resource directives for your cluster. Run full models only inside a
GPU allocation, never on an HPC login node.



## License and attribution

Project code is released under the [MIT License](LICENSE). External libraries,
Stable Diffusion weights and third-party datasets retain their own licenses.
See [THIRD_PARTY.md](THIRD_PARTY.md) and [data provenance](data/SOURCES.json).
Third-party data must be obtained separately and retains its own terms. The local experiment-table inventory is excluded from the public release.

## Citation

```bibtex
@inproceedings{he-etal-2025-minimal,
  title = "Minimal, Local, and Robust: Embedding-Only Edits for Implicit Bias in {T}2{I} Models",
  author = "He, Feng and Zhang, Chao and Zhao, Zhixue",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  year = "2025",
  month = nov,
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  pages = "15374--15392",
  doi = "10.18653/v1/2025.emnlp-main.777",
  url = "https://aclanthology.org/2025.emnlp-main.777/"
}
```
