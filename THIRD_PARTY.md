# Third-party components

The MIT license in this repository applies to the released EmbEdit code. It
does not relicense dependencies, pretrained weights, or datasets.

| Component | Upstream / terms |
| --- | --- |
| PyTorch | https://github.com/pytorch/pytorch — BSD-style license |
| Hugging Face Diffusers | https://github.com/huggingface/diffusers — Apache-2.0 |
| Hugging Face Transformers | https://github.com/huggingface/transformers — Apache-2.0 |
| Accelerate / Safetensors | Hugging Face projects; see their distributed license notices |
| NumPy / Pillow | See the licenses accompanying the installed packages |
| Stable Diffusion v1.4 | https://huggingface.co/CompVis/stable-diffusion-v1-4 — CreativeML Open RAIL-M model terms |
| SDXL base 1.0 | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 — CreativeML Open RAIL++-M model terms |
| SDXL refiner 1.0 | https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0 — see the model card and license |

Model weights are downloaded by the user from the upstream model repositories,
or supplied locally. No CompVis LDM codebase, Stability generative-models
codebase or ReFACT/UCE implementation is bundled.

## External datasets

TIMED (TIME; Orgad et al., 2023), RoAD (ReFACT; Arad et al., 2024) and COCO are external datasets. Acquisition links and attribution are in [data/README.md](data/README.md). Research CSVs and the local provenance inventory are excluded from Git and the public release. The MIT code license does not grant rights to upstream datasets or pretrained models.

`examples/edits.csv` is an original format illustration, not a redistributed benchmark. Original reference function excerpts retain their recorded project provenance; preserve any additional authorship notices identified by the authors.
