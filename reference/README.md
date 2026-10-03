# Original MSE source excerpts

These are unchanged function bodies extracted from the original project for
provenance and comparison. They are **not command-line entry points**. Use
`python -m embedit.cli` for editing and generation.

- `sd14_mse.py`: the SD1.4 single-pair MSE optimizer.
- `sd14_joint_mse.py`: the SD1.4 simultaneous multi-pair MSE optimizer.
- `sdxl_mse.py`: the SDXL single-encoder MSE optimizer, called for both encoders
  by the consolidated CLI.
- `sdxl_gender_mse.py`: the experimental weighted gender MSE variant; limitations
  are documented in `docs/IMPLEMENTATION.md`.

Only minimal import/device preambles were added. The CPU device in these
preambles supports the synthetic comparison tests; these files do not load
pretrained models. Some original comments/signatures refer to inactive FID
code, but the extracted active implementations use MSE only. Source filenames
and hashes are recorded in `sources.json` without private absolute paths.
