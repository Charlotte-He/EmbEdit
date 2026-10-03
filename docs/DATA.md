# Input data format

The public release supplies `examples/edits.csv`, an original synthetic example.
Research datasets must be acquired separately; see [data sources](../data/README.md).
Local experiment CSVs and their provenance inventory are ignored by Git.

Files are read as UTF-8, with an optional BOM. Pass a CSV path through `--csv`.

| Column | Required | Meaning |
| --- | --- | --- |
| `old` | Yes | Source concept whose token embedding rows are optimized |
| `new` | Yes | Target concept used to construct the MSE target |
| `prompt` | No | Additional generation prompt |
| `positive1`, `positive2`, … | No | Related generation prompts |
| `negative1`, `negative2`, … | No | Unrelated generation prompts |
| `ex1`, `ex2`, … | No | Generation examples, including gender-task tables |
| `validation` | No | Additional generation prompt, not an automatic model-selection split |
| `gt1`, … / `gn1`, … | No | Accepted but not used by this editing/generation CLI |

The `old` text is always a default generation prompt. Repeated `--prompt`
arguments replace the CSV-derived generation prompts. Quote fields containing
commas. Blank `old`/`new` cells and invalid index ranges are rejected before
model loading. Very long edit texts are rejected instead of silently editing
tokens removed by truncation.

The MIT license covers project code and the original example, not external datasets.
The CSV interface alone does not establish paper-specific splits or evaluation.
