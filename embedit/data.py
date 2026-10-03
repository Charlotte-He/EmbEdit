"""Read user-supplied edit tables without redistributing datasets."""

import csv


def read_edits(path, begin=0, end=None):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"old", "new"}.issubset(reader.fieldnames or []):
            raise ValueError("The CSV must have old and new columns.")
        rows = list(reader)
    end = len(rows) if end is None else end
    if not 0 <= begin < end <= len(rows):
        raise ValueError(f"Require 0 <= begin < end <= {len(rows)}.")
    selected = []
    for index in range(begin, end):
        row = rows[index]
        if None in row:
            raise ValueError(f"CSV row {index + 2} has extra fields; quote embedded commas.")
        if not all((row.get(key) or "").strip() for key in ("old", "new")):
            raise ValueError(f"CSV row {index + 2} has an empty old/new value.")
        selected.append({key: (value or "").strip() for key, value in row.items()})
    return selected


def prompts_for(row):
    """Keep prompt text in metadata, never use it as a filesystem path."""
    prompts = [("base", row["old"])]
    for key, value in row.items():
        if value and (key in {"prompt", "validation"} or
                      any(key.startswith(prefix) and key[len(prefix):].isdigit()
                          for prefix in ("positive", "negative", "ex"))):
            prompts.append((key, value))
    return prompts
