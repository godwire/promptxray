"""Load the labelled examples that the classifier is measured against."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    """One labelled row: the text to classify and the correct answer."""

    id: int
    text: str
    label: str


def load(path: str | Path, text_column: str = "text", label_column: str = "label") -> list[Example]:
    """Read a .csv or .jsonl file into a list of Example objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    missing = {text_column, label_column} - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Dataset {path} is missing column(s): {', '.join(sorted(missing))}. "
            f"Found: {', '.join(rows[0].keys())}"
        )

    examples: list[Example] = []
    for i, row in enumerate(rows):
        text = (row.get(text_column) or "").strip()
        label = (row.get(label_column) or "").strip()
        if text and label:
            examples.append(Example(id=i, text=text, label=label))
    return examples


def labels_of(examples: list[Example]) -> list[str]:
    """All distinct gold labels, sorted, so reports stay stable between runs."""
    return sorted({e.label for e in examples})


def error_enriched_subset(
    examples: list[Example],
    wrong_ids: set[int],
    size: int,
    seed: int = 0,
) -> list[Example]:
    """Pick the rows to re-run during ablation.

    Every example the baseline got wrong is included, because that is where a
    prompt change shows up. The rest of the budget is filled with correct rows,
    so the tool can also see a block that *breaks* something that used to work.
    """
    wrong = [e for e in examples if e.id in wrong_ids]
    right = [e for e in examples if e.id not in wrong_ids]

    if size <= 0 or size >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    chosen = wrong[:size]
    remaining = size - len(chosen)
    if remaining > 0:
        rng.shuffle(right)
        chosen += right[:remaining]
    return sorted(chosen, key=lambda e: e.id)


def stratified_split(
    examples: list[Example], holdout_ratio: float = 0.5, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Split into (train, holdout), keeping the class balance in both halves.

    `suggest` needs this: choosing which blocks to drop and then measuring the
    result on the same examples would flatter the answer every time.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    train: list[Example] = []
    holdout: list[Example] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        cut = int(round(len(group) * (1 - holdout_ratio)))
        cut = min(max(cut, 1), len(group) - 1) if len(group) > 1 else len(group)
        train += group[:cut]
        holdout += group[cut:]

    return sorted(train, key=lambda e: e.id), sorted(holdout, key=lambda e: e.id)
