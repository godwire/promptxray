"""Score a set of predictions. Pure Python, no numpy."""

from __future__ import annotations

from dataclasses import dataclass, field

UNPARSED = "<unparsed>"


@dataclass
class ClassScore:
    label: str
    support: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class Score:
    labels: list[str]
    per_class: dict[str, ClassScore]
    confusion: dict[str, dict[str, int]]
    total: int
    correct: int
    unparsed: int = 0
    wrong_ids: set[int] = field(default_factory=set)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def macro_f1(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(c.f1 for c in self.per_class.values()) / len(self.per_class)


def normalise(raw: str, labels: list[str]) -> str:
    """Map whatever the model said onto one of the known labels.

    Models add punctuation, quotes, or a whole sentence. An answer that still
    cannot be matched is counted as <unparsed> rather than silently dropped -
    an unusable output is a real failure of the prompt.
    """
    cleaned = raw.strip().strip(".,:;!\"'`").lower()
    for label in labels:
        if cleaned == label.lower():
            return label
    for label in labels:
        if label.lower() in cleaned:
            return label
    return UNPARSED


def score(gold: dict[int, str], predicted: dict[int, str], labels: list[str]) -> Score:
    """Compare gold labels to predictions, keyed by example id."""
    per_class = {label: ClassScore(label=label) for label in labels}
    confusion = {g: {p: 0 for p in labels + [UNPARSED]} for g in labels}

    correct = 0
    unparsed = 0
    wrong_ids: set[int] = set()

    for example_id, truth in gold.items():
        guess = predicted.get(example_id, UNPARSED)
        confusion.setdefault(truth, {p: 0 for p in labels + [UNPARSED]})
        confusion[truth][guess] = confusion[truth].get(guess, 0) + 1
        per_class[truth].support += 1

        if guess == UNPARSED:
            unparsed += 1

        if guess == truth:
            correct += 1
            per_class[truth].true_positives += 1
        else:
            wrong_ids.add(example_id)
            per_class[truth].false_negatives += 1
            if guess in per_class:
                per_class[guess].false_positives += 1

    return Score(
        labels=labels,
        per_class=per_class,
        confusion=confusion,
        total=len(gold),
        correct=correct,
        unparsed=unparsed,
        wrong_ids=wrong_ids,
    )


def macro_f1_from_pairs(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Macro F1 straight from (gold, predicted) pairs.

    Bootstrap resampling draws the same example more than once, so it cannot use
    the id-keyed score() above. This works on a plain list instead.
    """
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for truth, guess in pairs:
        if truth == guess:
            tp[truth] = tp.get(truth, 0) + 1
        else:
            fn[truth] = fn.get(truth, 0) + 1
            if guess in fp:
                fp[guess] += 1

    total = 0.0
    for label in labels:
        precision_denominator = tp[label] + fp[label]
        recall_denominator = tp[label] + fn[label]
        precision = tp[label] / precision_denominator if precision_denominator else 0.0
        recall = tp[label] / recall_denominator if recall_denominator else 0.0
        total += 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return total / len(labels) if labels else 0.0
