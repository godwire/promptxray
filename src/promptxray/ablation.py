"""Leave-one-out ablation: measure what each block of the prompt actually does.

The idea is borrowed from occlusion methods in explainable ML. There you hide
one input feature and watch the prediction move. Here the "features" are the
blocks of your prompt: remove one, re-run, and see how the score moves.

To keep this affordable, ablated runs use an error-enriched subset instead of
the whole dataset, and every delta is reported against the baseline measured
on that same subset - never against the full-dataset number, which would not
be a fair comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import blocks as blocks_mod
from .blocks import Block
from .cache import Cache
from .dataset import Example, error_enriched_subset
from .metrics import Score, macro_f1_from_pairs, score
from .providers import Provider
from .runner import RunResult, run

# A delta smaller than this is treated as zero even if the arithmetic says
# otherwise, and a confidence interval wider than this is not worth a verdict.
NOISE_FLOOR = 0.005
TIGHT_ENOUGH = 0.02


@dataclass
class BlockEffect:
    """What happened when one block was removed."""

    block: Block
    delta_macro_f1: float = 0.0
    delta_accuracy: float = 0.0
    fixed: list[int] = field(default_factory=list)   # wrong with the block, right without it
    broken: list[int] = field(default_factory=list)  # right with the block, wrong without it
    delta_per_class: dict[str, float] = field(default_factory=dict)
    ci_low: float | None = None
    ci_high: float | None = None
    skipped_reason: str = ""

    @property
    def verdict(self) -> str:
        """Plain-language reading of the measurement.

        The confidence interval decides, not the point estimate. A block only
        gets a verdict when resampling the examples keeps giving the same sign;
        otherwise the honest answer is that the run cannot tell.
        """
        if self.skipped_reason:
            return "not tested"
        if self.ci_low is None:
            if not self.fixed and not self.broken:
                return "no effect"
            return "carries its weight" if self.delta_macro_f1 < 0 else "hurts the score"

        if self.ci_high < -NOISE_FLOOR:
            return "carries its weight"
        if self.ci_low > NOISE_FLOOR:
            return "hurts the score"
        if abs(self.ci_low) <= TIGHT_ENOUGH and abs(self.ci_high) <= TIGHT_ENOUGH:
            return "no effect"
        return "too noisy to call"

    @property
    def confident(self) -> bool:
        """True when the interval stays on one side of zero."""
        return self.verdict in {"carries its weight", "hurts the score"}

    @property
    def ci_text(self) -> str:
        if self.ci_low is None:
            return ""
        return f"90% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]"

    @property
    def contribution(self) -> float:
        """How much the block helps. Positive = removing it made things worse."""
        return -self.delta_macro_f1


@dataclass
class AblationReport:
    baseline_score: Score
    baseline_subset_score: Score
    effects: list[BlockEffect]
    subset_size: int
    api_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int


def bootstrap_ci(
    pairs_baseline: list[tuple[str, str]],
    pairs_ablated: list[tuple[str, str]],
    labels: list[str],
    resamples: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """A 90% confidence interval for the delta, by resampling the examples.

    The point estimate answers "what happened on these examples". This answers
    the question that actually matters: would the sign survive a different draw
    of examples from the same pile?
    """
    size = len(pairs_baseline)
    if size < 5 or resamples <= 0:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        base = macro_f1_from_pairs([pairs_baseline[i] for i in picks], labels)
        abl = macro_f1_from_pairs([pairs_ablated[i] for i in picks], labels)
        deltas.append(abl - base)

    deltas.sort()
    low = deltas[int(0.05 * (len(deltas) - 1))]
    high = deltas[int(0.95 * (len(deltas) - 1))]
    return (low, high)


def measure_removal(
    block: Block,
    with_block: RunResult,
    without_block: RunResult,
    examples: list[Example],
    labels: list[str],
    resamples: int = 400,
    seed: int = 0,
) -> BlockEffect:
    """Compare two runs over the same examples: prompt with the block, and without it.

    Shared by one-shot ablation and by the greedy minimiser, so both judge a
    block by exactly the same rules.
    """
    gold = {e.id: e.label for e in examples}
    before = score(gold, {e.id: with_block.predictions.get(e.id, "") for e in examples}, labels)
    after = score(gold, {e.id: without_block.predictions.get(e.id, "") for e in examples}, labels)

    effect = BlockEffect(block=block)
    effect.delta_macro_f1 = after.macro_f1 - before.macro_f1
    effect.delta_accuracy = after.accuracy - before.accuracy
    effect.delta_per_class = {
        label: after.per_class[label].f1 - before.per_class[label].f1 for label in labels
    }

    pairs_before = []
    pairs_after = []
    for example in examples:
        guess_before = with_block.predictions.get(example.id, "")
        guess_after = without_block.predictions.get(example.id, "")
        pairs_before.append((example.label, guess_before))
        pairs_after.append((example.label, guess_after))

        was_right = guess_before == example.label
        now_right = guess_after == example.label
        if not was_right and now_right:
            effect.fixed.append(example.id)
        elif was_right and not now_right:
            effect.broken.append(example.id)

    low, high = bootstrap_ci(pairs_before, pairs_after, labels, resamples, seed)
    if low == low:  # not NaN
        effect.ci_low, effect.ci_high = low, high
    return effect


def ablate(
    blocks: list[Block],
    examples: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    baseline: RunResult,
    baseline_score: Score,
    subset_size: int = 60,
    workers: int = 4,
    seed: int = 0,
    resamples: int = 400,
) -> AblationReport:
    """Remove each unpinned block in turn and measure the damage."""
    subset = error_enriched_subset(examples, baseline_score.wrong_ids, subset_size, seed)
    subset_ids = [e.id for e in subset]
    gold_subset = {e.id: e.label for e in subset}

    baseline_subset = score(
        gold_subset,
        {i: baseline.predictions[i] for i in subset_ids},
        labels,
    )

    effects: list[BlockEffect] = []
    calls = hits = in_tokens = out_tokens = 0

    for block in blocks:
        effect = BlockEffect(block=block)
        if block.pinned:
            effect.skipped_reason = block.pin_reason or "pinned"
            effects.append(effect)
            continue

        variant = blocks_mod.render(blocks, skip=block.index)
        result = run(
            variant, subset, labels, provider, cache,
            workers=workers,
            progress_label=f"ablating block #{block.index + 1}",
        )
        calls += result.api_calls
        hits += result.cache_hits
        in_tokens += result.input_tokens
        out_tokens += result.output_tokens

        effects.append(measure_removal(block, baseline, result, subset, labels, resamples, seed))

    return AblationReport(
        baseline_score=baseline_score,
        baseline_subset_score=baseline_subset,
        effects=effects,
        subset_size=len(subset),
        api_calls=calls,
        cache_hits=hits,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
    )