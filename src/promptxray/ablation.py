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

from dataclasses import dataclass, field

from . import blocks as blocks_mod
from .blocks import Block
from .cache import Cache
from .dataset import Example, error_enriched_subset
from .metrics import Score, score
from .providers import Provider
from .runner import RunResult, run


@dataclass
class BlockEffect:
    """What happened when one block was removed."""

    block: Block
    delta_macro_f1: float = 0.0
    delta_accuracy: float = 0.0
    fixed: list[int] = field(default_factory=list)   # wrong with the block, right without it
    broken: list[int] = field(default_factory=list)  # right with the block, wrong without it
    delta_per_class: dict[str, float] = field(default_factory=dict)
    skipped_reason: str = ""

    @property
    def verdict(self) -> str:
        """Plain-language reading of the number, for the report."""
        if self.skipped_reason:
            return "not tested"
        if not self.fixed and not self.broken:
            return "no effect"
        if self.delta_macro_f1 < -0.01:
            return "carries its weight"
        if self.delta_macro_f1 > 0.01:
            return "hurts the score"
        return "no clear effect"

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

        ablated = score(gold_subset, result.predictions, labels)
        effect.delta_macro_f1 = ablated.macro_f1 - baseline_subset.macro_f1
        effect.delta_accuracy = ablated.accuracy - baseline_subset.accuracy
        effect.delta_per_class = {
            label: ablated.per_class[label].f1 - baseline_subset.per_class[label].f1
            for label in labels
        }

        for example in subset:
            was_right = baseline.predictions.get(example.id) == example.label
            now_right = result.predictions.get(example.id) == example.label
            if not was_right and now_right:
                effect.fixed.append(example.id)
            elif was_right and not now_right:
                effect.broken.append(example.id)

        effects.append(effect)

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
