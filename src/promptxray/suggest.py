"""Propose a shorter prompt, then prove it on examples that were not used to pick it.

Two ways to choose which blocks go:

one-shot  Judge every block once, against the full prompt, and drop all the ones
          that are useless or harmful in a single cut. Cheap, but blind to
          redundancy: when two blocks say the same thing, removing either one
          alone changes nothing, both look useless, and both get dropped - even
          though together they were doing the work.

greedy    Backward elimination. Drop the single weakest block, re-measure what
          is left, and repeat until every remaining block earns its place. Each
          decision is made against the prompt as it currently stands, so a
          block that only mattered as a backup is caught the moment its twin is
          gone. This is the default.

Either way the dataset is split first. Blocks are judged on the train half, and
the proposed prompt is scored on a holdout the selection never saw. Measuring
on the same examples that did the choosing would flatter every prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import blocks as blocks_mod
from .ablation import BlockEffect, measure_removal
from .blocks import Block
from .cache import Cache
from .dataset import Example, error_enriched_subset, stratified_split
from .metrics import score
from .providers import Provider
from .runner import RunResult, run, tokens_per_call

REMOVABLE = {"no effect", "hurts the score"}


@dataclass
class Step:
    """One round of elimination: which block went, and why."""

    number: int
    removed: Block
    effect: BlockEffect
    candidates_tested: int


@dataclass
class Redundancy:
    """A block that looked useless until another block was removed."""

    block: Block
    twin: Block
    became_useful_at_step: int


@dataclass
class Economics:
    tokens_before: float = 0.0
    tokens_after: float = 0.0
    estimated: bool = False

    @property
    def saved_per_call(self) -> float:
        return self.tokens_before - self.tokens_after

    @property
    def saved_share(self) -> float:
        return self.saved_per_call / self.tokens_before if self.tokens_before else 0.0

    def dollars_per_million_calls(self, price_per_mtok: float) -> float:
        """Saving on input tokens for one million calls at the given price."""
        return self.saved_per_call * price_per_mtok


@dataclass
class Suggestion:
    strategy: str = "greedy"
    kept: list[Block] = field(default_factory=list)
    dropped: list[tuple[Block, str]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    redundancies: list[Redundancy] = field(default_factory=list)
    prompt_text: str = ""
    train_size: int = 0
    eval_size: int = 0
    holdout_size: int = 0
    before_macro_f1: float = 0.0
    after_macro_f1: float = 0.0
    fixed: list[Example] = field(default_factory=list)
    broken: list[Example] = field(default_factory=list)
    chars_saved: int = 0
    economics: Economics = field(default_factory=Economics)
    api_calls: int = 0

    @property
    def delta(self) -> float:
        return self.after_macro_f1 - self.before_macro_f1

    @property
    def worth_it(self) -> bool:
        """Shorter and no worse is a win; anything that loses score is not."""
        return self.delta >= -0.005 and bool(self.dropped)


def _reason(effect: BlockEffect) -> str:
    if effect.verdict == "hurts the score":
        return f"costs {effect.delta_macro_f1:+.3f} macro F1"
    return "no measurable effect"


def _one_round(
    current: list[Block],
    current_run: RunResult,
    evaluation: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    skip_indexes: set[int],
    workers: int,
    seed: int,
    resamples: int,
    label: str,
) -> tuple[dict[int, BlockEffect], dict[int, RunResult], int]:
    """Remove each candidate from the current prompt in turn and measure it."""
    effects: dict[int, BlockEffect] = {}
    runs: dict[int, RunResult] = {}
    calls = 0
    for block in current:
        if block.pinned or block.index in skip_indexes:
            continue
        variant = blocks_mod.render(current, skip=block.index)
        result = run(variant, evaluation, labels, provider, cache, workers=workers,
                     progress_label=f"{label}: without #{block.index + 1}")
        calls += result.api_calls
        effects[block.index] = measure_removal(
            block, current_run, result, evaluation, labels, resamples, seed)
        runs[block.index] = result
    return effects, runs, calls


def minimise_greedy(
    blocks: list[Block],
    evaluation: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    workers: int = 4,
    seed: int = 0,
    resamples: int = 400,
    max_steps: int = 0,
    full_run: RunResult | None = None,
) -> tuple[list[Block], list[Step], list[Redundancy], int]:
    """Backward elimination over the unpinned blocks.

    Blocks that are confidently useful in the first round are not re-tested in
    later rounds. That is a deliberate shortcut: it cuts the cost from roughly
    n squared runs to a fraction of that, and a block which clearly helps the
    full prompt almost never becomes useless as the prompt gets shorter.
    """
    current = list(blocks)
    if full_run is None:
        full_run = run(blocks_mod.render(current), evaluation, labels, provider, cache,
                       workers=workers, progress_label="greedy: full prompt")
        calls = full_run.api_calls
    else:
        calls = 0
    current_run = full_run

    steps: list[Step] = []
    first_verdict: dict[int, str] = {}
    history: dict[int, list[str]] = {}
    protected: set[int] = set()

    while True:
        if max_steps and len(steps) >= max_steps:
            break

        effects, runs, spent = _one_round(
            current, current_run, evaluation, labels, provider, cache, protected,
            workers, seed, resamples, label=f"greedy round {len(steps) + 1}")
        calls += spent
        if not effects:
            break

        for index, effect in effects.items():
            history.setdefault(index, []).append(effect.verdict)
            if not steps:
                first_verdict[index] = effect.verdict
                if effect.verdict == "carries its weight":
                    protected.add(index)

        candidates = [e for e in effects.values() if e.verdict in REMOVABLE]
        if not candidates:
            break

        # Remove the block whose absence helps most; on a tie, the longest one,
        # because it saves the most tokens for the same score.
        chosen = max(candidates, key=lambda e: (e.delta_macro_f1, len(e.block.text)))
        steps.append(Step(number=len(steps) + 1, removed=chosen.block, effect=chosen,
                          candidates_tested=len(effects)))
        current = [b for b in current if b.index != chosen.block.index]
        current_run = runs[chosen.block.index]

    # A block that did nothing in round one but carries its weight later was a
    # backup for something removed in between. Name the pair.
    redundancies: list[Redundancy] = []
    by_index = {b.index: b for b in blocks}
    for index, verdicts in history.items():
        if first_verdict.get(index) != "no effect":
            continue
        for round_number, verdict in enumerate(verdicts):
            if verdict == "carries its weight" and round_number > 0:
                twin = steps[round_number - 1].removed
                redundancies.append(Redundancy(block=by_index[index], twin=twin,
                                               became_useful_at_step=round_number))
                break

    return current, steps, redundancies, calls


def minimise_one_shot(
    blocks: list[Block],
    evaluation: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    workers: int = 4,
    seed: int = 0,
    resamples: int = 400,
    full_run: RunResult | None = None,
) -> tuple[list[Block], list[Step], int]:
    """Judge every block once against the full prompt and cut in one go."""
    already_paid = full_run is not None
    if full_run is None:
        full_run = run(blocks_mod.render(blocks), evaluation, labels, provider, cache,
                       workers=workers, progress_label="one-shot: full prompt")
    effects, _runs, spent = _one_round(
        blocks, full_run, evaluation, labels, provider, cache, set(),
        workers, seed, resamples, label="one-shot")

    steps = [
        Step(number=i + 1, removed=e.block, effect=e, candidates_tested=len(effects))
        for i, e in enumerate(e for e in effects.values() if e.verdict in REMOVABLE)
    ]
    gone = {step.removed.index for step in steps}
    own_calls = 0 if already_paid else full_run.api_calls
    return [b for b in blocks if b.index not in gone], steps, own_calls + spent


def estimate_calls(blocks: list[Block], evaluation_size: int, holdout_size: int,
                   strategy: str, train_size: int = 0) -> int:
    """Worst-case number of model calls, shown before a long run starts."""
    free = sum(1 for b in blocks if not b.pinned)
    if strategy == "one-shot":
        runs = free
    else:
        runs = free * (free + 1) // 2
    return train_size + runs * evaluation_size + 2 * holdout_size


def suggest(
    blocks: list[Block],
    examples: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    subset_size: int = 60,
    holdout_ratio: float = 0.5,
    workers: int = 4,
    seed: int = 0,
    resamples: int = 400,
    strategy: str = "greedy",
    max_steps: int = 0,
) -> Suggestion:
    train, holdout = stratified_split(examples, holdout_ratio, seed)
    full_prompt = blocks_mod.render(blocks)

    # The evaluation set is fixed once, from the train half, so every round of
    # the greedy search compares candidates on the same examples.
    train_run = run(full_prompt, train, labels, provider, cache,
                    workers=workers, progress_label="baseline (train)")
    train_score = score({e.id: e.label for e in train}, train_run.predictions, labels)
    evaluation = error_enriched_subset(train, train_score.wrong_ids, subset_size, seed)

    # The evaluation set is a subset of train, so the train run already holds
    # the full prompt's answers for it. Re-asking the model would be paid twice.
    if strategy == "one-shot":
        kept, steps, calls = minimise_one_shot(
            blocks, evaluation, labels, provider, cache, workers, seed, resamples,
            full_run=train_run)
        redundancies: list[Redundancy] = []
    else:
        kept, steps, redundancies, calls = minimise_greedy(
            blocks, evaluation, labels, provider, cache, workers, seed, resamples, max_steps,
            full_run=train_run)

    candidate = blocks_mod.render(kept)
    result = Suggestion(
        strategy=strategy,
        kept=kept,
        dropped=[(step.removed, _reason(step.effect)) for step in steps],
        steps=steps,
        redundancies=redundancies,
        prompt_text=candidate,
        train_size=len(train),
        eval_size=len(evaluation),
        holdout_size=len(holdout),
        chars_saved=len(full_prompt) - len(candidate),
        api_calls=train_run.api_calls + calls,
    )
    if not steps:
        return result

    # Score both prompts on examples the selection never touched.
    gold = {e.id: e.label for e in holdout}
    before = run(full_prompt, holdout, labels, provider, cache,
                 workers=workers, progress_label="original (holdout)")
    after = run(candidate, holdout, labels, provider, cache,
                workers=workers, progress_label="suggested (holdout)")
    result.api_calls += before.api_calls + after.api_calls

    result.before_macro_f1 = score(gold, before.predictions, labels).macro_f1
    result.after_macro_f1 = score(gold, after.predictions, labels).macro_f1

    for example in holdout:
        was_right = before.predictions.get(example.id) == example.label
        now_right = after.predictions.get(example.id) == example.label
        if not was_right and now_right:
            result.fixed.append(example)
        elif was_right and not now_right:
            result.broken.append(example)

    tokens_before, estimated_before = tokens_per_call(before, full_prompt, holdout)
    tokens_after, estimated_after = tokens_per_call(after, candidate, holdout)
    result.economics = Economics(tokens_before=tokens_before, tokens_after=tokens_after,
                                 estimated=estimated_before or estimated_after)
    return result
