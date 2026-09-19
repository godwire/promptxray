"""Greedy minimisation must catch redundancy that one-shot ablation cannot see.

The setup: two blocks that say the same thing in different words. Either one
is enough for the model to route refund requests correctly. Removing one on its
own changes nothing, so leave-one-out ablation calls both of them useless, and
a one-shot cut deletes both - which breaks every refund example. The greedy
search removes one, re-measures, and keeps the other.
"""

from promptxray import blocks as blocks_mod
from promptxray.cache import Cache
from promptxray.dataset import Example
from promptxray.providers import Answer, Provider
from promptxray.suggest import estimate_calls, minimise_greedy, minimise_one_shot, suggest

PROMPT = """Sort support messages into urgent or normal.

Flag refund requests as urgent.

Anything that mentions getting money back is urgent too.

Be kind and professional.

Answer with one word. [[keep]]

Message: {input}"""

LABELS = ["normal", "urgent"]


class RefundAwareModel(Provider):
    """Routes refunds correctly as long as at least one of the two rules is present."""

    name = "refund-aware"

    def complete(self, prompt: str) -> Answer:
        instructions, _, body = prompt.rpartition("Message:")
        knows_refunds = ("refund requests as urgent" in instructions
                         or "money back is urgent" in instructions)
        label = "urgent" if knows_refunds and "refund" in body.lower() else "normal"
        return Answer(text=label, input_tokens=len(prompt) // 4, output_tokens=1)


def _examples() -> list[Example]:
    rows = [(f"I would like a refund for order {i}", "urgent") for i in range(14)]
    rows += [(f"How do I change my avatar, question {i}?", "normal") for i in range(14)]
    return [Example(id=i, text=text, label=label) for i, (text, label) in enumerate(rows)]


def _setup():
    blocks = blocks_mod.parse_blocks(PROMPT)
    return blocks, _examples(), RefundAwareModel("refund-aware"), Cache(enabled=False)


def test_one_shot_deletes_both_twins_and_would_break_the_prompt():
    blocks, examples, model, cache = _setup()
    kept, _steps, _calls = minimise_one_shot(blocks, examples, LABELS, model, cache,
                                             workers=2, resamples=200)
    kept_text = blocks_mod.render(kept)
    assert "refund requests" not in kept_text
    assert "money back" not in kept_text


def test_greedy_keeps_exactly_one_twin():
    blocks, examples, model, cache = _setup()
    kept, steps, _redundancies, _calls = minimise_greedy(
        blocks, examples, LABELS, model, cache, workers=2, resamples=200)
    kept_text = blocks_mod.render(kept)

    twins_left = ("refund requests" in kept_text) + ("money back" in kept_text)
    assert twins_left == 1
    assert "Be kind" not in kept_text

    removed = {step.removed.index for step in steps}
    assert 3 in removed                  # the filler goes
    assert len(removed & {1, 2}) == 1    # exactly one of the twins goes


def test_greedy_names_the_redundant_pair():
    blocks, examples, model, cache = _setup()
    _kept, _steps, redundancies, _calls = minimise_greedy(
        blocks, examples, LABELS, model, cache, workers=2, resamples=200)

    assert len(redundancies) == 1
    pair = {redundancies[0].block.index, redundancies[0].twin.index}
    assert pair == {1, 2}


def test_pinned_blocks_are_never_candidates():
    blocks, examples, model, cache = _setup()
    kept, _steps, _redundancies, _calls = minimise_greedy(
        blocks, examples, LABELS, model, cache, workers=2, resamples=200)
    assert "{input}" in blocks_mod.render(kept)
    assert "Answer with one word." in blocks_mod.render(kept)


def test_max_steps_limits_the_search():
    blocks, examples, model, cache = _setup()
    _kept, steps, _redundancies, _calls = minimise_greedy(
        blocks, examples, LABELS, model, cache, workers=2, resamples=200, max_steps=1)
    assert len(steps) == 1


def test_suggest_end_to_end_reports_savings():
    blocks, examples, model, cache = _setup()
    result = suggest(blocks, examples, LABELS, model, cache, subset_size=0,
                     holdout_ratio=0.3, workers=2, resamples=200)

    assert result.worth_it
    assert result.after_macro_f1 >= result.before_macro_f1 - 0.005
    assert result.economics.tokens_after < result.economics.tokens_before
    assert result.economics.estimated is False  # the fake model reports usage


def test_estimate_grows_quadratically_for_greedy_only():
    blocks = blocks_mod.parse_blocks(PROMPT)
    greedy = estimate_calls(blocks, evaluation_size=10, holdout_size=5, strategy="greedy")
    one_shot = estimate_calls(blocks, evaluation_size=10, holdout_size=5, strategy="one-shot")
    assert greedy > one_shot
