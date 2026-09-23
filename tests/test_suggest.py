"""`suggest` must never grade its own homework on the examples it learned from."""

from pathlib import Path

from promptxray import blocks as blocks_mod
from promptxray import dataset as dataset_mod
from promptxray.cache import Cache
from promptxray.dataset import stratified_split
from promptxray.providers import build
from promptxray.suggest import suggest

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ROOT / "examples" / "prompt.txt"
DATA = ROOT / "examples" / "data.csv"


def test_split_is_disjoint_and_keeps_every_class():
    examples = dataset_mod.load(DATA)
    train, holdout = stratified_split(examples, 0.5, seed=0)

    assert not {e.id for e in train} & {e.id for e in holdout}
    assert len(train) + len(holdout) == len(examples)
    assert {e.label for e in train} == {e.label for e in holdout}


def test_split_is_reproducible():
    examples = dataset_mod.load(DATA)
    first, _ = stratified_split(examples, 0.5, seed=7)
    second, _ = stratified_split(examples, 0.5, seed=7)
    assert [e.id for e in first] == [e.id for e in second]



def _run_suggest(seed: int = 0):
    blocks = blocks_mod.parse_blocks(PROMPT.read_text(encoding="utf-8"))
    examples = dataset_mod.load(DATA)
    labels = dataset_mod.labels_of(examples)
    return suggest(blocks, examples, labels, build("mock", "mock-classifier"),
                   Cache(enabled=False), subset_size=0, workers=2, seed=seed, resamples=120)


def test_suggested_prompt_keeps_the_pinned_blocks():
    result = _run_suggest()
    assert "{input}" in result.prompt_text
    assert "single label" in result.prompt_text


def test_dropped_blocks_are_gone_from_the_suggestion():
    result = _run_suggest()
    for block, _reason in result.dropped:
        assert block.text not in result.prompt_text


def test_holdout_is_scored_and_reported():
    result = _run_suggest()
    if result.dropped:
        assert result.holdout_size > 0
        assert 0.0 < result.after_macro_f1 <= 1.0
        assert result.chars_saved > 0
