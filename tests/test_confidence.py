"""The bootstrap interval is what turns a number into a verdict."""

from promptxray.ablation import BlockEffect, bootstrap_ci
from promptxray.blocks import Block

LABELS = ["a", "b"]
BLOCK = Block(index=0, text="some instruction", pinned=False)


def test_identical_predictions_give_a_zero_interval():
    pairs = [("a", "a"), ("b", "b")] * 15
    low, high = bootstrap_ci(pairs, pairs, LABELS, resamples=200)
    assert abs(low) < 1e-9 and abs(high) < 1e-9


def test_clear_improvement_keeps_the_interval_above_zero():
    baseline = [("a", "b")] * 15 + [("b", "b")] * 15   # every 'a' wrong
    ablated = [("a", "a")] * 15 + [("b", "b")] * 15    # all correct
    low, high = bootstrap_ci(baseline, ablated, LABELS, resamples=300)
    assert low > 0 and high > 0


def test_tiny_sample_returns_no_interval():
    pairs = [("a", "a")] * 3
    low, high = bootstrap_ci(pairs, pairs, LABELS, resamples=200)
    assert low != low  # NaN


def test_verdict_defers_when_the_interval_crosses_zero():
    effect = BlockEffect(block=BLOCK, delta_macro_f1=0.04, ci_low=-0.09, ci_high=0.17)
    assert effect.verdict == "too noisy to call"
    assert effect.confident is False


def test_verdict_commits_when_the_interval_does_not():
    helps = BlockEffect(block=BLOCK, delta_macro_f1=-0.12, ci_low=-0.20, ci_high=-0.05)
    hurts = BlockEffect(block=BLOCK, delta_macro_f1=0.10, ci_low=0.03, ci_high=0.18)
    assert helps.verdict == "carries its weight"
    assert hurts.verdict == "hurts the score"
    assert helps.confident and hurts.confident


def test_tight_interval_around_zero_is_a_real_no_effect():
    effect = BlockEffect(block=BLOCK, delta_macro_f1=0.0, ci_low=-0.004, ci_high=0.004)
    assert effect.verdict == "no effect"
