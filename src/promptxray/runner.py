"""Run one prompt over a list of examples and collect the predictions."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .cache import Cache
from .dataset import Example
from .metrics import normalise
from .providers import Provider

# Rough prices per 1M tokens, only used for the estimate shown in the report.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}


@dataclass
class RunResult:
    predictions: dict[int, str] = field(default_factory=dict)
    raw: dict[int, str] = field(default_factory=dict)
    api_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model: str) -> float | None:
        price = None
        for name, values in PRICES.items():
            if name in model:
                price = values
                break
        if price is None:
            return None
        return self.input_tokens / 1e6 * price[0] + self.output_tokens / 1e6 * price[1]


def run(
    prompt_template: str,
    examples: list[Example],
    labels: list[str],
    provider: Provider,
    cache: Cache,
    workers: int = 4,
    progress_label: str = "",
) -> RunResult:
    """Fill {input} with each example, ask the model, normalise the answer."""
    result = RunResult()
    done = 0
    total = len(examples)

    def ask(example: Example) -> tuple[Example, str, bool, int, int]:
        filled = prompt_template.replace("{input}", example.text)
        cached = cache.get(provider.model, filled)
        if cached is not None:
            return example, cached.text, True, cached.input_tokens, cached.output_tokens
        answer = provider.complete(filled)
        cache.put(provider.model, filled, answer)
        return example, answer.text, False, answer.input_tokens, answer.output_tokens

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for example, text, was_cached, in_tokens, out_tokens in pool.map(ask, examples):
            result.raw[example.id] = text
            result.predictions[example.id] = normalise(text, labels)
            if was_cached:
                result.cache_hits += 1
            else:
                result.api_calls += 1
                result.input_tokens += in_tokens
                result.output_tokens += out_tokens

            done += 1
            if progress_label and sys.stderr.isatty():
                print(f"\r{progress_label} {done}/{total}", end="", file=sys.stderr)

    if progress_label and sys.stderr.isatty():
        print("", file=sys.stderr)
    return result
