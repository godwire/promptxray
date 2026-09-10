"""Command line entry point: promptxray run | ablate | diff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import blocks as blocks_mod
from . import dataset as dataset_mod
from .ablation import ablate
from .cache import DEFAULT_PATH, Cache
from .metrics import score
from .providers import ProviderError, build
from .report import render_report
from .runner import run


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="CSV or JSONL file with labelled examples")
    parser.add_argument("--model", default="mock-classifier", help="model name")
    parser.add_argument("--provider", default="mock",
                        choices=["mock", "ollama", "lmstudio", "openrouter", "groq",
                                 "openai", "anthropic", "local"])
    parser.add_argument("--base-url", default=None, help="override the API base URL")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N examples")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("--no-cache", action="store_true", help="ignore the on-disk cache")
    parser.add_argument("--cache-path", default=None, help="SQLite cache file (default: .promptxray-cache.sqlite)")
    parser.add_argument("--report", default=None, help="where to write the HTML report")
    parser.add_argument("--json", dest="json_out", default=None, help="also write raw results as JSON")


def _make_cache(args) -> Cache:
    return Cache(path=DEFAULT_PATH if args.cache_path is None else args.cache_path,
                 enabled=not args.no_cache)


def _load(args) -> tuple[str, list, list[str]]:
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    examples = dataset_mod.load(args.data, args.text_column, args.label_column)
    if args.limit:
        examples = examples[: args.limit]
    return prompt_text, examples, dataset_mod.labels_of(examples)


def cmd_run(args) -> int:
    prompt_text, examples, labels = _load(args)
    blocks = blocks_mod.parse_blocks(prompt_text)
    blocks_mod.validate(blocks)

    provider = build(args.provider, args.model, args.base_url)
    cache = _make_cache(args)
    try:
        result = run(blocks_mod.render(blocks), examples, labels, provider, cache,
                     workers=args.workers, progress_label="scoring")
    finally:
        cache.close()

    gold = {e.id: e.label for e in examples}
    result_score = score(gold, result.predictions, labels)

    print(f"accuracy  {result_score.accuracy:.1%}")
    print(f"macro F1  {result_score.macro_f1:.3f}")
    print(f"unparsed  {result_score.unparsed}")
    print(f"calls     {result.api_calls} ({result.cache_hits} from cache)")

    _write_report(args, prompt_text, examples, labels, provider, result, result_score, None)
    return 0


def cmd_ablate(args) -> int:
    prompt_text, examples, labels = _load(args)
    blocks = blocks_mod.parse_blocks(prompt_text)
    blocks_mod.validate(blocks)

    provider = build(args.provider, args.model, args.base_url)
    cache = _make_cache(args)
    try:
        baseline = run(blocks_mod.render(blocks), examples, labels, provider, cache,
                       workers=args.workers, progress_label="baseline")
        gold = {e.id: e.label for e in examples}
        baseline_score = score(gold, baseline.predictions, labels)

        report = ablate(blocks, examples, labels, provider, cache, baseline, baseline_score,
                        subset_size=args.subset_size, workers=args.workers, seed=args.seed)
    finally:
        cache.close()

    print(f"\nbaseline macro F1 {baseline_score.macro_f1:.3f} on {baseline_score.total} examples")
    print(f"ablation subset   {report.subset_size} examples\n")
    ranked = sorted([e for e in report.effects if not e.skipped_reason],
                    key=lambda e: e.contribution, reverse=True)
    for effect in ranked:
        print(f"  #{effect.block.index + 1:<3} {effect.delta_macro_f1:+.3f}  "
              f"{effect.verdict:<20} {effect.block.preview[:60]}")
    dead = [e for e in ranked if not e.fixed and not e.broken]
    if dead:
        print(f"\n{len(dead)} of {len(ranked)} tested blocks changed nothing at all.")

    _write_report(args, prompt_text, examples, labels, provider, baseline, baseline_score, report)
    return 0


def cmd_diff(args) -> int:
    examples = dataset_mod.load(args.data, args.text_column, args.label_column)
    if args.limit:
        examples = examples[: args.limit]
    labels = dataset_mod.labels_of(examples)
    gold = {e.id: e.label for e in examples}

    provider = build(args.provider, args.model, args.base_url)
    cache = _make_cache(args)
    try:
        results = []
        for path in (args.before, args.after):
            text = Path(path).read_text(encoding="utf-8")
            blocks = blocks_mod.parse_blocks(text)
            blocks_mod.validate(blocks)
            results.append(run(blocks_mod.render(blocks), examples, labels, provider, cache,
                               workers=args.workers, progress_label=Path(path).name))
    finally:
        cache.close()

    before, after = results
    before_score = score(gold, before.predictions, labels)
    after_score = score(gold, after.predictions, labels)

    fixed, broken = [], []
    for example in examples:
        was = before.predictions.get(example.id) == example.label
        now = after.predictions.get(example.id) == example.label
        if not was and now:
            fixed.append(example)
        elif was and not now:
            broken.append(example)

    print(f"{Path(args.before).name}: macro F1 {before_score.macro_f1:.3f}")
    print(f"{Path(args.after).name}:  macro F1 {after_score.macro_f1:.3f} "
          f"({after_score.macro_f1 - before_score.macro_f1:+.3f})")
    print(f"\nfixed  {len(fixed)}")
    for example in fixed[:10]:
        print(f"  + [{example.label}] {example.text[:70]}")
    print(f"broken {len(broken)}")
    for example in broken[:10]:
        got = after.predictions.get(example.id)
        print(f"  - [{example.label} -> {got}] {example.text[:70]}")

    if args.fail_under is not None and after_score.macro_f1 < args.fail_under:
        print(f"\nmacro F1 {after_score.macro_f1:.3f} is below --fail-under {args.fail_under}")
        return 1
    return 0


def cmd_cache(args) -> int:
    """Show cache stats or wipe the on-disk cache."""
    cache = Cache(path=DEFAULT_PATH if args.cache_path is None else args.cache_path,
                  enabled=True)
    try:
        if args.clear:
            removed = cache.clear()
            print(f"cleared {removed} cached answer(s) from {cache.path}")
            return 0
        stats = cache.stats
        print(f"cache      {cache.path}")
        print(f"entries    {stats['entries']}")
        print(f"hits       {stats['hits']}")
        print(f"misses     {stats['misses']}")
        return 0
    finally:
        cache.close()


def _write_report(args, prompt_text, examples, labels, provider, result, result_score, ablation) -> None:
    if args.report:
        html = render_report(
            prompt_path=args.prompt,
            data_path=args.data,
            model=provider.model,
            provider_name=provider.name,
            baseline=result,
            baseline_score=result_score,
            examples=examples,
            ablation=ablation,
        )
        Path(args.report).write_text(html, encoding="utf-8")
        print(f"\nreport    {args.report}")

    if args.json_out:
        payload = {
            "accuracy": result_score.accuracy,
            "macro_f1": result_score.macro_f1,
            "unparsed": result_score.unparsed,
            "per_class": {
                label: {
                    "precision": c.precision, "recall": c.recall,
                    "f1": c.f1, "support": c.support,
                }
                for label, c in result_score.per_class.items()
            },
            "confusion": result_score.confusion,
        }
        if ablation:
            payload["blocks"] = [
                {
                    "index": e.block.index,
                    "text": e.block.text,
                    "pinned": e.block.pinned,
                    "delta_macro_f1": e.delta_macro_f1,
                    "verdict": e.verdict,
                    "fixed": e.fixed,
                    "broken": e.broken,
                }
                for e in ablation.effects
            ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json      {args.json_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="promptxray",
        description="Find out which line of your prompt is causing your classifier's errors.",
    )
    parser.add_argument("--version", action="version", version=f"promptxray {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="score a prompt on a labelled dataset")
    p_run.add_argument("--prompt", required=True)
    _add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_ablate = sub.add_parser("ablate", help="measure what every block of the prompt does")
    p_ablate.add_argument("--prompt", required=True)
    p_ablate.add_argument("--subset-size", type=int, default=60,
                          help="how many examples each ablated run uses")
    p_ablate.add_argument("--seed", type=int, default=0)
    _add_common(p_ablate)
    p_ablate.set_defaults(func=cmd_ablate)

    p_diff = sub.add_parser("diff", help="compare two prompt versions example by example")
    p_diff.add_argument("--before", required=True)
    p_diff.add_argument("--after", required=True)
    p_diff.add_argument("--fail-under", type=float, default=None,
                        help="exit with code 1 if the new prompt scores below this macro F1")
    _add_common(p_diff)
    p_diff.set_defaults(func=cmd_diff)

    p_cache = sub.add_parser("cache", help="show or wipe the on-disk model-answer cache")
    p_cache.add_argument("--cache-path", default=None,
                         help="SQLite cache file (default: .promptxray-cache.sqlite)")
    p_cache.add_argument("--clear", action="store_true",
                         help="delete every cached answer")
    p_cache.set_defaults(func=cmd_cache)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ProviderError, ValueError, FileNotFoundError) as exc:
        print(f"promptxray: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())