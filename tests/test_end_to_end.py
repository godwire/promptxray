"""The whole pipeline on the mock provider - no network, no API key."""

from pathlib import Path

from promptxray import blocks as blocks_mod
from promptxray import dataset as dataset_mod
from promptxray.ablation import ablate
from promptxray.cache import Cache
from promptxray.cli import main
from promptxray.metrics import score
from promptxray.providers import build
from promptxray.runner import run

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ROOT / "examples" / "prompt.txt"
DATA = ROOT / "examples" / "data.csv"


def _baseline():
    text = PROMPT.read_text(encoding="utf-8")
    blocks = blocks_mod.parse_blocks(text)
    examples = dataset_mod.load(DATA)
    labels = dataset_mod.labels_of(examples)
    provider = build("mock", "mock-classifier")
    cache = Cache(enabled=False)
    result = run(blocks_mod.render(blocks), examples, labels, provider, cache, workers=2)
    gold = {e.id: e.label for e in examples}
    return blocks, examples, labels, provider, cache, result, score(gold, result.predictions, labels)


def test_baseline_scores_the_example_dataset():
    *_, result, result_score = _baseline()
    assert result_score.total == len(result.predictions)
    assert 0.5 < result_score.accuracy < 1.0  # imperfect on purpose


def test_ablation_finds_a_useful_and_a_harmful_block():
    blocks, examples, labels, provider, cache, result, result_score = _baseline()
    report = ablate(blocks, examples, labels, provider, cache, result, result_score,
                    subset_size=0, workers=2)

    tested = [e for e in report.effects if not e.skipped_reason]
    assert any(e.delta_macro_f1 < -0.01 for e in tested), "no block carries its weight"
    assert any(e.delta_macro_f1 > 0.01 for e in tested), "no block hurts the score"
    assert any(e.verdict == "no effect" for e in tested), "no dead block detected"


def test_pinned_blocks_are_never_removed():
    blocks, examples, labels, provider, cache, result, result_score = _baseline()
    report = ablate(blocks, examples, labels, provider, cache, result, result_score,
                    subset_size=10, workers=2)
    for effect in report.effects:
        if effect.block.pinned:
            assert effect.skipped_reason


def test_cli_writes_a_report(tmp_path):
    out = tmp_path / "report.html"
    code = main([
        "ablate", "--prompt", str(PROMPT), "--data", str(DATA),
        "--provider", "mock", "--no-cache", "--subset-size", "20",
        "--report", str(out),
    ])
    assert code == 0
    html = out.read_text(encoding="utf-8")
    assert "What each block of the prompt does" in html
    assert "Confusion matrix" in html


def test_cli_run_accepts_custom_cache_path(tmp_path):
    cache_path = tmp_path / "custom.sqlite"
    code = main([
        "run", "--prompt", str(PROMPT), "--data", str(DATA),
        "--provider", "mock", "--cache-path", str(cache_path),
    ])
    assert code == 0
    assert cache_path.exists()


def test_cli_run_reuses_cache_across_two_calls(tmp_path):
    cache_path = tmp_path / "shared.sqlite"
    first = main([
        "run", "--prompt", str(PROMPT), "--data", str(DATA),
        "--provider", "mock", "--cache-path", str(cache_path),
    ])
    second = main([
        "run", "--prompt", str(PROMPT), "--data", str(DATA),
        "--provider", "mock", "--cache-path", str(cache_path),
    ])
    assert first == second == 0


def test_cli_cache_subcommand_reports_then_clears(tmp_path, capsys):
    cache_path = tmp_path / "c.sqlite"
    assert main(["run", "--prompt", str(PROMPT), "--data", str(DATA),
                 "--provider", "mock", "--cache-path", str(cache_path)]) == 0

    assert main(["cache", "--cache-path", str(cache_path)]) == 0
    out = capsys.readouterr().out
    assert "entries" in out
    assert "cache" in out

    assert main(["cache", "--cache-path", str(cache_path), "--clear"]) == 0
    out = capsys.readouterr().out
    assert "cleared" in out
    # after clearing, a follow-up cache report shows zero entries
    assert main(["cache", "--cache-path", str(cache_path)]) == 0
    final = capsys.readouterr().out
    assert "entries    0" in final
