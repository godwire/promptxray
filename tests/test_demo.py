"""`promptxray demo` is the first thing a new user runs, straight after pip install."""

from importlib import resources
from pathlib import Path

from promptxray.cli import main

ROOT = Path(__file__).resolve().parent.parent


def test_demo_runs_offline_and_writes_a_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--no-open"]) == 0

    html = (tmp_path / "promptxray-demo.html").read_text(encoding="utf-8")
    assert "What each block of the prompt does" in html
    # Short names, not a path deep inside site-packages.
    assert "demo/prompt.txt" in html
    assert "site-packages" not in html

    out = capsys.readouterr().out
    assert "baseline macro F1" in out
    assert "--provider ollama" in out


def test_demo_leaves_no_cache_file_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["demo", "--no-open"])
    assert not (tmp_path / ".promptxray-cache.sqlite").exists()


def test_bundled_demo_matches_the_examples_folder():
    # The README walks through examples/; the demo must show the same thing.
    demo = resources.files("promptxray") / "demo"
    for name in ("prompt.txt", "data.csv"):
        bundled = (demo / name).read_text(encoding="utf-8")
        assert bundled == (ROOT / "examples" / name).read_text(encoding="utf-8"), name
