# Contributing

Thanks for taking the time. Bug reports with a reproducible example are the
most useful thing you can send, and small focused pull requests are the
easiest to merge.

## Set up

```bash
git clone https://github.com/<you>/promptxray
cd promptxray
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
pytest -q
ruff check src tests
```

Both must pass; CI runs the same two commands on Python 3.10 to 3.13.

## Ground rules

- **No runtime dependencies.** The package runs on the standard library only,
  so it installs anywhere in seconds. Development tools are fine in `[dev]`.
- **Tests run offline.** Use the `mock` provider or a small fake `Provider`
  subclass in the test itself. Never call a real API from the test suite.
- **Numbers must come from measurements.** Every verdict and explanation in the
  report is derived from what the run actually observed. If the data cannot
  support a claim, the tool says so rather than guessing.
- **Stay a tool, not a platform.** No server, accounts or dashboards. One
  question, one answer, one HTML file.

## Reporting a bug

Open an issue with the command you ran, the full output, your Python version
and provider. A tiny prompt and five rows of data that reproduce the problem
make a fix much faster.

## Releasing (maintainers)

1. Update `CHANGELOG.md` and the version in `pyproject.toml` and
   `src/promptxray/__init__.py`.
2. Commit, then tag: `git tag v0.X.Y && git push --tags`.
3. The `publish` workflow builds the package and uploads it to PyPI.
