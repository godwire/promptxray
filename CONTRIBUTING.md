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
mypy
```

All three must pass; CI runs the same checks on Python 3.10 to 3.13. This
project follows the [Code of Conduct](CODE_OF_CONDUCT.md) — please read it
before participating.

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

1. Move the `CHANGELOG.md` entries under a `## [0.X.Y] - YYYY-MM-DD` heading
   and set `__version__` in `src/promptxray/__init__.py` (the only place the
   version lives).
2. Commit, push, wait for CI to go green, then tag:
   `git tag v0.X.Y && git push origin v0.X.Y`.
3. The `publish` workflow runs the tests, checks the tag matches
   `__version__`, uploads to PyPI, and creates the GitHub release with that
   version's changelog section as its notes.

One-time setup, before the very first release: on PyPI, add a trusted
publisher at <https://pypi.org/manage/account/publishing/> with project
`promptxray`, owner `godwire`, repository `promptxray`, workflow
`publish.yml` and environment `pypi`.
