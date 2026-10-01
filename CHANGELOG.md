# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-10-01

The first release published to PyPI, and the first tagged version of the
GitHub Action (`uses: godwire/promptxray@v0.4.0`). Versions 0.1.0 to 0.3.0
were development milestones and were never published.

### Added
- `promptxray demo`: runs the bundled example end to end, offline, with no API
  key and no files of your own, and opens the report. The fastest way to see
  what the tool does straight after `pip install`.
- `--provider gemini` for Google's Generative Language API (`GEMINI_API_KEY`),
  another free-tier option alongside Ollama, OpenRouter and Groq.
- `--provider azure` for Azure OpenAI Service, with `--base-url` and
  `--api-version`. Both are also inputs on the GitHub Action.
- `py.typed` marker: the package ships type information for anyone importing
  it as a library.
- Tagging a release now also creates the GitHub release, with this changelog's
  section as its notes. The publish workflow runs the test suite first.
- `mypy` in CI, and unit tests for the network providers (OpenAI-compatible,
  Anthropic, Gemini, Azure) including retry-on-rate-limit, with `urlopen`
  stubbed so they need no network and no API key.
- `CODE_OF_CONDUCT.md` and `SECURITY.md`.

### Changed
- Development status is now Beta.
- The version is defined once, in `src/promptxray/__init__.py`.

### Fixed
- The quick start no longer depends on files that only exist in a git clone;
  it uses `promptxray demo`.

## 0.3.0 - 2026-09-19

### Added
- `suggest` now removes blocks one at a time and re-measures after each
  removal (greedy backward elimination). It catches blocks that duplicate each
  other, which one-shot ablation cannot see, and names the pair in its output.
- `suggest --strategy one-shot` keeps the previous single-cut behaviour for a
  cheaper run.
- `suggest` reports prompt tokens per call before and after, the saving per
  million calls, and the saving in dollars with `--price-per-mtok`.
- `suggest --max-steps` to cap the greedy search; the worst-case number of
  model calls is printed before a run starts, and the real number after it.
- A GitHub Action that runs `diff` or `ablate` in CI, installs Ollama on the
  runner for a free open model, writes the result to the job summary and
  uploads the HTML report.
- A manual `real model` workflow that runs the case study against an open
  model on GitHub's machines.
- A live demo report on GitHub Pages.
- `ruff` in CI, Python 3.13 in the test matrix, contributor guide, and issue and
  pull request templates.

### Changed
- Blocks that are too noisy to judge are shown with a hatched background in the
  report, so they can no longer be mistaken for blocks with no effect.

### Fixed
- `suggest` no longer runs the full prompt over the training examples twice.

## 0.2.0 - 2026-09-08

### Added
- 90% bootstrap confidence interval for every block's effect. A block is only
  called useful or harmful when the interval stays on one side of zero;
  otherwise the verdict is "too noisy to call".
- `suggest` command: proposes a shorter prompt from the blocks that earn their
  place and verifies it on a stratified holdout the selection never saw.
- Release workflow that publishes to PyPI when a version tag is pushed.

### Fixed
- `openrouter` and `groq` providers now use their own endpoints and API key
  variables, with a clear error when the key is missing.

## 0.1.0 - 2026-09-08

### Added
- `run`, `ablate` and `diff` commands.
- Leave-one-out ablation over blank-line-separated prompt blocks, with
  `[[keep]]` pinning and error-enriched subsets.
- Single-file HTML report: the prompt coloured block by block, with hover
  explanations built from measured results.
- SQLite answer cache, built-in mock provider, zero runtime dependencies.

[0.4.0]: https://github.com/godwire/promptxray/releases/tag/v0.4.0
