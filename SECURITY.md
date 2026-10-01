# Security policy

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/promptxray/) is
supported. There is no backport policy for older versions.

## Reporting a vulnerability

Please **do not open a public issue** for a security vulnerability.

Instead, use GitHub's private reporting:
[Report a vulnerability](https://github.com/godwire/promptxray/security/advisories/new)
(Security tab → "Report a vulnerability" on this repository). If that page
says private reporting is not enabled, open an issue titled "Security contact
request" with no details at all, and the maintainer will reach out privately.

Include, if you can:

- The version of `promptxray` and your Python version.
- The command or code path that triggers the issue.
- What you expected to happen and what happened instead.

You should get a response within a few days.

## Scope notes

promptxray has no runtime dependencies and runs entirely on your own machine:
it reads a local prompt file and dataset, and makes outbound HTTP requests
only to the model provider you pick with `--provider` (or none at all with
`--provider mock`). The main things worth a security report are:

- An API key or other secret ending up somewhere it should not (a log, the
  on-disk cache, the HTML report).
- A crafted prompt or dataset file causing something worse than a bad score
  (e.g. path traversal, code execution, unbounded resource use).
- A dependency-confusion or supply-chain concern in the published package or
  the GitHub Action.
