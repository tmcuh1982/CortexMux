# CortexMux 1.0.0

Release date: 2026-10-21.

The first major stable release brings the 0.6 release-candidate series and YAML
exports into version 1.0.0. This release keeps the existing routing, provider
configuration and JSON output behavior.

## Highlights

- Typed synchronous and asynchronous APIs with deterministic provider routing.
- Local Ollama and ComfyUI support; opt-in OpenAI, Gemini, Grok, DeepSeek,
  Qwen Cloud, TypeSafe and Codex integrations.
- Deterministic local data analysis, structured calculation verification using
  Decimal, and opt-in bounded web extraction with source provenance.
- `response.to_yaml()` and CLI `--format text|json|yaml`, retaining `--json`
  compatibility. YAML requires the optional `cortexmux[yaml]` extra.
- Secret redaction, explicit remote-host approval and network-free unit tests.

## Installation

Python 3.11 or newer is required. From the GitHub release assets:

```bash
python -m pip install ./cortexmux-1.0.0-py3-none-any.whl
# Include YAML support:
python -m pip install './cortexmux-1.0.0-py3-none-any.whl[yaml]'
```

Ollama, ComfyUI and model weights are external prerequisites. Optional providers
remain opt-in; remote services require their documented configuration and
credentials. No downstream application is modified by this release.

## Compatibility and integration limits

The major version does not remove the documented restrictions of optional
integrations. In particular, the Codex adapter remains experimental, requires
Codex CLI 0.140.0 and retains its documented isolation limits. See
[Codex provider](codex.md), [configuration](configuration.md) and
[output formats](output-formats.md).

Authenticated services and public-network integrations are excluded from the
offline release checks. Their integration tests require explicit opt-in.

## Artifacts

- `cortexmux-1.0.0-py3-none-any.whl`
- `cortexmux-1.0.0.tar.gz`
- `SHA256SUMS`

## Release validation

- Ruff lint and formatting checks: passed.
- Strict mypy: passed for 90 source files.
- Offline test suite: 362 passed, 3 opt-in integration tests skipped.
- `git diff --check`: passed.
- Isolated source archive and wheel build: passed.
- Twine metadata checks for both 1.0.0 distributions: passed.
- Wheel smoke check: version 1.0.0 and YAML response export passed.

The historical `dist/` directory also contains versioned subdirectories, so
Twine checks target the two release files explicitly.
