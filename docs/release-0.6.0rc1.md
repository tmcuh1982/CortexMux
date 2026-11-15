# CortexMux 0.6.0rc1 — delivery record

Prepared on 2026-09-10. GitHub prerelease publication of `v0.6.0rc1`
was authorized after the validation recorded below. UniversRobot remains
unchanged; adopting this candidate is a separate integration step.

## Change inventory

- `src/cortexmux/providers/codex/{client,provider,schemas,errors,__init__}.py`:
  version-checked stdio transport, account lifecycle, text/structured generation,
  streaming, cancellation, explicit isolation limits and typed public models.
- `src/cortexmux/core/config.py`: opt-in Codex settings.
- `src/cortexmux/core/router.py`: deterministic provider stream cleanup.
- `src/cortexmux/facade.py`: sync/async account operations, generic streams,
  structured-stream options, and cleanup through the shared Runner.
- `src/cortexmux/schemas/responses.py`: optional correlation metadata on a
  validated structured completion.
- `pyproject.toml`, `src/cortexmux/version.py`: candidate version 0.6.0rc1.
- `tests/helpers/codex_fake_server.py`,
  `tests/unit/providers/codex/test_codex.py`: offline fake-process protocol tests.
- `tests/integration/test_codex_subscription.py`: opt-in authenticated test.
- `tests/unit/cli/test_cli.py`: verify the CLI against the package version.
- `examples/codex_universrobot.py`: login, explicit model selection, validated
  text analysis and account limits from an installed official release.
- `docs/codex.md`, `docs/providers.md`, `docs/configuration.md`, `README.md`,
  `CHANGELOG.md`, and this record: prerequisites, limits and delivery notes.

## Validation

- Ruff check and format check: passed.
- Strict mypy: passed, 67 source files.
- `PYTHONPATH=src .venv/bin/pytest -q`: **131 passed, 3 skipped**.
  `PYTHONPATH=src` is required in this development environment because its venv
  contains an older installed CortexMux. It is not an UniversRobot dependency.
- `git diff --check`: passed.
- Real CLI 0.140.0: generated stable protocol JSON schemas; initialize,
  initialized, account/read with an empty dedicated home, and cleanup succeeded.
- Authenticated live generation/login: not run. The optional integration test
  requires explicit opt-in and an already connected dedicated account.
- `python -m build`: candidate wheel and source archive built successfully.
  The initial restricted-network attempt could not fetch Hatchling; the same
  isolated build succeeded with the required dependency-download access.
- Twine checks for both candidate distributions: passed. A blanket `dist/*`
  check also encounters pre-existing historical subdirectories, which Twine
  cannot treat as distributions; candidate files were checked explicitly.

Artifacts:

- `dist/cortexmux-0.6.0rc1-py3-none-any.whl`
- `dist/cortexmux-0.6.0rc1.tar.gz`

## Published candidate limitations

The protocol is pinned to CLI 0.140.0. Strict zero-tool isolation is unsupported
and fails closed by default; generation requires explicit acceptance of the
sandbox limitations. ChatGPT service eligibility and successful authenticated
end-to-end use remain to be confirmed with the opt-in test on the consuming
application's account. This is a release candidate, not a stable release.
UniversRobot must adopt an approved GitHub release through its existing
`Tools/External/CortexMux/` release process.
