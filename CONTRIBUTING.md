# Contributing

Use Python 3.11+ and install an editable development environment:

```bash
python -m pip install -e ".[dev,all]"
pre-commit install
```

Before a pull request, run `ruff check .`, `ruff format --check .`,
`mypy src/cortexmux`, and `pytest -m "not integration"`. Format with
`ruff format .`. Integration tests require `CORTEXMUX_RUN_INTEGRATION_TESTS=1`
and local services; never make CI depend on a GPU or installed model.

Keep provider behavior out of the router, lazily import optional packages, add
tests and a changelog entry for public API changes, and document security or
compatibility effects. Pull requests should be focused and explain validation.

