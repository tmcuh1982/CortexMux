# CortexMux

> One interface. Multiple models. Full control.

[![CI](https://github.com/example/cortexmux/actions/workflows/ci.yml/badge.svg)](https://github.com/example/cortexmux/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CortexMux is a modular, local-first Python library for routing AI tasks across
language models, vision models, ComfyUI image workflows, and deterministic
data-analysis engines. Version 0.1.0 is an alpha-quality stable MVP: its public
surface is tested, but production deployments should pin the patch version.

## Features

- Deterministic provider/model routing with no ambiguous fallback.
- Native Ollama text, chat, JSON, vision, embedding, and incremental streaming.
- Native ComfyUI API-workflow binding, queueing, monitoring, and safe downloads.
- Safe Pandas analysis plus optional Polars and DuckDB loading.
- Whitelisted analysis plans and deterministic Matplotlib charts.
- Typed Pydantic schemas, synchronous/asynchronous APIs, Typer CLI, and TOML/env config.
- Loopback-only network policy by default.

## Architecture

Applications call the `CortexMux` facade, which validates a typed request. The
router selects one provider from the instance registry, a provider-specific
client performs the work, and the result is normalized. Data analysis follows a
separate validate/profile/plan/deterministic-execute/report pipeline. See
[architecture](docs/architecture.md).

## Requirements and installation

Python 3.11 or newer is required. Ollama and ComfyUI are external applications:
CortexMux does **not** install them, bundle model weights, or download models.
Availability and output quality depend on the models and workflows installed by
the user.

```bash
python -m pip install cortexmux
python -m pip install "cortexmux[data,visualization]"
python -m pip install "cortexmux[all]"
```

Extras are `comfyui`, `data`, `polars`, `duckdb`, `excel`, `visualization`,
`all`, and `dev`. Optional packages are imported only when their feature is used.

## Quick start

```python
from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    response = mux.chat(
        prompt="Explain how a solar inverter works.",
        provider="ollama",
        model="my-local-model",
    )
    print(response.content)
```

Use `async with CortexMux.from_env()` and `await mux.achat(...)` in asynchronous
programs. Calling a synchronous method from a running event loop is rejected.

Structured output accepts `response_model=YourPydanticModel`. Vision accepts a
path, bytes, or validated base64. Embeddings accept one string or a list.

## ComfyUI

ComfyUI workflows must be exported in **API format**, not the browser save
format. Bind normalized fields to node inputs:

```python
result = mux.generate_image(
    prompt="An industrial sensor in a wheat field",
    workflow="workflows/text-to-image.json",
    bindings={"prompt": {"node_id": "6", "input": "text"}},
    expected_output_nodes=["9"],
)
```

Bindings are strict by default and the original graph is never mutated. See
[ComfyUI workflows](docs/comfyui-workflows.md).

## Data analysis

```python
result = mux.analyze_data(
    "sales.csv",
    instruction="Analyze monthly revenue and unusual changes.",
    engine="auto",
)
print(result.report_markdown)
```

Without an interpretation model, this is entirely deterministic. If local
Ollama interpretation is configured, only a redacted bounded sample, schema,
aggregates, and results are sent. The model never executes code or SQL. See
[data analysis](docs/data-analysis.md) and [security](docs/data-analysis-security.md).

## CLI

```bash
cortexmux version
cortexmux doctor
cortexmux providers
cortexmux models list --provider ollama
cortexmux chat --provider ollama --model my-model --prompt "Hello"
cortexmux data analyze sales.csv --engine auto
```

Commands that return structured values support `--json`.

## Configuration

Precedence is explicit Python overrides, environment, TOML, then built-ins.
The default config is the platform user-config path; override it with
`CORTEXMUX_CONFIG`. See [configuration](docs/configuration.md) and
[the example](configs/cortexmux.example.toml).

## Security model

Provider URLs are resolved and must be loopback/localhost unless remote access
is explicitly enabled. Prompts, responses, secrets, images, and datasets are
not logged at INFO. Output paths are contained under a configured root and are
never overwritten. Enabling remote hosts can transmit prompts or summaries
outside the machine; review the remote service's policy first. See [SECURITY](SECURITY.md).

## v0.1 limitations

CortexMux is a library and CLI, not an agent platform, GUI, web server, SaaS,
vector database, model installer, GPU manager, or distributed orchestrator.
ComfyUI graph inputs must be explicitly bound. Model capabilities are
configuration/introspection driven, not guessed from names.

## Development

```bash
python -m pip install -e ".[dev,all]"
ruff check .
ruff format --check .
mypy src/cortexmux
pytest -m "not integration" --cov=src/cortexmux --cov-report=term-missing
python -m build
twine check dist/*
```

See [CONTRIBUTING](CONTRIBUTING.md), [ROADMAP](ROADMAP.md), and
[CHANGELOG](CHANGELOG.md). CortexMux is available under the [MIT License](LICENSE).

