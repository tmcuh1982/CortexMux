# CortexMux

> One interface. Multiple models. Full control.

[![CI](https://github.com/example/cortexmux/actions/workflows/ci.yml/badge.svg)](https://github.com/example/cortexmux/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CortexMux is a modular, local-first Python library for routing AI tasks across
language models, vision models, ComfyUI image workflows, and deterministic
data-analysis engines. Version 0.5.1 is an alpha-quality stable release: its public
surface is tested, but production deployments should pin the patch version.

## Features

- Deterministic provider/model routing with no ambiguous fallback.
- Deterministic cross-provider model qualification with machine-specific
  JSON/YAML recommendation manifests.
- Native Ollama text, chat, JSON, vision, embedding, and incremental streaming.
- Explicit opt-in OpenAI Responses API text, chat, and structured output.
- Native ComfyUI workflow catalogs, typed progress, binding, queueing, and safe downloads.
- Safe Pandas analysis plus optional Polars and DuckDB loading.
- Whitelisted analysis plans and bounded structured results.
- Independent Decimal-based verification of structured AI calculation claims.
- Opt-in bounded web-page text and table extraction with source provenance.
- Opt-in read-only CapitalForge MCP context for native Ollama tool calls.
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
`yaml`, `all`, and `dev`. Optional packages are imported only when their feature
is used.

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

CapitalForge can remain a fully separate local application while Ollama reads
its bounded portfolio context through MCP stdio. The integration is disabled by
default and has no trading or configuration-write capability; see
[CapitalForge MCP](docs/capitalforge-mcp.md).

Structured output accepts `response_model=YourPydanticModel`. Vision accepts a
path, bytes, or validated base64. Embeddings accept one string or a list.

Ollama reasoning models such as Qwen3 can disable reasoning for reliable
schema-constrained JSON. CortexMux sends `think` at the `/api/generate` payload
root, separately from generation options:

```python
schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

with CortexMux.from_env() as mux:
    response = mux.structured(
        prompt="Return a concise answer.",
        json_schema=schema,
        provider="ollama",
        model="qwen3:4b",
        think=False,
    )
```

Structured output can also be streamed as an explicitly unvalidated draft. Do
not parse or use `StructuredStreamChunk.content` as application data. CortexMux
concatenates the fragments and emits `StructuredStreamCompleted` only after the
complete JSON has passed parsing and JSON Schema validation:

```python
import asyncio
import json

from cortexmux import CortexMux
from cortexmux.schemas import StructuredStreamChunk, StructuredStreamCompleted


async def main() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
    }
    draft = ""

    async with CortexMux.from_env() as mux:
        async for event in mux.astream_structured(
            prompt="Answer concisely and estimate confidence.",
            json_schema=schema,
            provider="ollama",
            model="ministral-3:8b",
            think=False,
        ):
            if isinstance(event, StructuredStreamChunk):
                draft += event.content
                print(f"\rDraft (unvalidated): {draft}", end="", flush=True)
            elif isinstance(event, StructuredStreamCompleted):
                print("\r" + " " * (len(draft) + 21), end="\r")
                print("Validated:", json.dumps(event.parsed, ensure_ascii=False))


asyncio.run(main())
```

For synchronous applications, iterate over `mux.stream_structured(...)` with
the same event types. A premature stream end raises
`StructuredStreamInterruptedError`; invalid final JSON or schema mismatch
raises `StructuredOutputValidationError`. Existing `structured()` and
`astructured()` calls remain non-streaming.

## ComfyUI

ComfyUI workflows must be exported in **API format**, not the browser save
format. Bind normalized fields to node inputs:

```python
result = mux.generate_image(
    prompt="An industrial sensor in a wheat field",
    workflow="text-to-image",
    checkpoint="model.safetensors",
    on_progress=lambda event: print(event.stage.value, event.progress),
)
```

The name resolves a `.cortexmux.json` manifest from the configured workflow
directory. Direct graph paths and inline graphs remain supported. Bindings are
strict by default and the original graph is never mutated. See [ComfyUI
workflows](docs/comfyui-workflows.md).

## Data analysis

```python
result = mux.analyze_data(
    extracted_records,
    instruction="Calcule l'évolution moyenne des prix.",
    engine="auto",
    interpretation_provider="ollama",
    interpretation_model="qwen2.5-coder:7b",
    require_verified_calculations=True,
)
print(result.results)
print(result.math_verification_passed)
```

`extracted_records` can be a dictionary or a list of dictionaries produced by
a website extraction layer. Without an interpretation model, analysis is
entirely deterministic. With local Ollama, only a redacted bounded sample,
schema, aggregates, and results are sent. Numerical claims are independently
recalculated from structured source references; the model never executes code
or SQL. See [data analysis](docs/data-analysis.md) and
[security](docs/data-analysis-security.md).

## Optional web extraction

Direct page retrieval is disabled by default. Enable it for specific hosts,
then extract visible text and simple HTML tables without giving the model
network access:

```python
from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig

config = CortexMuxConfig.model_validate(
    {"web": {"enabled": True, "allowed_hosts": ["data.example.org"]}}
)

with CortexMux(config) as mux:
    page = mux.fetch_web_page("https://data.example.org/statistics")
    records = page.tables[0].to_records()
```

For semantic extraction, `mux.extract_web_page(...)` sends bounded cleaned text
to a structured-output provider such as local Ollama
`qwen2.5-coder:7b` and preserves source metadata in the response. Initial URLs
and redirects are revalidated; private addresses, oversized responses, URL
credentials, unsupported ports, and non-text content are rejected. See
[web extraction](docs/web-extraction.md).

## CLI

```bash
cortexmux version
cortexmux doctor
cortexmux providers
cortexmux models list --provider ollama
cortexmux models qualify \
  --suite configs/model-qualification.example.json \
  --output outputs/model-qualification.json
cortexmux workflows list
cortexmux chat --provider ollama --model my-model --prompt "Hello"
cortexmux image generate --workflow text-to-image --prompt "A local workflow"
cortexmux data analyze sales.csv --engine auto
```

Commands that return structured values support `--json`.
See [model qualification](docs/model-qualification.md) for candidate-specific
options, scoring, OpenAI opt-in, and portable manifests.

## End-to-end local demonstration

The complete demo creates a CSV dataset, lists every provider model or data
engine known to CortexMux, runs deterministic analysis, uses a local
`qwen2.5-coder` Ollama tag for interpretation, and generates an image through
ComfyUI when both services are available:

```bash
python examples/full_local_demo.py
python examples/full_local_demo.py --strict
python examples/full_local_demo.py \
  --ollama-model qwen2.5-coder \
  --checkpoint my-checkpoint.safetensors
```

The untagged Ollama name resolves installed tags such as
`qwen2.5-coder:latest`. Without `--strict`, unavailable local services are
reported and the deterministic CSV analysis still runs.

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
search engine, JavaScript browser, vector database, model installer, GPU
manager, or distributed orchestrator. ComfyUI graph inputs must be explicitly
bound. Model capabilities are configuration/introspection driven, not guessed
from names.

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
