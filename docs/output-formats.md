# JSON and YAML output

CortexMux responses remain typed Pydantic models. YAML is an export format:
structured generation still uses JSON schemas and the existing validation path.
Providers and routing are unchanged.

## Installation

YAML uses the optional PyYAML dependency:

```bash
pip install "cortexmux[yaml]"
```

Without this extra, JSON and text output remain available. Requesting YAML raises
`OptionalDependencyError` with the installation command (CLI exit code 4).

## Python responses

Every response derived from `CortexResponse` exposes `to_yaml()`, including text,
chat, structured output, vision, embeddings, images, decisions and data analysis:

```python
from pathlib import Path

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    response = mux.chat("Bonjour", provider="ollama", model="my-model")

Path("response.yaml").write_text(response.to_yaml(), encoding="utf-8")
```

`to_yaml()` exports the entire response, including metadata and, for structured
responses, `parsed`. It uses `model_dump(mode="json")` followed by
`yaml.safe_dump`: enums, paths, dates and decimals use Pydantic's JSON-compatible
representation, secrets retain their Pydantic redaction, and Unicode remains
readable. The response is not modified. No Python-specific YAML object tags are
emitted.

To export only a structured response's validated data:

```python
import yaml
from pydantic import BaseModel

# response is the result of mux.structured(...) or await mux.astructured(...).
data = response.parsed
if isinstance(data, BaseModel):
    data = data.model_dump(mode="json")

output = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
```

Streaming fragments remain unvalidated text. Export the final validated response
or the completed event's normalized data when the stream finishes.

## CLI formats

`--format text|json|yaml` is available on:

- `providers`, `doctor`, `chat`, `generate`, `vision`, `embed`;
- `models list`, `models qualify`;
- `image generate`, `data analyze`;
- `config show`, `workflows list`, `workflows show`.

```bash
cortexmux chat --model my-model --prompt "Bonjour" --format yaml > response.yaml
cortexmux data analyze sales.csv --format yaml > analysis.yaml
cortexmux config show --format yaml
cortexmux models list --provider ollama --format json
```

Existing defaults are preserved: text display for these commands, except
`models qualify`, whose stdout summary defaults to JSON. The manifest written by
`models qualify --output` continues to use the file extension (`.json`, `.yaml`,
`.yml`), independently of `--format`.

`--json` remains an alias for JSON output. Combining it with `--format json` is
allowed; combining it with `--format text` or `--format yaml` produces a usage
error (exit code 2) before invoking a provider. Unknown formats are also usage
errors.

JSON and YAML exports contain the complete normalized response rather than only
the generated text. YAML is written directly to stdout without terminal markup,
coloring, or image progress messages, so it can be redirected to a file.
Configuration export keeps the existing secret redaction.
