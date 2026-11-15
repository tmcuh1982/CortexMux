# Markdown input and output formats

CortexMux accepts Markdown as prompt text and can export responses as JSON, YAML,
or Markdown. Structured generation still uses JSON schemas and the existing
validation path.

## Markdown files as input

```bash
cortexmux chat --model my-model --prompt-file prompt.md --format markdown > reply.md
cortexmux data analyze sales.csv --instruction-file analysis.md --format md > report.md
```

- `chat`, `generate`, `vision` and `image generate` accept `--prompt-file` in
  place of `--prompt`.
- `embed` accepts one or more `--text-file` options in place of `--text`.
- `data analyze` accepts `--instruction-file` in place of `--instruction`. The
  data source still uses a supported tabular format, such as CSV.

Input files must be local, contain nonblank UTF-8 text, and be at most 1 MiB each.
Markdown headings, lists, and code blocks are passed as text. CortexMux does not
resolve links or load files referenced in the document.

## Installation

YAML uses the optional PyYAML dependency:

```bash
pip install "cortexmux[yaml]"
```

Without this extra, JSON, Markdown, and text output remain available. Requesting YAML raises
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

Use `to_markdown()` without an extra dependency. Python prompt methods already
accept strings, so read a Markdown file and pass its contents as the prompt:

```python
from pathlib import Path

from cortexmux import CortexMux

prompt = Path("prompt.md").read_text(encoding="utf-8")
with CortexMux.from_env() as mux:
    response = mux.chat(prompt, provider="ollama", model="my-model")

Path("reply.md").write_text(response.to_markdown(), encoding="utf-8")
```

`to_markdown()` returns the generated text for chat, text generation, and vision,
the existing Markdown report for data analysis, and a fenced JSON block for
other responses. Text is preserved as returned by the model; ask for headings or
lists in the prompt if you want the model to use them. Use JSON or YAML if you
need all fields of a text response or data analysis.

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

`--format text|json|yaml|markdown` is available on the commands below. `md` is an
alias for `markdown`.

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

`--json` remains an alias for JSON output and can be combined with `--format json`.
Combining it with another format or specifying an unknown format produces a usage
error (exit code 2) before invoking a provider.

JSON and YAML exports contain the complete normalized response rather than only
the generated text. YAML and Markdown are written directly to stdout without
terminal formatting or image progress messages, so they can be redirected to
files. Configuration export keeps the existing secret redaction.
