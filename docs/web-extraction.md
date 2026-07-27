# Web extraction

CortexMux can retrieve a specific web page, extract its visible text and simple
HTML tables, then optionally ask a configured model for schema-constrained
information. This feature is disabled by default.

It is a direct-page extractor, not a search engine or a browser automation
system. It does not execute JavaScript, authenticate to sites, bypass paywalls,
or discover which page should be visited.

## Configuration

Enable web access and, preferably, restrict it to the exact public hosts needed
by the application:

```toml
[web]
enabled = true
allowed_hosts = ["data.example.org"]
allowed_ports = [443]
timeout_seconds = 15
max_response_bytes = 2000000
max_text_characters = 100000
max_redirects = 3
max_tables = 20
max_table_rows = 1000
```

An empty `allowed_hosts` accepts any host that resolves exclusively to public
IP addresses. Setting an explicit list is recommended for production.
Environment equivalents are available for the most common switches:

```bash
export CORTEXMUX_WEB_ENABLED=true
export CORTEXMUX_WEB_ALLOWED_HOSTS=data.example.org
```

## Deterministic retrieval

```python
from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig

config = CortexMuxConfig.model_validate(
    {
        "web": {
            "enabled": True,
            "allowed_hosts": ["data.example.org"],
        }
    }
)

with CortexMux(config) as mux:
    page = mux.fetch_web_page("https://data.example.org/population")
    print(page.title)
    print(page.text)

    if page.tables:
        records = page.tables[0].to_records()
        analysis = mux.analyze_data(records, instruction="Summarize the latest values.")
```

`WebPage` contains the requested and final URLs, HTTP status, content type,
visible text, extracted tables, byte count, and a truncation flag. Scripts,
styles, templates, SVG content, and `noscript` blocks are excluded.

## Structured extraction with local Ollama

Use `extract_web_page` when a model needs to locate a fact in the cleaned page:

```python
from pydantic import BaseModel


class PopulationFact(BaseModel):
    year: int
    population: int
    source_quote: str


with CortexMux(config) as mux:
    result = mux.extract_web_page(
        "https://data.example.org/population",
        "Extract the latest published population.",
        response_model=PopulationFact,
        provider="ollama",
        model="qwen2.5-coder:7b",
    )

    print(result.parsed)
    print(result.raw_metadata["web_source"])
```

Only bounded cleaned text is sent to the selected model. Page content is
wrapped as untrusted evidence and the model is instructed to ignore directions
embedded in the page. The final response includes source URL and truncation
metadata under `raw_metadata["web_source"]`.

## Network safeguards

- Every initial URL and redirect must use HTTP or HTTPS.
- URL credentials are rejected.
- Ports default to 80 and 443.
- Private, loopback, link-local, multicast, reserved, and otherwise non-public
  resolved addresses are rejected unless `allow_private_hosts=true`.
- Response bytes, extracted characters, redirects, tables, and table rows are
  bounded.
- Only HTML, XHTML, and plain text are accepted.

DNS resolution and the HTTP connection are separate operating-system
operations, so applications exposed to hostile tenants should also enforce
egress firewall or proxy rules. Enabling `allow_private_hosts` can expose local
services and should only be used in a trusted, isolated environment.
