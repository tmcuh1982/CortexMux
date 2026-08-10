# CapitalForge MCP read-only integration

CortexMux can start a separate, local CapitalForge MCP server over **stdio**.
There is no Python import from CapitalForge: CortexMux knows only the MCP
protocol and five explicitly allowlisted read tools.

Enable it only in a local configuration you control:

```toml
[mcp.capitalforge]
enabled = true
command = "/Users/lolo/Documents/DEV/CapitalForge/.venv/bin/capitalforge"
arguments = [
  "mcp",
  "/Users/lolo/Documents/DEV/CapitalForge/configs/local/capitalforge.yaml",
  "--database",
  "/Users/lolo/Documents/DEV/CapitalForge/capitalforge.db",
]
timeout_seconds = 20
shutdown_timeout_seconds = 5
max_result_characters = 200000
```

The subprocess is started lazily for the first request and reaped by
`CortexMux.close()` / the context manager. The client proposes MCP protocol
`2025-11-25` and accepts an MCP 2025 protocol version selected by the Python
1.29 server. Its stdout is protocol data only and is never copied to logs.

## Ollama usage

```python
from cortexmux import CortexMux

with CortexMux.from_env(config_path="config.toml") as mux:
    answer = mux.capitalforge_chat(
        "Fais une analyse générale du portefeuille.",
        model="qwen3:4b",
        temperature=0.1,
    )
    print(answer.content)
```

For a general analysis, the system instruction directs the model to read the
source catalog and portfolio summary first, then request only relevant sources.
It responds in the user's language, distinguishes local facts from external
signals and inferences, cites source/date/URL when supplied, and flags missing,
delayed, or partial inputs. A public signal is context, never advice or an
order. A sale of an asset absent from the local portfolio may be ignored, while
an external purchase or reinforcement must be surfaced.

## Safety boundary

Only these tools can cross the boundary:

- `capitalforge_source_catalog()`
- `capitalforge_portfolio_summary()`
- `capitalforge_positions(limit: 1..200)`
- `capitalforge_zonebourse_signals(market?: "usa" | "europe")`
- `capitalforge_public_signals(plugin_id?: string, limit: 1..100)`

The client requires all five tools to be advertised, rejects every other name,
validates tool arguments locally, accepts only bounded JSON-object structured
results, and rejects fields whose names indicate a secret. It starts no shell,
passes only a minimal `PATH` and C locale to the child, and never forwards
conversation history to MCP: the server receives only a selected tool name and
validated arguments. A request permits at most eight calls and rejects repeated
tool calls to stop loops.

There is intentionally no API for trading, orders, imports, YAML writes, or
candidate activation. Any configuration proposal stays text/structured data and
must still be accepted by CapitalForge's own validator; CortexMux never applies
it.

## Disable immediately

Set `enabled = false` under `[mcp.capitalforge]`, or remove that section, then
restart the process. Existing `CortexMux` contexts terminate their child process
when closed. No MCP connection is opened when the integration is disabled.

Run the real local integration test only with explicit opt-in:

```bash
CORTEXMUX_RUN_CAPITALFORGE_INTEGRATION_TESTS=1 \
  PYTHONPATH=src .venv/bin/pytest -q tests/integration/test_capitalforge_mcp.py
```
