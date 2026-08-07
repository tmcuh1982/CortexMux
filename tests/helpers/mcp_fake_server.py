"""Small deterministic JSON-RPC stdio server used by CortexMux MCP unit tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

TOOLS = (
    "capitalforge_source_catalog",
    "capitalforge_portfolio_summary",
    "capitalforge_positions",
    "capitalforge_zonebourse_signals",
    "capitalforge_public_signals",
)


def send(payload: dict[str, Any]) -> None:
    """Emit one JSON-RPC message to the protocol stream."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Serve a minimal compatible subset of MCP's stdio transport."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices={"normal", "timeout", "environment", "secret_result"}, default="normal"
    )
    mode = parser.parse_args().mode
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        request_id = request.get("id")
        if method == "notifications/initialized":
            continue
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-capitalforge", "version": "1"},
                    },
                }
            )
        elif method == "tools/list":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": name,
                                "description": f"Read-only {name}",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                            for name in TOOLS
                        ]
                    },
                }
            )
        elif method == "tools/call":
            if mode == "timeout":
                time.sleep(3)
                continue
            params = request["params"]
            data: dict[str, Any] = {
                "tool": params["name"],
                "as_of": "2026-08-07",
                "url": "https://example.invalid/source",
                "arguments": params.get("arguments", {}),
            }
            if mode == "environment":
                data["environment_was_minimal"] = "CORTEXMUX_TEST_SECRET" not in os.environ
            if mode == "secret_result":
                data["access_token"] = "must-not-reach-ollama"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"structuredContent": data, "content": []},
                }
            )


if __name__ == "__main__":
    main()
