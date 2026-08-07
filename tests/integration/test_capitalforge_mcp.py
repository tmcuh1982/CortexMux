"""Opt-in integration test against a real local CapitalForge MCP executable."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from cortexmux.mcp import MCPServerConfig, MCPStdioClient

pytestmark = pytest.mark.integration

_TOOLS = frozenset(
    {
        "capitalforge_source_catalog",
        "capitalforge_portfolio_summary",
        "capitalforge_positions",
        "capitalforge_zonebourse_signals",
        "capitalforge_public_signals",
    }
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("CORTEXMUX_RUN_CAPITALFORGE_INTEGRATION_TESTS") != "1",
    reason="set CORTEXMUX_RUN_CAPITALFORGE_INTEGRATION_TESTS=1 for local CapitalForge",
)
async def test_real_capitalforge_stdio_read_tools(tmp_path: Path) -> None:
    """Negotiate and invoke every read-only tool against an explicitly enabled local server."""
    command = os.environ.get(
        "CORTEXMUX_CAPITALFORGE_COMMAND",
        "/Users/lolo/Documents/DEV/CapitalForge/.venv/bin/capitalforge",
    )
    config = os.environ.get(
        "CORTEXMUX_CAPITALFORGE_CONFIG",
        "/Users/lolo/Documents/DEV/CapitalForge/configs/local/capitalforge.yaml",
    )
    database = os.environ.get(
        "CORTEXMUX_CAPITALFORGE_DATABASE",
        "/Users/lolo/Documents/DEV/CapitalForge/capitalforge.db",
    )
    # CapitalForge initializes SQLite's journal mode even for read operations;
    # isolate that write from the user's active portfolio database.
    copied_database = tmp_path / "capitalforge-integration.db"
    shutil.copy2(database, copied_database)
    client = MCPStdioClient(
        MCPServerConfig(
            enabled=True,
            command=command,
            arguments=["mcp", config, "--database", str(copied_database)],
        ),
        allowed_tools=_TOOLS,
    )
    try:
        assert {tool.name for tool in await client.list_tools()} == _TOOLS
        for name, arguments in {
            "capitalforge_source_catalog": {},
            "capitalforge_portfolio_summary": {},
            "capitalforge_positions": {"limit": 1},
            "capitalforge_zonebourse_signals": {},
            "capitalforge_public_signals": {"limit": 1},
        }.items():
            assert (await client.call_tool(name, arguments)).data
    finally:
        await client.close()
