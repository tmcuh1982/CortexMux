"""MCP stdio lifecycle and CapitalForge allowlist tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from cortexmux.core.exceptions import MCPProtocolError, MCPToolNotAllowedError, ProviderTimeoutError
from cortexmux.mcp import MCPServerConfig, MCPStdioClient

TOOLS = frozenset(
    {
        "capitalforge_source_catalog",
        "capitalforge_portfolio_summary",
        "capitalforge_positions",
        "capitalforge_zonebourse_signals",
        "capitalforge_public_signals",
    }
)
FAKE_SERVER = Path(__file__).parents[2] / "helpers" / "mcp_fake_server.py"


def client_for(*, mode: str = "normal", timeout: float = 1) -> MCPStdioClient:
    """Create one test client backed by the deterministic stdio fake."""
    return MCPStdioClient(
        MCPServerConfig(
            enabled=True,
            command=sys.executable,
            arguments=[str(FAKE_SERVER), "--mode", mode],
            timeout_seconds=timeout,
            shutdown_timeout_seconds=0.2,
        ),
        allowed_tools=TOOLS,
    )


@pytest.mark.asyncio
async def test_stdio_negotiates_downgrade_lists_and_calls_each_read_tool() -> None:
    """Accept a 2025 MCP downgrade and validate every CapitalForge result."""
    client = client_for()
    try:
        assert {tool.name for tool in await client.list_tools()} == TOOLS
        assert client.protocol_version == "2025-06-18"
        calls = {
            "capitalforge_source_catalog": {},
            "capitalforge_portfolio_summary": {},
            "capitalforge_positions": {"limit": 2},
            "capitalforge_zonebourse_signals": {"market": "usa"},
            "capitalforge_public_signals": {"limit": 2},
        }
        for name, arguments in calls.items():
            result = await client.call_tool(name, arguments)
            assert result.tool_name == name
            assert result.data["tool"] == name
            assert result.data["arguments"] == arguments
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_allowlist_argument_bounds_and_no_environment_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unapproved capabilities and launch the process without parent secrets."""
    monkeypatch.setenv("CORTEXMUX_TEST_SECRET", "must-not-reach-mcp")
    client = client_for(mode="environment")
    try:
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("capitalforge_place_order")
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("capitalforge_positions", {"limit": 201})
        with pytest.raises(MCPToolNotAllowedError):
            await client.call_tool("capitalforge_zonebourse_signals", {"market": "asia"})
        result = await client.call_tool("capitalforge_source_catalog")
        assert result.data["environment_was_minimal"] is True
        assert "must-not-reach-mcp" not in repr(result)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_timeout_terminates_the_stdio_subprocess() -> None:
    """Bound a stalled server call and reap its child process during shutdown."""
    client = client_for(mode="timeout", timeout=0.5)
    try:
        with pytest.raises(ProviderTimeoutError):
            await client.call_tool("capitalforge_source_catalog")
        process = client._process
        assert process is not None
    finally:
        await client.close()
    assert process.returncode is not None
    assert process.returncode != 0
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_secret_result_is_rejected_without_echoing_its_value() -> None:
    """Prevent a malicious server field from becoming prompt or diagnostic content."""
    client = client_for(mode="secret_result")
    try:
        with pytest.raises(MCPProtocolError) as error:
            await client.call_tool("capitalforge_source_catalog")
        assert "must-not-reach-ollama" not in str(error.value)
        assert "must-not-reach-ollama" not in repr(error.value.context)
    finally:
        await client.close()
