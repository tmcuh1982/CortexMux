"""A bounded JSON-RPC MCP client over one local stdio subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from cortexmux.core.exceptions import (
    ConfigurationError,
    MCPProtocolError,
    MCPToolNotAllowedError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from cortexmux.mcp.schemas import MCPServerConfig, MCPTool, MCPToolResult

_PROTOCOL_VERSION = "2025-11-25"
_MAX_PROTOCOL_LINE_BYTES = 1_000_000
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|credential|password|secret|token|private[_-]?key)", re.I
)


class MCPStdioClient:
    """Use a local read-only MCP server without exposing process environment data.

    The client deliberately reads stdout only as JSON-RPC protocol messages and
    never writes it to logs.  It launches no shell and gives the subprocess a
    minimal environment, so host credentials are not inherited.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        allowed_tools: frozenset[str],
        client_name: str = "cortexmux",
    ) -> None:
        self.config = config
        self._allowed_tools = allowed_tools
        self._client_name = client_name
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._tools: dict[str, MCPTool] | None = None
        self.protocol_version: str | None = None

    async def list_tools(self) -> list[MCPTool]:
        """Negotiate the session and return the allowlisted tool catalog."""
        async with self._lock:
            await self._ensure_started()
            response = await self._request("tools/list", {})
            raw_tools = response.get("tools")
            if not isinstance(raw_tools, list):
                raise MCPProtocolError("MCP tools/list returned an invalid tool catalog.")
            tools: dict[str, MCPTool] = {}
            for raw in raw_tools:
                tool = _parse_tool(raw)
                if tool.name in self._allowed_tools:
                    tools[tool.name] = tool
            missing = self._allowed_tools - tools.keys()
            if missing:
                raise MCPToolNotAllowedError(
                    "CapitalForge MCP server did not expose every required read-only tool.",
                    missing_tools=sorted(missing),
                )
            self._tools = tools
            return list(tools.values())

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> MCPToolResult:
        """Call one allowlisted tool and accept only a bounded JSON object result."""
        if name not in self._allowed_tools:
            raise MCPToolNotAllowedError("MCP tool is not allowed.", tool=name)
        normalized_arguments = _validate_arguments(name, arguments or {})
        async with self._lock:
            await self._ensure_started()
            if self._tools is None:
                await self._list_tools_locked()
            response = await self._request(
                "tools/call", {"name": name, "arguments": normalized_arguments}
            )
        if response.get("isError") is True:
            raise ProviderResponseError("CapitalForge MCP tool returned an error.", tool=name)
        data = response.get("structuredContent")
        if not isinstance(data, dict):
            raise MCPProtocolError("MCP tool result has no structured object content.", tool=name)
        _validate_safe_result(data, max_characters=self.config.max_result_characters)
        return MCPToolResult(tool_name=name, data=data)

    async def _list_tools_locked(self) -> None:
        """Populate the cached catalog while the process lock is held."""
        response = await self._request("tools/list", {})
        raw_tools = response.get("tools")
        if not isinstance(raw_tools, list):
            raise MCPProtocolError("MCP tools/list returned an invalid tool catalog.")
        tools: dict[str, MCPTool] = {}
        for item in raw_tools:
            tool = _parse_tool(item)
            tools[tool.name] = tool
        available = {name: tool for name, tool in tools.items() if name in self._allowed_tools}
        missing = self._allowed_tools - available.keys()
        if missing:
            raise MCPToolNotAllowedError(
                "CapitalForge MCP server did not expose every required read-only tool.",
                missing_tools=sorted(missing),
            )
        self._tools = available

    async def _ensure_started(self) -> None:
        """Start and initialize the local process once."""
        if self._process is not None and self._process.returncode is None:
            return
        command = self.config.command
        if not self.config.enabled or command is None:
            raise ConfigurationError("CapitalForge MCP is disabled or has no command configured.")
        self._process = await asyncio.create_subprocess_exec(
            command,
            *self.config.arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=_minimal_environment(),
            limit=_MAX_PROTOCOL_LINE_BYTES,
        )
        self._request_id = 0
        try:
            response = await self._request(
                "initialize",
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self._client_name, "version": "0.5.2"},
                },
            )
            version = response.get("protocolVersion")
            if not isinstance(version, str) or not version.startswith("202"):
                raise MCPProtocolError("MCP server selected an unsupported protocol version.")
            self.protocol_version = version
            await self._notify("notifications/initialized")
        except BaseException:
            await self.close()
            raise

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON-RPC request and wait only for its matching response."""
        process = self._require_process()
        self._request_id += 1
        request_id = self._request_id
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        try:
            while True:
                line = await asyncio.wait_for(
                    self._read_protocol_line(process), timeout=self.config.timeout_seconds
                )
                message = _decode_message(line)
                if message.get("id") != request_id:
                    continue
                error = message.get("error")
                if isinstance(error, dict):
                    raise MCPProtocolError("MCP server rejected a request.", method=method)
                result = message.get("result")
                if not isinstance(result, dict):
                    raise MCPProtocolError(
                        "MCP server returned an invalid response object.", method=method
                    )
                return result
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                "CapitalForge MCP request timed out.", method=method
            ) from exc

    async def _notify(self, method: str) -> None:
        """Send a JSON-RPC notification without waiting for output."""
        await self._write({"jsonrpc": "2.0", "method": method, "params": {}})

    async def _write(self, payload: dict[str, Any]) -> None:
        """Write one compact protocol line; no request contents are logged."""
        process = self._require_process()
        if process.stdin is None:
            raise MCPProtocolError("MCP server stdin is unavailable.")
        process.stdin.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        try:
            await asyncio.wait_for(process.stdin.drain(), timeout=self.config.timeout_seconds)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise MCPProtocolError("CapitalForge MCP process closed its input.") from exc

    async def _read_protocol_line(self, process: asyncio.subprocess.Process) -> bytes:
        """Read one protocol line without ever copying stdout to logs."""
        if process.stdout is None:
            raise MCPProtocolError("MCP server stdout is unavailable.")
        try:
            line = await process.stdout.readline()
        except asyncio.LimitOverrunError as exc:
            raise MCPProtocolError("MCP server emitted an oversized protocol message.") from exc
        if not line:
            raise MCPProtocolError("CapitalForge MCP process ended unexpectedly.")
        if len(line) > _MAX_PROTOCOL_LINE_BYTES:
            raise MCPProtocolError("MCP server emitted an oversized protocol message.")
        return line

    def _require_process(self) -> asyncio.subprocess.Process:
        """Return the active subprocess or fail without leaking process output."""
        if self._process is None or self._process.returncode is not None:
            raise MCPProtocolError("CapitalForge MCP process is not running.")
        return self._process

    async def close(self) -> None:
        """Terminate the subprocess and guarantee reaping it within the configured bound."""
        process, self._process = self._process, None
        self._tools = None
        self.protocol_version = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.config.shutdown_timeout_seconds)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()


def _parse_tool(raw: object) -> MCPTool:
    """Validate the small safe subset of a server-advertised tool definition."""
    if not isinstance(raw, dict):
        raise MCPProtocolError("MCP tool catalog contains a non-object item.")
    name = raw.get("name")
    description = raw.get("description")
    input_schema = raw.get("inputSchema", raw.get("input_schema", {}))
    if (
        not isinstance(name, str)
        or not isinstance(description, str)
        or not isinstance(input_schema, dict)
    ):
        raise MCPProtocolError("MCP tool catalog contains an invalid tool definition.")
    try:
        return MCPTool(name=name, description=description, input_schema=input_schema)
    except ValueError as exc:
        raise MCPProtocolError("MCP server advertised an unsafe tool definition.") from exc


def _decode_message(line: bytes) -> dict[str, Any]:
    """Decode one JSON-RPC message while discarding all malformed payload text."""
    try:
        message = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MCPProtocolError("MCP server emitted malformed JSON-RPC.") from exc
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise MCPProtocolError("MCP server emitted an invalid JSON-RPC message.")
    return message


def _minimal_environment() -> dict[str, str]:
    """Return only locale and executable lookup settings needed by a local server."""
    result = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    return result


def _validate_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce CapitalForge's five bounded read-only query contracts."""
    value = dict(arguments)
    allowed: dict[str, set[str]] = {
        "capitalforge_source_catalog": set(),
        "capitalforge_portfolio_summary": set(),
        "capitalforge_positions": {"limit"},
        "capitalforge_zonebourse_signals": {"market"},
        "capitalforge_public_signals": {"plugin_id", "limit"},
    }
    if name not in allowed or set(value) - allowed[name]:
        raise MCPToolNotAllowedError("MCP tool arguments are not allowed.", tool=name)
    if name == "capitalforge_positions":
        _validate_limit(value, maximum=200)
    if name == "capitalforge_public_signals":
        _validate_limit(value, maximum=100)
        plugin_id = value.get("plugin_id")
        if plugin_id is not None and (not isinstance(plugin_id, str) or not plugin_id.strip()):
            raise MCPToolNotAllowedError("plugin_id must be a non-empty string.", tool=name)
    if name == "capitalforge_zonebourse_signals":
        market = value.get("market")
        if market is not None and market not in {"usa", "europe"}:
            raise MCPToolNotAllowedError("market must be 'usa' or 'europe'.", tool=name)
    return value


def _validate_limit(arguments: dict[str, Any], *, maximum: int) -> None:
    """Validate an optional positive bounded integer argument."""
    limit = arguments.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= maximum
    ):
        raise MCPToolNotAllowedError("limit is outside the allowed range.")


def _validate_safe_result(data: dict[str, Any], *, max_characters: int) -> None:
    """Reject secrets, non-JSON values, and unbounded result payloads."""
    try:
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MCPProtocolError("MCP tool result is not JSON-compatible.") from exc
    if len(encoded) > max_characters:
        raise MCPProtocolError("MCP tool result exceeds the configured size limit.")
    _reject_sensitive_keys(data)


def _reject_sensitive_keys(value: object) -> None:
    """Prevent an unexpected server result from reaching an Ollama prompt."""
    if isinstance(value, dict):
        for key, child in value.items():
            # A server may safely report capability flags such as
            # ``broker_credentials_exposed: false``. Any actual value is
            # rejected before it can become model context.
            if (
                isinstance(key, str)
                and _SENSITIVE_KEY.search(key)
                and child is not False
                and child is not None
            ):
                raise MCPProtocolError("MCP tool result contains a sensitive field.")
            _reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child)
