"""Bounded JSONL App Server transport with a single reader and no retries."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from cortexmux.core.exceptions import CortexMuxError
from cortexmux.providers.codex.errors import CodexError, server_error
from cortexmux.providers.codex.schemas import CodexConfig
from cortexmux.version import __version__

# Audited against locally generated stable schemas. New versions require review.
SUPPORTED_VERSION = "0.140.0"
_OVERRIDES = {
    "forced_login_method": '"chatgpt"',
    "cli_auth_credentials_store": '"file"',
    "model_provider": '"openai"',
    "approval_policy": '"never"',
    "approvals_reviewer": '"user"',
    "sandbox_mode": '"read-only"',
    "web_search": '"disabled"',
    "project_doc_max_bytes": "0",
    "features.shell_tool": "false",
    "features.unified_exec": "false",
    "features.shell_snapshot": "false",
    "features.apps": "false",
    "features.hooks": "false",
    "features.multi_agent": "false",
    "features.memories": "false",
    "features.remote_plugin": "false",
    "mcp_servers": "{}",
}


class CodexClient:
    """Own one lazy process, isolated working directory, and correlated requests."""

    def __init__(self, config: CodexConfig) -> None:
        self.config = config
        self.version: str | None = None
        self.cwd: str | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._subscribers: set[asyncio.Queue[dict[str, Any] | CortexMuxError]] = set()
        self._sequence = 0
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._closed = False
        self._failure: CortexMuxError | None = None

    async def start(self) -> None:
        """Check the pinned version and initialize once without reading user auth."""
        async with self._start_lock:
            if self._closed:
                raise CodexError("closed")
            if self._failure:
                raise self._failure
            if self._process is not None:
                return
            executable = shutil.which(self.config.executable)
            if executable is None:
                raise CodexError("executable_missing")
            auth = self.config.auth_directory.expanduser().resolve()
            personal = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
            if auth == personal or auth == (Path.home() / ".codex").resolve():
                raise CodexError("shared_auth_directory_refused")
            if (auth / "auth.json").is_symlink():
                raise CodexError("shared_auth_directory_refused")
            # This directory belongs exclusively to the integration. Never read auth.json.
            if any(
                (auth / name).exists()
                for name in ("config.toml", "AGENTS.md", "plugins", "hooks.json")
            ):
                raise CodexError("personal_configuration_refused")
            skills = auth / "skills"
            if skills.exists() and any(entry.name != ".system" for entry in skills.iterdir()):
                raise CodexError("personal_configuration_refused")
            auth.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._directory = tempfile.TemporaryDirectory(prefix="cortexmux-codex-")
            self.cwd = self._directory.name
            # Do not inherit API keys, endpoint overrides, MCP, or workload identity.
            env = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "USER", "SYSTEMROOT", "WINDIR", "LANG", "TMPDIR")
                if key in os.environ
            }
            env["CODEX_HOME"] = str(auth)
            try:
                probe = await asyncio.create_subprocess_exec(
                    executable,
                    "--version",
                    cwd=self.cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    async with asyncio.timeout(self.config.startup_timeout_seconds):
                        assert probe.stdout is not None
                        version_line = await probe.stdout.read(1024)
                        await probe.wait()
                finally:
                    if probe.returncode is None:
                        probe.kill()
                        await probe.wait()
                if (
                    probe.returncode != 0
                    or re.fullmatch(rb"codex-cli 0\.140\.0\s*", version_line) is None
                ):
                    raise CodexError("incompatible_version")
                self.version = SUPPORTED_VERSION
                args = [executable, "app-server", "--listen", "stdio://"]
                for key, value in _OVERRIDES.items():
                    args.extend(["-c", f"{key}={value}"])
                self._process = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self.cwd,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=self.config.max_message_bytes,
                )
                self._reader = asyncio.create_task(self._read())
                await self._request(
                    "initialize",
                    {"clientInfo": {"name": "cortexmux", "version": __version__}},
                    self.config.startup_timeout_seconds,
                )
                await self._write({"method": "initialized", "params": {}})
            except BaseException as exc:
                await self.close()
                if isinstance(exc, TimeoutError):
                    raise CodexError("timeout") from None
                if isinstance(exc, OSError):
                    raise CodexError("executable_unavailable") from None
                raise

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one correlated request; never retry an ambiguous operation."""
        await self.start()
        return await self._request(method, params or {}, timeout or self.config.timeout_seconds)

    async def _request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self._failure:
            raise self._failure
        self._sequence += 1
        request_id = self._sequence
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with asyncio.timeout(timeout):
                await self._write({"id": request_id, "method": method, "params": params})
                return await future
        except TimeoutError:
            raise CodexError("timeout") from None
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def _write(self, message: dict[str, Any]) -> None:
        data = (json.dumps(message) + "\n").encode()
        if len(data) > self.config.max_message_bytes:
            raise CodexError("message_limit")
        async with self._write_lock:
            if self._process is None or self._process.stdin is None:
                raise CodexError("closed")
            try:
                self._process.stdin.write(data)
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionError):
                raise CodexError("process_exited") from None

    @contextlib.asynccontextmanager
    async def events(self) -> AsyncIterator[asyncio.Queue[dict[str, Any] | CortexMuxError]]:
        """Subscribe before submitting work to retain early notifications."""
        queue: asyncio.Queue[dict[str, Any] | CortexMuxError] = asyncio.Queue(
            self.config.max_queued_events
        )
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    def _publish(self, event: dict[str, Any] | CortexMuxError) -> None:
        for queue in self._subscribers:
            if queue.full():
                raise CodexError("event_limit")
            queue.put_nowait(event)

    async def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        failure: CortexMuxError = CodexError("process_exited")
        try:
            while line := await self._process.stdout.readline():
                if len(line) > self.config.max_message_bytes:
                    raise CodexError("message_limit")
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise CodexError("invalid_response")
                if "method" in message:
                    if not isinstance(message["method"], str) or not isinstance(
                        message.get("params", {}), dict
                    ):
                        raise CodexError("invalid_response")
                    if "id" in message:
                        # Every server-initiated request is rejected, never approved.
                        await self._write(
                            {
                                "id": message["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Permission denied.",
                                },
                            }
                        )
                        self._publish(
                            {
                                "method": "cortexmux/permissionDenied",
                                "params": message.get("params", {}),
                            }
                        )
                    else:
                        self._publish(message)
                else:
                    response_id = message.get("id")
                    future = (
                        self._pending.get(response_id) if isinstance(response_id, int) else None
                    )
                    if future is None or future.done():
                        continue
                    if isinstance(message.get("error"), dict):
                        future.set_exception(server_error(message["error"]))
                    elif isinstance(message.get("result"), dict):
                        future.set_result(message["result"])
                    else:
                        future.set_exception(CodexError("invalid_response"))
        except asyncio.CancelledError:
            return
        except CortexMuxError as exc:
            failure = exc
        except (ValueError, TypeError, OSError):
            failure = CodexError("invalid_response")
        finally:
            self._failure = failure
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(failure)
            for queue in self._subscribers:
                while queue.full():
                    queue.get_nowait()
                queue.put_nowait(failure)
            if self._process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._process.terminate()

    async def close(self) -> None:
        """Stop the process with bounded terminate/kill escalation and reap it."""
        self._closed = True
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), self.config.shutdown_timeout_seconds)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
