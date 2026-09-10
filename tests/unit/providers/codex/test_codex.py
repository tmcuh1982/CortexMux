"""Offline protocol, account, lifecycle, isolation, and facade regression tests."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.providers.codex import CodexClient, CodexConfig, CodexError, CodexProvider
from cortexmux.schemas.requests import StructuredOutputRequest, TextGenerationRequest
from cortexmux.schemas.responses import StreamChunk, StructuredStreamCompleted


@pytest.fixture
def settings(tmp_path):
    executable = tmp_path / "fake-codex"
    helper = Path(__file__).resolve().parents[3] / "helpers" / "codex_fake_server.py"
    executable.write_text(f"#!{sys.executable}\n" + helper.read_text())
    executable.chmod(0o700)
    return CodexConfig(
        enabled=True,
        executable=str(executable),
        auth_directory=tmp_path / "auth",
        timeout_seconds=2,
        shutdown_timeout_seconds=0.2,
    )


def request(prompt="normal", **options):
    return TextGenerationRequest(
        prompt=prompt,
        provider="codex",
        model="test-model",
        options={"require_no_tools": False, **options},
    )


def messages(settings):
    return [
        json.loads(line)
        for line in (settings.auth_directory / "fake-requests.jsonl").read_text().splitlines()
    ]


async def test_interleaved_stream_and_fresh_context(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "SECRET")
    async with CodexProvider(CodexClient(settings)) as provider:
        events = [event async for event in provider.stream(request())]
        assert [e.content for e in events] == ["answer", "answer"]
        assert events[-1].done
        first = events[-1].raw_metadata["thread_id"]
        result = await provider.execute(request())
        assert result.raw_metadata["thread_id"] != first
        assert result.model == "test-model"
        assert "PRIVATE" not in str(events)
    calls = messages(settings)
    assert len([m for m in calls if m.get("method") == "initialize"]) == 1
    starts = [m["params"] for m in calls if m.get("method") == "thread/start"]
    assert all(s["ephemeral"] and s["sandbox"] == "read-only" for s in starts)


async def test_account_login_limits_and_no_secrets(settings):
    async with CodexProvider(CodexClient(settings)) as provider:
        assert (await provider.healthcheck()).available
        assert "SECRET" not in (await provider.account()).model_dump_json()
        for device in (False, True):
            login = await provider.login_start(device_code=device)
            assert (await provider.login_result(login.login_id)).success
            await provider.login_cancel(login.login_id)
        limits = await provider.rate_limits()
        assert limits.rate_limits.primary.used_percent == 25
        assert limits.rate_limits.primary.resets_at is None
        assert limits.by_limit_id is None
        assert "SECRET" not in limits.model_dump_json()
        await provider.logout()
        assert not (await provider.account()).connected
        with pytest.raises(CodexError, match="connection_required"):
            await provider.execute(request())
    assert not any(m.get("method") == "turn/start" for m in messages(settings))


async def test_structured_validation_and_stream(settings):
    async with CodexProvider(CodexClient(settings)) as provider:
        req = StructuredOutputRequest(
            prompt="normal",
            model="test-model",
            options={"require_no_tools": False},
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
            },
        )
        events = [e async for e in provider.stream(req)]
        assert isinstance(events[-1], StructuredStreamCompleted)
        assert events[-1].parsed == {"answer": 42}
        assert events[-1].raw_metadata["turn_id"]
        with pytest.raises(StructuredOutputValidationError):
            await provider.execute(req.model_copy(update={"prompt": "bad-json"}))


@pytest.mark.parametrize(
    "prompt,reason",
    [
        ("empty", "invalid_response"),
        ("interrupted", "cancelled"),
        ("permission", "permission_required"),
        ("quota", "quota_reached"),
        ("malformed", "invalid_response"),
        ("crash", "process_exited"),
    ],
)
async def test_failed_generation(settings, prompt, reason):
    async with CodexProvider(CodexClient(settings)) as provider:
        with pytest.raises(CodexError, match=reason) as caught:
            await provider.execute(request(prompt))
        assert "SECRET" not in str(caught.value.to_dict())


async def test_preflight_and_persistence(settings):
    async with CodexProvider(CodexClient(settings)) as provider:
        with pytest.raises(UnsupportedTaskError):
            await provider.execute(request(require_no_tools=True))
        assert provider.client._process is None
        with pytest.raises(InvalidRequestError):
            await provider.execute(request(temperature=0.2))
        with pytest.raises(InvalidRequestError):
            await provider.execute(request(reasoning_effort="ultra"))
        with pytest.raises(ModelNotFoundError):
            await provider.execute(request().model_copy(update={"model": "no-model"}))
        first = await provider.execute(request(persist_conversation=True))
        second = await provider.execute(request(conversation_id=first.raw_metadata["thread_id"]))
        assert second.raw_metadata["thread_id"] == first.raw_metadata["thread_id"]
        with pytest.raises(InvalidRequestError):
            await provider.execute(request(conversation_id="foreign-thread"))


@pytest.mark.parametrize("prompt", ["hang", "unknown-outcome"])
async def test_timeout_never_retries(settings, prompt):
    async with CodexProvider(CodexClient(settings)) as provider:
        await provider.client.start()
        with pytest.raises(CodexError, match="timeout"):
            await provider.execute(request(prompt).model_copy(update={"timeout": 0.1}))
    calls = messages(settings)
    assert len([m for m in calls if m.get("method") == "turn/start"]) == 1
    assert any(m.get("method") == "turn/interrupt" for m in calls) == (prompt == "hang")


async def test_task_cancellation_and_close(settings):
    provider = CodexProvider(CodexClient(settings))
    task = asyncio.create_task(provider.execute(request("hang")))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if (settings.auth_directory / "fake-requests.jsonl").exists() and any(
            m.get("method") == "turn/start" for m in messages(settings)
        ):
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    process = provider.client._process
    cwd = provider.client.cwd
    await provider.close()
    assert process.returncode is not None
    assert not Path(cwd).exists()
    assert any(m.get("method") == "turn/interrupt" for m in messages(settings))


async def test_consumer_closes_stream(settings):
    async with CodexProvider(CodexClient(settings)) as provider:
        stream = provider.stream(request("partial-hang"))
        assert isinstance(await anext(stream), StreamChunk)
        await stream.aclose()
        # Explicitly closing the outer iterator must promptly close the inner turn.
        await asyncio.sleep(0.05)
    assert any(m.get("method") == "turn/interrupt" for m in messages(settings))


async def test_rpc_errors_and_parallel_ids(settings):
    async with CodexProvider(CodexClient(settings)) as provider:
        results = await asyncio.gather(*(provider.account() for _ in range(5)))
        assert all(r.connected for r in results)
        with pytest.raises(CodexError, match="incompatible_version") as caught:
            await provider.client.request("test/error")
        assert "SECRET" not in str(caught.value)


async def test_missing_version_and_personal_config(settings, tmp_path):
    with pytest.raises(CodexError, match="executable_missing"):
        await CodexClient(settings.model_copy(update={"executable": "/nonexistent/codex"})).start()
    settings.auth_directory.mkdir()
    (settings.auth_directory / "config.toml").write_text('model_provider="unexpected"')
    with pytest.raises(CodexError, match="personal_configuration_refused"):
        await CodexClient(settings).start()
    executable = tmp_path / "old-codex"
    executable.write_text(f'#!{sys.executable}\nprint("codex-cli 0.1.0")\n')
    executable.chmod(0o700)
    with pytest.raises(CodexError, match="incompatible_version"):
        await CodexClient(
            settings.model_copy(
                update={"executable": str(executable), "auth_directory": tmp_path / "other"}
            )
        ).start()


def test_sync_facade(settings):
    config = CortexMuxConfig()
    config.providers.codex = settings
    config.providers.ollama.enabled = False
    config.providers.comfyui.enabled = False
    with CortexMux(config) as mux:
        assert mux.codex_account().connected
        assert (
            mux.generate(
                "normal", provider="codex", model="test-model", require_no_tools=False
            ).content
            == "answer"
        )
        assert mux.codex_rate_limits().rate_limits.primary.used_percent == 25


def test_sync_facade_structured_stream_and_login(settings):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: int

    config = CortexMuxConfig()
    config.providers.codex = settings
    config.providers.ollama.enabled = False
    config.providers.comfyui.enabled = False
    with CortexMux(config) as mux:
        login = mux.codex_login_start(device_code=True)
        assert mux.codex_login_result(login.login_id).success
        events = list(
            mux.stream_structured(
                "normal",
                response_model=Answer,
                provider="codex",
                model="test-model",
                require_no_tools=False,
            )
        )
        assert events[-1].parsed.answer == 42
        assert events[-1].raw_metadata["thread_id"]
        assert list(mux.stream(request()))[-1].done


async def test_response_bounds(settings):
    async with CodexProvider(
        CodexClient(settings.model_copy(update={"max_output_characters": 3}))
    ) as provider:
        with pytest.raises(CodexError, match="output_limit"):
            await provider.execute(request())


@pytest.mark.parametrize(
    "info, reason",
    [
        ("unauthorized", "connection_expired"),
        ("sandboxError", "permission_required"),
        ({"httpConnectionFailed": {"httpStatusCode": 429}}, "quota_reached"),
    ],
)
def test_structured_error_classification(info, reason):
    from cortexmux.providers.codex.errors import server_error

    error = server_error({"codexErrorInfo": info, "message": "SECRET"})
    assert error.context["reason"] == reason
    assert "SECRET" not in str(error.to_dict())


async def test_default_disabled_and_capabilities(settings):
    assert not CortexMuxConfig().providers.codex.enabled
    provider = CodexProvider(CodexClient(settings))
    capability = (await provider.get_capabilities())[0]
    assert capability.structured_output and capability.streaming
    assert capability.metadata["strict_no_tools"] is False
    assert not capability.locally_hosted
    assert provider.client._process is None
    await provider.close()


async def test_pending_login_timeout_cancel_and_close(settings):
    settings.auth_directory.mkdir()
    (settings.auth_directory / "fake-pending-login").touch()
    async with CodexProvider(CodexClient(settings)) as provider:
        login = await provider.login_start()
        with pytest.raises(CodexError, match="timeout"):
            await provider.login_result(login.login_id, timeout=0.01)
        await provider.login_cancel(login.login_id)
        assert not (await provider.login_result(login.login_id)).success
        await provider.login_start()
    assert len([m for m in messages(settings) if m.get("method") == "account/login/cancel"]) == 2


async def test_output_limit_does_not_emit_oversized_delta(settings):
    async with CodexProvider(
        CodexClient(settings.model_copy(update={"max_output_characters": 3}))
    ) as provider:
        stream = provider.stream(request())
        with pytest.raises(CodexError, match="output_limit"):
            await anext(stream)
        await stream.aclose()


def test_sync_stream_timeout_after_first_delta(settings):
    config = CortexMuxConfig()
    config.providers.codex = settings
    config.providers.ollama.enabled = False
    config.providers.comfyui.enabled = False
    with CortexMux(config) as mux:
        mux.codex_account()
        stream = mux.stream(request("partial-hang").model_copy(update={"timeout": 0.15}))
        assert next(stream).content == "answer"
        with pytest.raises(CodexError, match="timeout"):
            next(stream)
        stream.close()
    assert any(m.get("method") == "turn/interrupt" for m in messages(settings))
