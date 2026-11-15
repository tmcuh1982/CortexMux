"""Offline tests for the opt-in OpenAI WebSocket conversation contract."""

from __future__ import annotations

from typing import Any

import pytest

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import InvalidRequestError
from cortexmux.providers.openai import (
    OpenAIAsyncSession,
    OpenAIFunctionTool,
    OpenAIReasoningEffort,
    OpenAIToolResult,
)
from cortexmux.providers.openai import schemas as openai_schemas
from cortexmux.providers.openai.schemas import OpenAIModelCapabilities


class FakeConnection:
    """A deterministic Responses WebSocket stand-in."""

    def __init__(self) -> None:
        self.response = self
        self.sent: list[dict[str, Any]] = []
        self.steers: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    async def __aenter__(self) -> FakeConnection:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def __aiter__(self) -> FakeConnection:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def create(self, **payload: Any) -> None:
        self.sent.append(payload)

    async def steer(self, **payload: Any) -> None:
        self.steers.append(payload)


def _session(connection: FakeConnection) -> OpenAIAsyncSession:
    return OpenAIAsyncSession(
        api_key="synthetic-key",
        tools=[
            OpenAIFunctionTool(
                name="lookup",
                description="Return a synthetic local lookup value.",
                parameters={
                    "type": "object",
                    "properties": {"term": {"type": "string"}},
                    "required": ["term"],
                    "additionalProperties": False,
                },
                async_=True,
            )
        ],
        instructions="Use only supplied tool results.",
        _connection_context=connection,
    )


@pytest.mark.asyncio
async def test_async_tool_result_uses_original_call_id_and_latest_response() -> None:
    connection = FakeConnection()
    async with _session(connection) as session:
        await session.start("Look up the synthetic value and keep working.")
        assert connection.sent[0]["store"] is False
        assert connection.sent[0]["tools"][0]["async"] is True
        assert connection.sent[0]["reasoning"] == {"effort": "low"}

        connection.events.extend(
            [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"term":"sample"}',
                        "async": True,
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "lookup",
                                "arguments": '{"term":"sample"}',
                                "async": True,
                            },
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "Independent work."}],
                            },
                        ],
                    },
                },
            ]
        )
        await session.next_event()
        call_event = await session.next_event()
        assert call_event.tool_call is not None
        assert call_event.tool_call.call_id == "call_1"
        assert session.pending_call_ids == {"call_1"}
        completed = await session.next_event()
        assert completed.response is not None
        assert completed.response.content == "Independent work."
        assert completed.response.tool_calls[0].call_id == "call_1"
        assert completed.response.tool_calls[0].async_ is True
        assert session.pending_call_ids == {"call_1"}

        with pytest.raises(InvalidRequestError, match="Unknown or duplicate"):
            await session.continue_with_tool_results(
                [OpenAIToolResult(call_id="invented", output="value")]
            )
        await session.continue_with_tool_results(
            [OpenAIToolResult(call_id="call_1", output='{"value":7}')]
        )
        assert connection.sent[1]["previous_response_id"] == "resp_1"
        assert connection.sent[1]["input"] == [
            {"type": "function_call_output", "call_id": "call_1", "output": '{"value":7}'}
        ]
        assert session.pending_call_ids == frozenset()


@pytest.mark.asyncio
async def test_steering_queues_correction_and_reads_automatic_successor() -> None:
    connection = FakeConnection()
    async with _session(connection) as session:
        await session.start("Draft a plan.")
        connection.events.append({"type": "response.created", "response": {"id": "resp_1"}})
        await session.next_event()
        await session.steer("Keep the scope to two weeks.")
        assert connection.steers == [
            {"previous_response_id": "resp_1", "input": "Keep the scope to two weeks."}
        ]
        connection.events.extend(
            [
                {"type": "response.steer.accepted", "steer": {"id": "steer_1"}},
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_1",
                        "incomplete_details": {"reason": "steered"},
                    },
                },
                {"type": "response.created", "response": {"id": "resp_2"}},
                {"type": "response.completed", "response": {"id": "resp_2", "output": []}},
            ]
        )
        accepted = await session.next_event()
        assert accepted.steer_id == "steer_1"
        await session.next_event()
        with pytest.raises(InvalidRequestError, match="No completed response"):
            await session.follow_up("Do something else.")
        await session.next_event()
        completed = await session.next_event()
        assert completed.response_id == "resp_2"
        assert session.latest_response_id == "resp_2"
        assert len(connection.sent) == 1


@pytest.mark.asyncio
async def test_reasoning_update_keeps_request_effort_and_prefix() -> None:
    connection = FakeConnection()
    async with _session(connection) as session:
        await session.start("Make a short plan.")
        connection.events.extend(
            [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {"type": "response.completed", "response": {"id": "resp_1", "output": []}},
            ]
        )
        await session.next_event()
        await session.next_event()
        await session.follow_up("Analyze failure modes.", reasoning_effort="high")
        assert connection.sent[1]["previous_response_id"] == "resp_1"
        assert connection.sent[1]["reasoning"] == {"effort": "low"}
        assert connection.sent[1]["input"] == [
            {"type": "configuration_update", "reasoning": {"effort": "high"}},
            {"role": "user", "content": "Analyze failure modes."},
        ]
        assert session.reasoning_effort == OpenAIReasoningEffort.HIGH


def test_session_rejects_unsupported_effort_before_connecting() -> None:
    with pytest.raises(InvalidRequestError, match="Unsupported OpenAI reasoning effort"):
        OpenAIAsyncSession(api_key="synthetic-key", reasoning_effort="none")


def test_session_rejects_models_without_registered_async_capability() -> None:
    with pytest.raises(InvalidRequestError, match="does not support OpenAI async sessions"):
        OpenAIAsyncSession(api_key="synthetic-key", model="gpt-5.6-luna")


@pytest.mark.asyncio
async def test_new_model_can_be_enabled_by_adding_its_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        openai_schemas._MODEL_CAPABILITIES,
        "future-model",
        OpenAIModelCapabilities(
            reasoning_efforts=frozenset({OpenAIReasoningEffort.LOW}),
            default_reasoning_effort=OpenAIReasoningEffort.LOW,
            async_session=True,
        ),
    )
    connection = FakeConnection()
    async with OpenAIAsyncSession(
        api_key="synthetic-key", model="future-model", _connection_context=connection
    ) as session:
        await session.start("Hello")
        assert connection.sent[0]["model"] == "future-model"
        connection.events.append({"type": "response.created", "response": {"id": "resp_future"}})
        await session.next_event()
        with pytest.raises(InvalidRequestError, match="mid-turn steering"):
            await session.steer("Change direction")


@pytest.mark.asyncio
async def test_facade_exposes_async_session_and_forwards_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cortexmux.facade as facade

    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            captured["closed"] = True

    monkeypatch.setattr(facade, "OpenAIAsyncSession", FakeSession)
    config = CortexMuxConfig.model_validate(
        {
            "core": {"allow_remote_hosts": True, "approved_hosts": ["api.openai.com"]},
            "providers": {"openai": {"enabled": True, "api_key": "synthetic-key"}},
        }
    )
    mux = CortexMux(config, register_builtin_providers=False)
    async with mux.openai_async_session(model="gpt-6-astra"):
        assert captured["model"] == "gpt-6-astra"
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_pending_steer_waits_for_tool_output_without_resending_user_input() -> None:
    connection = FakeConnection()
    async with _session(connection) as session:
        await session.start("Look up a synthetic value.")
        connection.events.extend(
            [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {"type": "response.steer.accepted", "steer": {"id": "steer_1"}},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "lookup",
                                "arguments": '{"term":"sample"}',
                                "async": True,
                            }
                        ],
                    },
                },
                {
                    "type": "response.steer.pending",
                    "required_input": [{"type": "function_call_output", "call_id": "call_1"}],
                    "steer": {"id": "steer_1"},
                },
            ]
        )
        await session.next_event()
        await session.next_event()
        await session.next_event()
        with pytest.raises(InvalidRequestError, match="No completed response"):
            await session.continue_with_tool_results(
                [OpenAIToolResult(call_id="call_1", output="7")]
            )
        pending = await session.next_event()
        assert pending.required_call_ids == ["call_1"]
        await session.continue_with_tool_results([OpenAIToolResult(call_id="call_1", output="7")])
        assert connection.sent[1]["previous_response_id"] == "resp_1"
        assert connection.sent[1]["input"] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "7"}
        ]
