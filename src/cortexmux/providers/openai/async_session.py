"""Opt-in OpenAI Responses WebSocket session.

The application owns tool execution. This session only transports typed calls,
results, and steering messages while retaining the latest response lineage.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cortexmux.core.exceptions import (
    InvalidRequestError,
    OptionalDependencyError,
    ProviderResponseError,
)
from cortexmux.core.json_schema import validate_json_schema, validate_json_schema_definition
from cortexmux.providers.openai.schemas import OpenAIReasoningEffort, capabilities_for


class OpenAIFunctionTool(BaseModel):
    """A caller-run function exposed through the Responses API."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: str = Field(default="function", pattern=r"^function$")
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    strict: bool = True
    async_: bool = Field(default=False, alias="async")


class OpenAIToolCall(BaseModel):
    """A function call to execute in application code."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str = Field(min_length=1)
    async_: bool = Field(default=False, alias="async")


class OpenAIToolResult(BaseModel):
    """A completed function result linked to the original API call ID."""

    call_id: str = Field(min_length=1)
    output: str


class OpenAIAsyncResponse(BaseModel):
    """One completed response with its application-owned tool calls."""

    id: str
    content: str = ""
    tool_calls: list[OpenAIToolCall] = Field(default_factory=list)


class OpenAIAsyncEvent(BaseModel):
    """A bounded event from a Responses WebSocket session."""

    type: str
    response_id: str | None = None
    response: OpenAIAsyncResponse | None = None
    tool_call: OpenAIToolCall | None = None
    text_delta: str | None = None
    steer_id: str | None = None
    required_call_ids: list[str] = Field(default_factory=list)


class OpenAIAsyncSession:
    """Manage a single response chain over an opt-in WebSocket connection."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-6-astra",
        instructions: str | None = None,
        tools: list[OpenAIFunctionTool] | None = None,
        reasoning_effort: OpenAIReasoningEffort | str | None = None,
        timeout: float = 120,
        _connection_context: Any | None = None,
    ) -> None:
        capabilities = capabilities_for(model)
        if not capabilities.async_session:
            raise InvalidRequestError(
                "The selected model does not support OpenAI async sessions.", provider="openai"
            )
        parsed_effort = (
            _parse_effort(reasoning_effort, model)
            if reasoning_effort is not None
            else capabilities.default_reasoning_effort
        )
        if not api_key:
            raise InvalidRequestError("An OpenAI API key is required.", provider="openai")
        if timeout <= 0:
            raise InvalidRequestError("Session timeout must be positive.", provider="openai")
        self.model = model
        self._capabilities = capabilities
        self.instructions = instructions
        self.tools = tools or []
        for tool in self.tools:
            try:
                validate_json_schema_definition(tool.parameters)
            except ValueError as exc:
                raise InvalidRequestError(
                    "A function tool has an invalid parameter schema.", provider="openai"
                ) from exc
            if tool.parameters.get("type") != "object":
                raise InvalidRequestError(
                    "Function parameters must have object schema.", provider="openai"
                )
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise InvalidRequestError("Function tool names must be unique.", provider="openai")
        self.reasoning_effort = parsed_effort
        self._request_reasoning_effort = parsed_effort
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._connection_context = _connection_context
        self._connection: Any | None = None
        self._events: Any | None = None
        self._sdk_client: Any | None = None
        self._active_response_id: str | None = None
        self._latest_response_id: str | None = None
        self._awaiting_response = False
        self._queued_steer = False
        self._pending_calls: set[str] = set()

    @property
    def pending_call_ids(self) -> frozenset[str]:
        """Return tool calls whose results have not been submitted."""
        return frozenset(self._pending_calls)

    @property
    def latest_response_id(self) -> str | None:
        """Return the latest completed response used for continuation."""
        return self._latest_response_id

    async def __aenter__(self) -> OpenAIAsyncSession:
        """Open the Responses WebSocket connection."""
        if self._connection is not None:
            raise InvalidRequestError("OpenAI async session is already open.", provider="openai")
        if self._connection_context is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise OptionalDependencyError("openai[realtime]>=3.8.0", "openai") from exc
            self._sdk_client = AsyncOpenAI(
                api_key=self._api_key, base_url=self._base_url, timeout=self._timeout
            )
            self._connection_context = self._sdk_client.responses.connect()
        try:
            self._connection = await self._connection_context.__aenter__()
        except BaseException:
            if self._sdk_client is not None:
                await self._sdk_client.close()
                self._sdk_client = None
            raise
        self._events = self._connection.__aiter__()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the WebSocket and the internally owned SDK client."""
        try:
            if self._connection_context is not None and self._connection is not None:
                await self._connection_context.__aexit__(None, None, None)
        finally:
            self._connection = None
            self._events = None
            if self._sdk_client is not None:
                await self._sdk_client.close()
                self._sdk_client = None

    async def start(self, prompt: str) -> None:
        """Start a response and return before its WebSocket events finish."""
        if self._latest_response_id is not None or self._awaiting_response:
            raise InvalidRequestError(
                "OpenAI async session has already started.", provider="openai"
            )
        await self._create([_user_message(prompt)], previous_response_id=None)

    async def follow_up(
        self, prompt: str, *, reasoning_effort: OpenAIReasoningEffort | str | None = None
    ) -> None:
        """Add a user turn, optionally changing effective effort with a cache-safe item."""
        self._require_idle()
        parsed_effort = (
            _parse_effort(reasoning_effort, self.model) if reasoning_effort is not None else None
        )
        inputs = self._inputs(prompt, parsed_effort)
        await self._create(inputs, previous_response_id=self._latest_response_id)
        if parsed_effort is not None:
            self.reasoning_effort = parsed_effort

    async def continue_with_tool_results(
        self,
        results: list[OpenAIToolResult],
        *,
        prompt: str | None = None,
        reasoning_effort: OpenAIReasoningEffort | str | None = None,
    ) -> None:
        """Return completed tool work using its original call IDs."""
        self._require_idle()
        if not results:
            raise InvalidRequestError("At least one tool result is required.", provider="openai")
        call_ids = [result.call_id for result in results]
        if len(set(call_ids)) != len(call_ids) or not set(call_ids) <= self._pending_calls:
            raise InvalidRequestError("Unknown or duplicate tool call ID.", provider="openai")
        parsed_effort = (
            _parse_effort(reasoning_effort, self.model) if reasoning_effort is not None else None
        )
        if parsed_effort is not None and prompt is None:
            raise InvalidRequestError(
                "A reasoning update requires a following user message.", provider="openai"
            )
        inputs: list[dict[str, Any]] = [
            {"type": "function_call_output", "call_id": item.call_id, "output": item.output}
            for item in results
        ]
        if prompt is not None:
            inputs.extend(self._inputs(prompt, parsed_effort))
        await self._create(inputs, previous_response_id=self._latest_response_id)
        self._pending_calls.difference_update(call_ids)
        if parsed_effort is not None:
            self.reasoning_effort = parsed_effort

    async def steer(self, instruction: str) -> None:
        """Queue a user correction for the currently running response."""
        connection = self._require_connection()
        if not self._capabilities.mid_turn_steering:
            raise InvalidRequestError(
                "The selected model does not support mid-turn steering.", provider="openai"
            )
        if self._active_response_id is None:
            raise InvalidRequestError("No active response to steer.", provider="openai")
        if not instruction.strip():
            raise InvalidRequestError("Steering instruction must be nonempty.", provider="openai")
        await connection.response.steer(
            previous_response_id=self._active_response_id, input=instruction
        )
        self._queued_steer = True

    async def next_event(self) -> OpenAIAsyncEvent:
        """Read and normalize the next event without losing steering acknowledgements."""
        if self._events is None:
            raise InvalidRequestError("OpenAI async session is not open.", provider="openai")
        try:
            raw_event = await self._events.__anext__()
        except StopAsyncIteration as exc:
            raise ProviderResponseError("OpenAI WebSocket closed.", provider="openai") from exc
        event = _to_dict(raw_event)
        kind = event.get("type")
        if not isinstance(kind, str):
            raise ProviderResponseError("OpenAI WebSocket event is invalid.", provider="openai")
        if kind == "response.steer.failed":
            self._queued_steer = False
            if self._active_response_id is None:
                self._awaiting_response = False
        if kind in {"error", "response.failed", "response.steer.failed"}:
            raise ProviderResponseError(
                "OpenAI WebSocket operation failed.", provider="openai", event_type=kind
            )
        response = event.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if kind == "response.created":
            if not isinstance(response_id, str):
                raise ProviderResponseError("OpenAI response ID is missing.", provider="openai")
            self._active_response_id = response_id
            self._awaiting_response = True
            self._queued_steer = False
        elif kind == "response.completed":
            if not isinstance(response, dict) or not isinstance(response_id, str):
                raise ProviderResponseError("OpenAI response is invalid.", provider="openai")
            completed = _completed_response(response)
            for call in completed.tool_calls:
                self._validate_call(call)
            self._pending_calls.update(call.call_id for call in completed.tool_calls)
            self._latest_response_id = response_id
            self._active_response_id = None
            self._awaiting_response = self._queued_steer
            return OpenAIAsyncEvent(type=kind, response_id=response_id, response=completed)
        elif kind == "response.incomplete":
            details = response.get("incomplete_details") if isinstance(response, dict) else None
            if not isinstance(details, dict) or details.get("reason") != "steered":
                raise ProviderResponseError("OpenAI response is incomplete.", provider="openai")
            if isinstance(response, dict) and isinstance(response.get("output"), list):
                for call in _completed_response(response).tool_calls:
                    self._validate_call(call)
                    self._pending_calls.add(call.call_id)
            self._active_response_id = None
            self._awaiting_response = True
        elif kind == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call = _function_call(item)
                self._validate_call(call)
                self._pending_calls.add(call.call_id)
                return OpenAIAsyncEvent(type=kind, response_id=response_id, tool_call=call)
        elif kind == "response.steer.accepted":
            self._queued_steer = True
        elif kind == "response.steer.pending":
            self._awaiting_response = False
        steer = event.get("steer")
        required_input = event.get("required_input")
        return OpenAIAsyncEvent(
            type=kind,
            response_id=response_id if isinstance(response_id, str) else None,
            text_delta=event.get("delta") if isinstance(event.get("delta"), str) else None,
            steer_id=steer.get("id") if isinstance(steer, dict) else None,
            required_call_ids=[
                item["call_id"]
                for item in required_input
                if isinstance(item, dict) and isinstance(item.get("call_id"), str)
            ]
            if isinstance(required_input, list)
            else [],
        )

    def _inputs(
        self, prompt: str, reasoning_effort: OpenAIReasoningEffort | None
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if reasoning_effort is not None:
            if not self._capabilities.reasoning_updates:
                raise InvalidRequestError(
                    "The selected model does not support reasoning updates.", provider="openai"
                )
            items.append(
                {"type": "configuration_update", "reasoning": {"effort": reasoning_effort.value}}
            )
        items.append(_user_message(prompt))
        return items

    async def _create(
        self, inputs: list[dict[str, Any]], *, previous_response_id: str | None
    ) -> None:
        connection = self._require_connection()
        payload: dict[str, Any] = {
            "stream_id": "main",
            "model": self.model,
            "store": False,
            "input": inputs,
            "tools": [tool.model_dump(by_alias=True) for tool in self.tools],
        }
        if self._request_reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._request_reasoning_effort.value}
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        elif self._latest_response_id is not None:
            raise InvalidRequestError("A previous response ID is required.", provider="openai")
        await connection.response.create(**payload)
        self._awaiting_response = True
        self._queued_steer = False

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise InvalidRequestError("OpenAI async session is not open.", provider="openai")
        return self._connection

    def _require_idle(self) -> None:
        self._require_connection()
        if self._latest_response_id is None or self._awaiting_response:
            raise InvalidRequestError("No completed response to continue.", provider="openai")

    def _validate_call(self, call: OpenAIToolCall) -> None:
        tool = next((item for item in self.tools if item.name == call.name), None)
        if tool is None:
            raise ProviderResponseError("OpenAI returned an unknown function.", provider="openai")
        try:
            arguments = json.loads(call.arguments)
            validate_json_schema(arguments, tool.parameters)
        except (ValueError, TypeError) as exc:
            raise ProviderResponseError(
                "OpenAI function arguments are invalid.", provider="openai"
            ) from exc


def _user_message(prompt: str) -> dict[str, Any]:
    if not prompt.strip():
        raise InvalidRequestError("User prompt must be nonempty.", provider="openai")
    return {"role": "user", "content": prompt}


def _parse_effort(value: OpenAIReasoningEffort | str, model: str) -> OpenAIReasoningEffort:
    try:
        effort = OpenAIReasoningEffort(value)
    except ValueError as exc:
        raise InvalidRequestError(
            "Unsupported OpenAI reasoning effort.", provider="openai"
        ) from exc
    if effort not in capabilities_for(model).reasoning_efforts:
        raise InvalidRequestError("Unsupported OpenAI reasoning effort.", provider="openai")
    return effort


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        result: Any = value.model_dump(mode="json", by_alias=True)
        if isinstance(result, dict):
            return result
    raise ProviderResponseError("OpenAI WebSocket event is invalid.", provider="openai")


def _completed_response(data: dict[str, Any]) -> OpenAIAsyncResponse:
    output = data.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("OpenAI response output is invalid.", provider="openai")
    fragments: list[str] = []
    calls: list[OpenAIToolCall] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            calls.append(_function_call(item))
        elif item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            fragments.append(text)
    return OpenAIAsyncResponse(id=data["id"], content="".join(fragments), tool_calls=calls)


def _function_call(item: dict[str, Any]) -> OpenAIToolCall:
    try:
        return OpenAIToolCall.model_validate(item)
    except ValueError as exc:
        raise ProviderResponseError("OpenAI function call is invalid.", provider="openai") from exc
