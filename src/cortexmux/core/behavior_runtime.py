"""Shared behavior validation around the existing provider-owned transports."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cortexmux.behavior import BehaviorBinding, ModelIdentity
from cortexmux.core.exceptions import (
    CortexMuxError,
    InvalidRequestError,
    ProviderResponseError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.json_schema import validate_json_schema, validate_json_schema_definition
from cortexmux.core.normalization import normalize_envelope, normalize_json_object
from cortexmux.schemas.common import UsageMetadata
from cortexmux.schemas.requests import CortexRequest, StructuredOutputRequest
from cortexmux.schemas.responses import StructuredResponse


class BehaviorOptions(BaseModel):
    """Provider-neutral intent accepted only by a bound behavior contract."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    behavior_profile: str | None = None
    capture_private: bool = False
    normalization: Literal["profile", "strict"] = "profile"
    required_capabilities: list[str] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, gt=0)
    thinking_budget: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class BehaviorDiagnostic(BaseModel):
    """Public, bounded provenance; never contains prompts, fixtures or HTTP secrets."""

    profile_id: str
    revision: int
    fingerprint: str
    rules_fingerprint: str
    provider: str
    requested_model: str
    reported_model: str | None = None
    reported_model_is_alias: bool | None = None
    server: ModelIdentity
    identity_source: Literal["declared", "observed"]
    qualification_kind: Literal["offline", "live"] | None = None
    requested_parameters: dict[str, Any] | None = None
    transmitted_parameters: dict[str, Any] | None = None
    output_cap_scope: str
    observed_server_version: str | None = None
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
    reasoning_tokens: int | None = None
    transformations: list[str] = Field(default_factory=list)
    category: str = "compatibility"
    response_body_truncated: bool = False
    duration_seconds: float | None = None


class CompletionClient(Protocol):
    """Existing provider client interface, including explicit private capture."""

    async def complete(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
        private_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use the provider-owned transport without adding retries."""
        ...


def validate_bindings(bindings: dict[str, BehaviorBinding], provider: str) -> None:
    """Reject conflicting aliases and contradictory registry keys at construction."""
    owners: dict[str, str] = {}
    revisions: dict[tuple[str, int], str] = {}
    for model, binding in bindings.items():
        key = (binding.profile.profile_id, binding.profile.revision)
        fingerprint = binding.profile.rules_fingerprint
        if key in revisions and revisions[key] != fingerprint:
            raise InvalidRequestError("Behavior profile revision has conflicting rules.")
        revisions[key] = fingerprint
        if model != binding.identity.model or binding.profile.provider != provider:
            raise InvalidRequestError("Behavior registry identity is contradictory.")
        for selector in (model, binding.selector, *binding.reported_selectors):
            if selector is None:
                continue
            if selector in owners and owners[selector] != model:
                raise InvalidRequestError("Behavior registry aliases are ambiguous.")
            owners[selector] = model


async def execute_behavior(
    request: CortexRequest,
    binding: BehaviorBinding,
    *,
    provider: str,
    server_version: str | None,
    client: CompletionClient,
    observe_version: Callable[[], Awaitable[str | None]] | None = None,
) -> StructuredResponse:
    """Validate, translate, execute and diagnose one resolved structured call."""
    profile = binding.profile
    diagnostic = BehaviorDiagnostic(
        profile_id=profile.profile_id,
        revision=profile.revision,
        fingerprint=profile.fingerprint,
        rules_fingerprint=profile.rules_fingerprint,
        provider=provider,
        requested_model=binding.identity.model,
        server=binding.identity,
        identity_source=binding.identity_source,
        output_cap_scope=profile.rules.output_cap_scope,
    )
    started = perf_counter()
    audit: dict[str, Any] = {}
    options: BehaviorOptions | None = None
    try:
        if not isinstance(request, StructuredOutputRequest):
            raise UnsupportedTaskError("Behavior profiles currently require structured output.")
        try:
            options = BehaviorOptions.model_validate(request.options)
        except ValidationError:
            raise InvalidRequestError("Invalid behavior request options.") from None
        diagnostic.requested_parameters = options.model_dump(
            exclude={"capture_private"}, exclude_unset=True
        )
        diagnostic.requested_parameters["think"] = request.think
        diagnostic.qualification_kind = binding.validate_target(
            provider, request.model or "", server_version
        )
        if request.task.value not in profile.rules.tasks:
            raise UnsupportedTaskError("Task is not qualified by the behavior profile.")
        if (
            options.behavior_profile is not None
            and options.behavior_profile != f"{profile.profile_id}@{profile.revision}"
        ):
            raise InvalidRequestError("Requested behavior profile is not bound.")
        capabilities: set[str] = set(profile.rules.tasks)
        if profile.rules.reasoning != "unsupported":
            capabilities.add("reasoning")
        if profile.rules.reasoning_budget:
            capabilities.add("reasoning_budget")
        if not set(options.required_capabilities) <= capabilities:
            raise UnsupportedTaskError("Required capability is unavailable.")
        for sampling_option in ("temperature", "top_p"):
            if (
                getattr(options, sampling_option) is not None
                and sampling_option not in profile.rules.sampling
            ):
                raise UnsupportedTaskError("Sampling control is not supported by this profile.")
        if options.max_tokens is None:
            raise InvalidRequestError("Behavior profile requires max_tokens.")
        if request.think is not None and profile.rules.reasoning == "unsupported":
            raise UnsupportedTaskError("Reasoning controls are not supported by this profile.")
        if options.thinking_budget is not None and (
            not profile.rules.reasoning_budget or request.think is not True
        ):
            raise UnsupportedTaskError(
                "Reasoning budget is unsupported or reasoning is not enabled."
            )
        if (
            provider == "deepseek"
            and request.think is True
            and (options.temperature is not None or options.top_p is not None)
        ):
            raise UnsupportedTaskError("Thinking with sampling overrides is not qualified.")
        schema = request.json_schema
        if schema is not None:
            try:
                validate_json_schema_definition(schema)
            except (ValueError, RecursionError):
                diagnostic.category = "schema"
                raise InvalidRequestError("Invalid or unsupported JSON Schema.") from None
        messages = [
            {
                "role": "system",
                "content": "Return only valid JSON."
                + (" Match this JSON Schema: " + json.dumps(schema) if schema is not None else ""),
            }
        ]
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        controls: dict[str, Any] = {"stream": False, "max_tokens": options.max_tokens}
        for name in ("temperature", "top_p"):
            if (value := getattr(options, name)) is not None:
                controls[name] = value
        if request.think is not None:
            if profile.rules.reasoning == "mlx":
                controls["enable_thinking"] = request.think
            else:
                controls["thinking"] = {"type": "enabled" if request.think else "disabled"}
        if options.thinking_budget is not None:
            controls["thinking_budget"] = options.thinking_budget
        payload = {"model": binding.selector or request.model, "messages": messages, **controls}
        if observe_version:
            diagnostic.category = "transport"
            observed = await observe_version()
            audit["observed_server_version"] = observed
            diagnostic.category = "compatibility"
            if observed is not None and observed != binding.identity.server_version:
                raise ProviderResponseError("Server version does not match the bound identity.")
            diagnostic.observed_server_version = observed
        diagnostic.category = "transport"
        diagnostic.transmitted_parameters = controls
        data = await client.complete(
            payload, request_id=request.request_id, timeout=request.timeout, private_audit=audit
        )
        choices = data.get("choices")
        diagnostic.category = "completion"
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderResponseError("Response must contain exactly one choice.")
        choice = choices[0]
        finish = choice.get("finish_reason")
        audit["finish_reason"] = finish
        diagnostic.finish_reason = (
            finish
            if isinstance(finish, str)
            and finish in {"stop", "length", "tool_calls", "content_filter"}
            else None
        )
        raw_usage = data.get("usage")
        diagnostic.usage = normalized_usage(raw_usage)
        details = (
            raw_usage.get("completion_tokens_details") if isinstance(raw_usage, dict) else None
        )
        diagnostic.reasoning_tokens = (
            _count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        )
        if finish == "length":
            diagnostic.category = "truncation"
            raise ProviderResponseError("Response was truncated.")
        if finish != profile.rules.finish_reason:
            raise ProviderResponseError(
                "Response did not complete with the required finish reason."
            )
        diagnostic.category = "compatibility"
        for version in (data.get("server_version"), audit.get("server_version_header")):
            if version is not None and version != binding.identity.server_version:
                raise ProviderResponseError("Server version does not match the bound identity.")
            if version is not None:
                diagnostic.observed_server_version = version
        reported = data.get("model")
        audit["reported_model"] = reported
        if not isinstance(reported, str) or reported not in (
            request.model,
            binding.selector,
            *binding.reported_selectors,
        ):
            raise ProviderResponseError("Response model does not match the explicit binding.")
        diagnostic.reported_model = request.model
        diagnostic.reported_model_is_alias = reported != request.model
        count = raw_usage.get("completion_tokens") if isinstance(raw_usage, dict) else None
        if count is not None and (_count(count) is None or count > options.max_tokens):
            diagnostic.category = "truncation"
            raise ProviderResponseError("Reported output count violates the contract.")
        if (
            options.thinking_budget is not None
            and diagnostic.reasoning_tokens is not None
            and diagnostic.reasoning_tokens > options.thinking_budget
        ):
            raise ProviderResponseError("Reported reasoning count exceeds the requested budget.")
        diagnostic.category = "completion"
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("tool_calls")
            or not isinstance(message.get("content"), str)
        ):
            raise ProviderResponseError("Response has no valid assistant text.")
        diagnostic.category = "syntax"
        try:
            normalized, transforms = normalize_envelope(
                message["content"],
                think_prefix=profile.rules.think_prefix and options.normalization == "profile",
                json_fence=profile.rules.json_fence and options.normalization == "profile",
            )
            diagnostic.transformations = transforms
            if options.capture_private:
                audit["normalized_response"] = normalized
            parsed = normalize_json_object(normalized)[1]
        except (ValueError, RecursionError):
            raise StructuredOutputValidationError("Invalid JSON syntax.") from None
        diagnostic.category = "schema"
        try:
            if schema is not None:
                validate_json_schema(parsed, schema)
        except (ValueError, RecursionError, re.error):
            raise StructuredOutputValidationError("JSON does not match the schema.") from None
        diagnostic.category = "success"
        diagnostic.duration_seconds = perf_counter() - started
        response = StructuredResponse(
            provider=provider,
            model=request.model,
            request_id=request.request_id,
            content=normalized,
            parsed=parsed,
            usage=diagnostic.usage,
            raw_metadata={"behavior": diagnostic.model_dump(mode="json")},
        )
        if options.capture_private:
            response._private_audit = audit
        return response
    except CortexMuxError as error:
        diagnostic.response_body_truncated = bool(audit.get("response_body_truncated"))
        if error.context.get("category") == "size":
            diagnostic.category = "size"
            diagnostic.response_body_truncated = True
        diagnostic.duration_seconds = perf_counter() - started
        error.context.update(
            provider=provider,
            request_id=request.request_id,
            behavior=diagnostic.model_dump(mode="json"),
        )
        if options is not None and options.capture_private:
            error._private_audit = audit
        raise


def normalized_usage(value: Any) -> UsageMetadata | None:
    """Keep absent or malformed counts unknown, including cache measurements."""
    if not isinstance(value, dict):
        return None
    return UsageMetadata(
        prompt_tokens=_count(value.get("prompt_tokens")),
        completion_tokens=_count(value.get("completion_tokens")),
        total_tokens=_count(value.get("total_tokens")),
        cache_hit_tokens=_count(value.get("prompt_cache_hit_tokens")),
        cache_miss_tokens=_count(value.get("prompt_cache_miss_tokens")),
    )


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None
