"""Typed request options and transport results for the OpenAI provider."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OpenAIServiceTier(StrEnum):
    """Responses API processing tiers accepted by CortexMux."""

    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    SCALE = "scale"
    PRIORITY = "priority"
    FAST = "fast"
    ULTRAFAST = "ultrafast"


class OpenAIReasoningEffort(StrEnum):
    """Known Responses API reasoning-effort values."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class OpenAIVerbosity(StrEnum):
    """Known Responses API text-verbosity values."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAIRequestOptions(BaseModel):
    """Validated provider options accepted by the OpenAI adapter."""

    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: OpenAIReasoningEffort | None = None
    service_tier: OpenAIServiceTier | None = None
    verbosity: OpenAIVerbosity | None = None


class OpenAIModelCapabilities(BaseModel):
    """Model-specific OpenAI options enforced before transport execution."""

    model_config = ConfigDict(frozen=True)

    reasoning_efforts: frozenset[OpenAIReasoningEffort]
    default_reasoning_effort: OpenAIReasoningEffort | None = None
    async_session: bool = False
    mid_turn_steering: bool = False
    reasoning_updates: bool = False


_GENERAL_REASONING = frozenset(OpenAIReasoningEffort)
_MODEL_CAPABILITIES = {
    "gpt-6-astra": OpenAIModelCapabilities(
        reasoning_efforts=frozenset(
            {
                OpenAIReasoningEffort.LOW,
                OpenAIReasoningEffort.MEDIUM,
                OpenAIReasoningEffort.HIGH,
                OpenAIReasoningEffort.XHIGH,
                OpenAIReasoningEffort.MAX,
            }
        ),
        default_reasoning_effort=OpenAIReasoningEffort.LOW,
        async_session=True,
        mid_turn_steering=True,
        reasoning_updates=True,
    )
}
_DEFAULT_CAPABILITIES = OpenAIModelCapabilities(reasoning_efforts=_GENERAL_REASONING)


def capabilities_for(model: str) -> OpenAIModelCapabilities:
    """Return the explicit option capabilities for an OpenAI model."""
    if model == "gpt-6-astra" or model.startswith("gpt-6-astra-"):
        return _MODEL_CAPABILITIES["gpt-6-astra"]
    return _MODEL_CAPABILITIES.get(model, _DEFAULT_CAPABILITIES)
