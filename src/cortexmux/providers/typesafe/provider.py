"""Normalize TypeSafe typed decisions without domain-specific policy."""

from __future__ import annotations

import re
from typing import Any

from pydantic import TypeAdapter, ValidationError

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    CortexMuxError,
    InvalidRequestError,
    ProviderResponseError,
    UnsupportedTaskError,
)
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.typesafe.client import TypeSafeClient
from cortexmux.schemas.common import HealthStatus, ModelInfo, UsageMetadata
from cortexmux.schemas.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from cortexmux.schemas.requests import CortexRequest, DecisionRequest
from cortexmux.schemas.responses import CortexResponse, DecisionResponse

_ANSWER_ADAPTER: TypeAdapter[DecisionAnswer] = TypeAdapter(DecisionAnswer)


class TypeSafeProvider(BaseProvider):
    """Expose TypeSafe's decision API through the normal provider contract."""

    name = "typesafe"

    def __init__(self, client: TypeSafeClient) -> None:
        self.client = client
        self._models_cache: list[ModelInfo] | None = None

    async def healthcheck(self) -> HealthStatus:
        """Check model discovery without exposing remote response bodies."""
        try:
            await self.list_models()
            return HealthStatus(
                provider=self.name, available=True, message="TypeSafe is available."
            )
        except CortexMuxError:
            return HealthStatus(
                provider=self.name, available=False, message="TypeSafe is unavailable."
            )

    async def list_models(self) -> list[ModelInfo]:
        """List model names and aliases reported by the authenticated API."""
        if self._models_cache is not None:
            return list(self._models_cache)
        data = await self.client.list_models()
        items = data.get("models")
        if not isinstance(items, list):
            raise ProviderResponseError("TypeSafe model list is invalid.", provider=self.name)
        result: list[ModelInfo] = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ProviderResponseError("TypeSafe model entry is invalid.", provider=self.name)
            result.append(ModelInfo(name=item["name"], provider=self.name))
        self._models_cache = result
        return list(result)

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the non-streaming, text-only decision capability."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset({TaskType.DECISION}),
                locally_hosted=False,
                batch=True,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Accept only typed decision requests."""
        return task == TaskType.DECISION

    def accepts_unlisted_model(self, model: str) -> bool:
        """Let the API validate versioned Jev IDs omitted from model discovery."""
        return re.fullmatch(r"jev-\d+\.\d+\.\d+", model) is not None

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Return every answer and probability after contract validation."""
        if not isinstance(request, DecisionRequest):
            raise UnsupportedTaskError("TypeSafe supports only decisions.", provider=self.name)
        if not request.model:
            raise InvalidRequestError("TypeSafe requests require a model.", provider=self.name)
        if request.options:
            raise InvalidRequestError(
                "TypeSafe decision options are not supported.", provider=self.name
            )
        payload = {
            "state": request.state,
            "model": request.model,
            "questions": {
                key: question.model_dump(mode="json", exclude_none=True)
                for key, question in request.questions.items()
            },
        }
        data = await self.client.evaluate(
            payload, request_id=request.request_id, timeout=request.timeout
        )
        raw_answers = data.get("answers")
        model = data.get("model")
        if not isinstance(raw_answers, dict) or set(raw_answers) != set(request.questions):
            raise ProviderResponseError(
                "TypeSafe answer IDs do not match questions.", provider=self.name
            )
        if not isinstance(model, str) or not model:
            raise ProviderResponseError("TypeSafe response model is missing.", provider=self.name)
        answers: dict[str, DecisionAnswer] = {}
        for key, question in request.questions.items():
            try:
                answer = _ANSWER_ADAPTER.validate_python(raw_answers[key])
            except ValidationError:
                raise ProviderResponseError(
                    "TypeSafe answer is invalid.", provider=self.name
                ) from None
            if not _answer_matches_question(answer, question):
                raise ProviderResponseError(
                    "TypeSafe answer does not match its question.", provider=self.name
                )
            answers[key] = answer
        usage = _usage(data.get("usage"))
        return DecisionResponse(
            provider=self.name,
            model=model,
            request_id=request.request_id,
            answers=answers,
            usage=usage,
        )

    async def close(self) -> None:
        """Release the TypeSafe HTTP client."""
        await self.client.close()


def _answer_matches_question(
    answer: DecisionAnswer, question: NoulQuestion | ChoiceQuestion | ScoreQuestion
) -> bool:
    if isinstance(question, NoulQuestion):
        return isinstance(answer, NoulAnswer)
    if isinstance(question, ChoiceQuestion):
        return (
            isinstance(answer, ChoiceAnswer)
            and set(answer.probabilities) == set(question.criteria)
            and answer.probabilities[answer.choice] == max(answer.probabilities.values())
        )
    expected_legend = {
        str(index): description for index, description in enumerate(question.criteria)
    }
    return isinstance(answer, ScoreAnswer) and answer.legend == expected_legend


def _usage(value: Any) -> UsageMetadata | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProviderResponseError("TypeSafe usage is invalid.", provider="typesafe")
    try:
        return UsageMetadata(
            prompt_tokens=value.get("input_tokens"),
            completion_tokens=value.get("output_tokens"),
        )
    except ValidationError:
        raise ProviderResponseError("TypeSafe usage is invalid.", provider="typesafe") from None
