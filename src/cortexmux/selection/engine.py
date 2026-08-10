"""Deterministic cross-provider model qualification and scoring."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from cortexmux.core.exceptions import CortexMuxError, ProviderResponseError
from cortexmux.core.registry import ProviderRegistry
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.schemas.common import ChatMessage, ModelInfo, UsageMetadata
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest
from cortexmux.schemas.responses import ChatResponse, StructuredResponse
from cortexmux.selection.machine import detect_machine_profile
from cortexmux.selection.schemas import (
    AdaptiveModelRoute,
    BenchmarkCase,
    CandidateConfiguration,
    CandidateQualification,
    CaseQualification,
    MachineProfile,
    ModelRecommendation,
    QualificationManifest,
    QualificationSuite,
    ValidationKind,
)


class ModelQualifier:
    """Benchmark exact candidate configurations and recommend validated routes."""

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        machine: MachineProfile | None = None,
    ) -> None:
        self.registry = registry
        self.machine = machine or detect_machine_profile()

    async def qualify(self, suite: QualificationSuite) -> QualificationManifest:
        """Run the bounded suite sequentially and return an auditable manifest."""
        qualifications = [await self._qualify_candidate(item, suite) for item in suite.candidates]
        recommendations = self._recommend(suite, qualifications)
        return QualificationManifest(
            machine=self.machine,
            candidates=qualifications,
            recommendations=recommendations,
            routing_profiles=_routing_profiles(recommendations),
        )

    async def _qualify_candidate(
        self,
        candidate: CandidateConfiguration,
        suite: QualificationSuite,
    ) -> CandidateQualification:
        try:
            provider = self.registry.get(candidate.provider)
            models = await provider.list_models()
        except CortexMuxError as exc:
            return self._unavailable(candidate, exc.message)
        model_info = next((item for item in models if item.name == candidate.model), None)
        if model_info is None:
            return self._unavailable(candidate, "Model is not available from this provider.")
        estimated_memory = candidate.estimated_memory_bytes or _estimate_memory(model_info)
        results: list[CaseQualification] = []
        applicable = [case for case in suite.cases if case.task in candidate.tasks]
        for case in applicable:
            for repetition in range(1, suite.repetitions + 1):
                results.append(await self._run_case(provider, candidate, case, repetition))
        return CandidateQualification(
            candidate_id=candidate.id,
            provider=candidate.provider,
            model=candidate.model,
            options=candidate.options,
            available=True,
            estimated_memory_bytes=estimated_memory,
            memory_fit=_memory_fit(estimated_memory, self.machine.total_memory_bytes),
            cases=results,
        )

    async def _run_case(
        self,
        provider: Any,
        candidate: CandidateConfiguration,
        case: BenchmarkCase,
        repetition: int,
    ) -> CaseQualification:
        started = perf_counter()
        try:
            if case.task is TaskType.CHAT:
                response = await provider.execute(
                    ChatRequest(
                        provider=candidate.provider,
                        model=candidate.model,
                        messages=[ChatMessage(role=MessageRole.USER, content=case.prompt)],
                        options=candidate.options,
                    )
                )
                if not isinstance(response, ChatResponse):
                    raise ProviderResponseError("Benchmark expected a chat response.")
                passed = _validate_text(response.content, case)
                observed = response.content[:1_000]
            else:
                response = await provider.execute(
                    StructuredOutputRequest(
                        provider=candidate.provider,
                        model=candidate.model,
                        prompt=case.prompt,
                        json_schema=case.json_schema,
                        think=False,
                        options=candidate.options,
                    )
                )
                if not isinstance(response, StructuredResponse):
                    raise ProviderResponseError("Benchmark expected a structured response.")
                passed = _validate_json(response.parsed, case)
                observed = json.dumps(response.parsed, ensure_ascii=False, default=str)[:1_000]
            latency = perf_counter() - started
            return CaseQualification(
                case_id=case.id,
                task=case.task,
                repetition=repetition,
                passed=passed,
                latency_seconds=latency,
                prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                completion_tokens=response.usage.completion_tokens if response.usage else None,
                estimated_cost_usd=_estimate_cost(response.usage, candidate),
                observed_excerpt=observed,
            )
        except CortexMuxError as exc:
            return CaseQualification(
                case_id=case.id,
                task=case.task,
                repetition=repetition,
                passed=False,
                latency_seconds=perf_counter() - started,
                error=type(exc).__name__,
            )

    def _recommend(
        self,
        suite: QualificationSuite,
        qualifications: list[CandidateQualification],
    ) -> list[ModelRecommendation]:
        candidate_by_id = {item.id: item for item in suite.candidates}
        recommendations: list[ModelRecommendation] = []
        for profile, weights in suite.profile_weights.items():
            if profile not in {"fast", "balanced", "quality"}:
                continue
            for task in (TaskType.CHAT, TaskType.STRUCTURED_OUTPUT):
                eligible = [
                    item
                    for item in qualifications
                    if item.available
                    and profile in candidate_by_id[item.candidate_id].profiles
                    and any(case.task is task for case in item.cases)
                ]
                metrics = [_candidate_metrics(item, task, suite) for item in eligible]
                metrics = [item for item in metrics if item[0] >= suite.minimum_quality]
                if not metrics:
                    continue
                fastest = min(item[1] for item in metrics)
                ranked: list[ModelRecommendation] = []
                for quality, latency, cost, item in metrics:
                    latency_score = fastest / latency if latency > 0 else 1
                    cost_score = 1 / (1 + cost / suite.cost_reference_usd)
                    score = (
                        quality * weights.quality
                        + latency_score * weights.latency
                        + cost_score * weights.cost
                        + item.memory_fit * weights.memory
                    )
                    ranked.append(
                        ModelRecommendation(
                            profile=profile,
                            task=task,
                            candidate_id=item.candidate_id,
                            provider=item.provider,
                            model=item.model,
                            options=item.options,
                            score=min(score, 1),
                            quality_score=quality,
                            latency_score=min(latency_score, 1),
                            cost_score=cost_score,
                            memory_score=item.memory_fit,
                        )
                    )
                recommendations.append(
                    max(
                        ranked,
                        key=lambda item: (
                            item.score,
                            -len(candidate_by_id[item.candidate_id].options),
                        ),
                    )
                )
        return recommendations

    def _unavailable(self, candidate: CandidateConfiguration, error: str) -> CandidateQualification:
        return CandidateQualification(
            candidate_id=candidate.id,
            provider=candidate.provider,
            model=candidate.model,
            options=candidate.options,
            available=False,
            error=error,
        )


def _validate_text(content: str, case: BenchmarkCase) -> bool:
    if case.validator is not ValidationKind.EXACT_TEXT or case.expected_text is None:
        return False
    return content.strip() == case.expected_text.strip()


def _validate_json(parsed: Any, case: BenchmarkCase) -> bool:
    if case.validator is not ValidationKind.EXACT_JSON or case.expected_json is None:
        return False
    return isinstance(parsed, dict) and _contains_expected(parsed, case.expected_json)


def _contains_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict) or not _contains_expected(
                actual_value, expected_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _estimate_cost(
    usage: UsageMetadata | None,
    candidate: CandidateConfiguration,
) -> float:
    if usage is None:
        return 0
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    return (
        prompt_tokens * candidate.input_cost_per_million
        + completion_tokens * candidate.output_cost_per_million
    ) / 1_000_000


def _estimate_memory(model: ModelInfo) -> int | None:
    return int(model.size_bytes * 1.25) if model.size_bytes is not None else None


def _memory_fit(estimated: int | None, total: int | None) -> float:
    if estimated is None or total is None:
        return 1
    ratio = estimated / total
    if ratio <= 0.7:
        return 1
    if ratio >= 1:
        return 0
    return (1 - ratio) / 0.3


def _candidate_metrics(
    qualification: CandidateQualification,
    task: TaskType,
    suite: QualificationSuite,
) -> tuple[float, float, float, CandidateQualification]:
    weights = {case.id: case.weight for case in suite.cases if case.task is task}
    cases = [case for case in qualification.cases if case.task is task]
    total_weight = sum(weights[case.case_id] for case in cases)
    passed_weight = sum(weights[case.case_id] for case in cases if case.passed)
    quality = passed_weight / total_weight if total_weight else 0
    latency = sum(case.latency_seconds for case in cases) / len(cases)
    cost = sum(case.estimated_cost_usd for case in cases) / len(cases)
    return quality, latency, cost, qualification


def _routing_profiles(
    recommendations: list[ModelRecommendation],
) -> dict[str, dict[TaskType, AdaptiveModelRoute]]:
    profiles: dict[str, dict[TaskType, AdaptiveModelRoute]] = {}
    for recommendation in recommendations:
        profiles.setdefault(recommendation.profile, {})[recommendation.task] = AdaptiveModelRoute(
            provider=recommendation.provider,
            model=recommendation.model,
            options=recommendation.options,
        )
    return profiles
