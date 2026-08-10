"""Deterministic model qualification tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import OutputPathError
from cortexmux.core.registry import ProviderRegistry
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import HealthStatus, ModelInfo, UsageMetadata
from cortexmux.schemas.requests import ChatRequest, CortexRequest, StructuredOutputRequest
from cortexmux.schemas.responses import ChatResponse, CortexResponse, StructuredResponse
from cortexmux.selection import (
    MachineProfile,
    ModelQualifier,
    QualificationSuite,
    load_qualification_suite,
    write_qualification_manifest,
)


class QualificationProvider(BaseProvider):
    """Return deterministic model-dependent outputs and latency."""

    name = "test"

    async def healthcheck(self) -> HealthStatus:
        return HealthStatus(provider=self.name, available=True, message="ok")

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="fast", provider=self.name, size_bytes=100),
            ModelInfo(name="quality", provider=self.name, size_bytes=200),
        ]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset({TaskType.CHAT, TaskType.STRUCTURED_OUTPUT}),
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        return task in {TaskType.CHAT, TaskType.STRUCTURED_OUTPUT}

    async def execute(self, request: CortexRequest) -> CortexResponse:
        await asyncio.sleep(0.001 if request.model == "fast" else 0.01)
        usage = UsageMetadata(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        if isinstance(request, ChatRequest):
            prompt = request.messages[-1].content
            content = "YES" if request.model == "quality" and "second" in prompt else "OK"
            return ChatResponse(
                provider=self.name,
                model=request.model,
                request_id=request.request_id,
                content=content,
                usage=usage,
            )
        if isinstance(request, StructuredOutputRequest):
            return StructuredResponse(
                provider=self.name,
                model=request.model,
                request_id=request.request_id,
                content='{"value": 2}',
                parsed={"value": 2},
                usage=usage,
            )
        raise AssertionError("unexpected request")


@pytest.mark.asyncio
async def test_qualifier_recommends_complete_configurations_by_profile() -> None:
    suite = QualificationSuite.model_validate(
        {
            "minimum_quality": 0.5,
            "candidates": [
                {
                    "id": "fast-config",
                    "provider": "test",
                    "model": "fast",
                    "tasks": ["chat"],
                    "options": {"num_ctx": 2048},
                },
                {
                    "id": "quality-config",
                    "provider": "test",
                    "model": "quality",
                    "tasks": ["chat"],
                    "options": {"num_ctx": 8192},
                },
            ],
            "cases": [
                {
                    "id": "first",
                    "task": "chat",
                    "prompt": "first",
                    "validator": "exact_text",
                    "expected_text": "OK",
                },
                {
                    "id": "second",
                    "task": "chat",
                    "prompt": "second",
                    "validator": "exact_text",
                    "expected_text": "YES",
                },
            ],
        }
    )
    registry = ProviderRegistry()
    registry.register(QualificationProvider())
    machine = MachineProfile(
        system="test",
        release="1",
        architecture="arm64",
        cpu_count=8,
        total_memory_bytes=1_000,
    )

    manifest = await ModelQualifier(registry, machine=machine).qualify(suite)

    recommendations = {(item.profile, item.task): item for item in manifest.recommendations}
    assert recommendations[("fast", TaskType.CHAT)].candidate_id == "fast-config"
    assert recommendations[("balanced", TaskType.CHAT)].candidate_id == "quality-config"
    assert recommendations[("quality", TaskType.CHAT)].candidate_id == "quality-config"
    assert recommendations[("fast", TaskType.CHAT)].options == {"num_ctx": 2048}
    assert manifest.routing_profiles["balanced"][TaskType.CHAT].model == "quality"


@pytest.mark.asyncio
async def test_unavailable_models_remain_auditable_and_are_not_recommended() -> None:
    suite = QualificationSuite.model_validate(
        {
            "candidates": [{"id": "missing", "provider": "test", "model": "missing"}],
            "cases": [
                {
                    "id": "case",
                    "task": "chat",
                    "prompt": "x",
                    "validator": "exact_text",
                    "expected_text": "OK",
                }
            ],
        }
    )
    registry = ProviderRegistry()
    registry.register(QualificationProvider())

    manifest = await ModelQualifier(registry).qualify(suite)

    assert not manifest.candidates[0].available
    assert manifest.candidates[0].error == "Model is not available from this provider."
    assert manifest.recommendations == []


@pytest.mark.asyncio
async def test_structured_validation_separates_format_schema_and_expected_values() -> None:
    suite = QualificationSuite.model_validate(
        {
            "candidates": [{"id": "quality", "provider": "test", "model": "quality"}],
            "cases": [
                {
                    "id": "semantic-mismatch",
                    "task": "structured_output",
                    "prompt": "return a value",
                    "validator": "exact_json",
                    "expected_json": {"value": 3},
                    "json_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "number"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                }
            ],
        }
    )
    registry = ProviderRegistry()
    registry.register(QualificationProvider())

    manifest = await ModelQualifier(registry).qualify(suite)

    result = manifest.candidates[0].cases[0]
    assert result.syntax_valid is True
    assert result.schema_valid is True
    assert result.expected_match is False
    assert result.passed is False


def test_json_suite_and_manifest_round_trip(tmp_path: Path) -> None:
    example = Path("configs/model-qualification.example.json")
    suite = load_qualification_suite(example)
    assert {item.model for item in suite.candidates} >= {"qwen3.6:27b", "gpt-5.6-luna"}

    registry = ProviderRegistry()
    registry.register(QualificationProvider())
    minimal = QualificationSuite.model_validate(
        {
            "candidates": [{"id": "fast", "provider": "test", "model": "fast"}],
            "cases": [
                {
                    "id": "case",
                    "task": "chat",
                    "prompt": "first",
                    "validator": "exact_text",
                    "expected_text": "OK",
                }
            ],
        }
    )
    manifest = asyncio.run(ModelQualifier(registry).qualify(minimal))
    output = tmp_path / "manifest.json"

    assert write_qualification_manifest(manifest, output) == output
    assert '"recommendations"' in output.read_text(encoding="utf-8")
    with pytest.raises(OutputPathError):
        write_qualification_manifest(manifest, output)
