"""Public facade convenience API tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from cortexmux import CortexMux
from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import InvalidRequestError
from cortexmux.core.types import ProgressStage, TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.ollama import OllamaClient, OllamaProvider
from cortexmux.schemas.calculations import VerificationStatus
from cortexmux.schemas.common import HealthStatus, ImageArtifact, ModelInfo
from cortexmux.schemas.progress import ProgressCallback, ProgressEvent, emit_progress
from cortexmux.schemas.requests import CortexRequest
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    DataAnalysisResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    StructuredResponse,
    TextResponse,
    VisionResponse,
)


class MultiProvider(BaseProvider):
    """Return a correctly typed response for every public task."""

    name = "multi"

    async def healthcheck(self) -> HealthStatus:
        return HealthStatus(provider=self.name, available=True, message="ok")

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="model", provider=self.name)]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset(TaskType),
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        return True

    async def execute(self, request: CortexRequest) -> CortexResponse:
        common: dict[str, Any] = {
            "provider": self.name,
            "model": request.model,
            "request_id": request.request_id,
        }
        if request.task is TaskType.TEXT_GENERATION:
            return TextResponse(**common, content="text")
        if request.task is TaskType.CHAT:
            return ChatResponse(**common, content="chat")
        if request.task is TaskType.STRUCTURED_OUTPUT:
            return StructuredResponse(**common, content='{"value":1}', parsed={"value": 1})
        if request.task is TaskType.VISION:
            return VisionResponse(**common, content="vision")
        if request.task is TaskType.EMBEDDING:
            return EmbeddingResponse(**common, embeddings=[[1.0]])
        if request.task is TaskType.IMAGE_GENERATION:
            return ImageGenerationResponse(
                **common,
                prompt_id="p",
                images=[
                    ImageArtifact(
                        path="/tmp/image.png",
                        filename="image.png",
                        node_id="9",
                        sequence=1,
                    )
                ],
            )
        return DataAnalysisResponse(
            **common,
            engine="pandas",
            profile={},
            report_markdown="# Report\n",
        )

    async def execute_with_progress(
        self,
        request: CortexRequest,
        on_progress: ProgressCallback,
    ) -> CortexResponse:
        await emit_progress(
            on_progress,
            ProgressEvent(
                request_id=request.request_id,
                provider=self.name,
                stage=ProgressStage.COMPLETED,
                progress=1,
            ),
        )
        return await self.execute(request)


def configured_mux() -> CortexMux:
    """Return a facade containing only the multi-task test provider."""
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    mux.register_provider(MultiProvider())
    return mux


def test_sync_facade_and_generic_validation() -> None:
    with configured_mux() as mux:
        assert mux.generate("hello", provider="multi", model="model").content == "text"
        assert mux.list_models("multi")[0].name == "model"
        assert mux.health()[0].available
        with pytest.raises(InvalidRequestError):
            mux.run("unknown", prompt="x")
        verification = mux.verify_calculation(
            {
                "label": "Total",
                "operation": "sum",
                "operands": [
                    {"source_path": "/results/0/value"},
                    {"literal": 2},
                ],
                "claimed_result": 5,
            },
            {"profile": {}, "results": [{"value": 3}]},
        )
        assert verification.status is VerificationStatus.VERIFIED
        assert verification.expected_result == Decimal("5")


@pytest.mark.asyncio
async def test_all_async_convenience_methods(tmp_path: Path) -> None:
    async with configured_mux() as mux:
        assert (await mux.agenerate("x", provider="multi", model="model")).content == "text"
        assert (await mux.achat("x", provider="multi", model="model")).content == "chat"
        assert (
            await mux.astructured(
                "x",
                provider="multi",
                model="model",
                json_schema={"type": "object"},
            )
        ).parsed == {"value": 1}
        assert (
            await mux.avision(b"image", prompt="x", provider="multi", model="model")
        ).content == "vision"
        assert (await mux.aembed(["x"], provider="multi", model="model")).embeddings == [[1.0]]
        progress: list[ProgressEvent] = []
        assert (
            await mux.agenerate_image(
                "x",
                provider="multi",
                model="model",
                workflow={"1": {"class_type": "X", "inputs": {}}},
                on_progress=progress.append,
            )
        ).prompt_id == "p"
        assert progress[0].stage is ProgressStage.COMPLETED
        assert (
            await mux.aanalyze_data(
                tmp_path / "unused.csv",
                provider="multi",
            )
        ).engine == "pandas"
        assert (await mux.alist_models("multi"))[0].name == "model"
        assert (await mux.ahealth())[0].available


def test_from_env_with_network_providers_disabled(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[providers.ollama]\nenabled=false\n[providers.comfyui]\nenabled=false\n",
        encoding="utf-8",
    )
    with CortexMux.from_env(config_path=config) as mux:
        assert [provider.name for provider in mux.registry.list()] == ["data"]


def test_builtin_provider_construction_and_cleanup(tmp_path: Path) -> None:
    config = CortexMuxConfig.model_validate({"core": {"output_dir": tmp_path}})
    with CortexMux(config) as mux:
        assert [provider.name for provider in mux.registry.list()] == [
            "comfyui",
            "data",
            "ollama",
        ]


def test_sync_network_calls_share_one_event_loop() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b"}]})

    client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(handler),
    )
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    mux.register_provider(
        OllamaProvider(OllamaClient("http://localhost:11434", timeout=2, client=client))
    )
    with mux:
        assert mux.health()[0].available
        assert mux.list_models("ollama")[0].name == "qwen2.5-coder:7b"
