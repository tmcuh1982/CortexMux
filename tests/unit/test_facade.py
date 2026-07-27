"""Public facade convenience API tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cortexmux import CortexMux
from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import InvalidRequestError
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import HealthStatus, ImageArtifact, ModelInfo
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
        assert (
            await mux.agenerate_image(
                "x",
                provider="multi",
                model="model",
                workflow={"1": {"class_type": "X", "inputs": {}}},
            )
        ).prompt_id == "p"
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
