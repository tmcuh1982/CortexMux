"""ComfyUI image-generation provider."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    CortexMuxError,
    ProviderResponseError,
    UnsupportedTaskError,
    WorkflowValidationError,
)
from cortexmux.core.security import safe_output_path
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.comfyui.client import ComfyUIClient
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition
from cortexmux.schemas.common import HealthStatus, ImageArtifact, ModelInfo
from cortexmux.schemas.requests import CortexRequest, ImageGenerationRequest
from cortexmux.schemas.responses import CortexResponse, ImageGenerationResponse


class ComfyUIProvider(BaseProvider):
    """Run caller-configured ComfyUI API workflows and download safe outputs."""

    name = "comfyui"

    def __init__(
        self,
        client: ComfyUIClient,
        *,
        workflow_dir: Path,
        output_dir: Path,
        model_folders: list[str],
        timeout: float,
    ) -> None:
        self.client = client
        self.workflow_dir = workflow_dir
        self.output_dir = output_dir
        self.model_folders = model_folders
        self.timeout = timeout

    async def healthcheck(self) -> HealthStatus:
        """Check ComfyUI system stats without raising for normal downtime."""
        try:
            data = await self.client.get_json(ComfyUIClient.SYSTEM_STATS)
            version = None
            if isinstance(data, dict):
                system = data.get("system")
                if isinstance(system, dict) and system.get("comfyui_version"):
                    version = str(system["comfyui_version"])
            return HealthStatus(
                provider=self.name,
                available=True,
                message="ComfyUI is available.",
                version=version,
            )
        except CortexMuxError as exc:
            return HealthStatus(provider=self.name, available=False, message=exc.message)

    async def list_models(self) -> list[ModelInfo]:
        """List configured ComfyUI model folders."""
        models: list[ModelInfo] = []
        for folder in self.model_folders:
            data = await self.client.get_json(f"{ComfyUIClient.MODELS}/{folder}")
            names = (
                data
                if isinstance(data, list)
                else data.get(folder, [])
                if isinstance(data, dict)
                else []
            )
            if not isinstance(names, list):
                continue
            models.extend(
                ModelInfo(name=name, provider=self.name, family=folder)
                for name in names
                if isinstance(name, str)
            )
        return models

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe ComfyUI image workflow capability."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset({TaskType.IMAGE_GENERATION}),
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether the request is image generation."""
        return task is TaskType.IMAGE_GENERATION

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Bind, submit, wait for, and download a ComfyUI workflow."""
        if not isinstance(request, ImageGenerationRequest):
            raise UnsupportedTaskError("ComfyUI supports only image generation.")
        definition = self._definition(request)
        values: dict[str, str | int | float | bool | None] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "checkpoint": request.checkpoint or request.model,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            **request.extra_inputs,
        }
        graph, ignored = definition.bind(values, strict=request.strict_bindings)
        client_id = str(uuid4())
        prompt_id = await self.client.submit(
            graph, client_id=client_id, request_id=request.request_id
        )
        history = await self.client.wait_for_completion(
            prompt_id,
            client_id=client_id,
            request_id=request.request_id,
            timeout=request.timeout or self.timeout,
        )
        images = await self._download_outputs(request, history)
        return ImageGenerationResponse(
            provider=self.name,
            model=request.model or request.checkpoint,
            request_id=request.request_id,
            images=images,
            prompt_id=prompt_id,
            raw_metadata={"ignored_bindings": ignored} if ignored else None,
        )

    def _definition(self, request: ImageGenerationRequest) -> WorkflowDefinition:
        workflow: Path | dict[str, Any]
        if isinstance(request.workflow, dict):
            workflow = request.workflow
            name = "inline"
        else:
            candidate = Path(request.workflow).expanduser()
            if not candidate.exists():
                candidate = self.workflow_dir / (
                    candidate.name if candidate.suffix else f"{candidate.name}.json"
                )
            workflow = candidate
            name = candidate.stem
        bindings: dict[str, InputBinding] = {}
        try:
            for key, value in request.bindings.items():
                bindings[key] = InputBinding.model_validate(value)
        except ValueError as exc:
            raise WorkflowValidationError("Workflow bindings are invalid.") from exc
        return WorkflowDefinition(
            name=name,
            workflow=workflow,
            input_bindings=bindings,
            expected_output_nodes=request.expected_output_nodes,
        )

    async def _download_outputs(
        self, request: ImageGenerationRequest, history: dict[str, Any]
    ) -> list[ImageArtifact]:
        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            raise ProviderResponseError(
                "ComfyUI history has no outputs.", request_id=request.request_id
            )
        root = (request.output_dir or self.output_dir).expanduser().resolve()
        artifacts: list[ImageArtifact] = []
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        allowed_nodes = set(request.expected_output_nodes)
        for node_id, output in outputs.items():
            if allowed_nodes and node_id not in allowed_nodes:
                continue
            image_items = output.get("images", []) if isinstance(output, dict) else []
            if not isinstance(image_items, list):
                continue
            for sequence, item in enumerate(image_items, start=1):
                if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
                    continue
                content = await self.client.download_image(
                    filename=item["filename"],
                    subfolder=str(item.get("subfolder", "")),
                    image_type=str(item.get("type", "output")),
                    request_id=request.request_id,
                )
                suffix = Path(item["filename"]).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    suffix = ".png"
                filename = (
                    f"{timestamp}_{request.request_id[:8]}_{str(node_id)[:24]}_{sequence}{suffix}"
                )
                destination = safe_output_path(root, filename)
                destination.write_bytes(content)
                artifacts.append(
                    ImageArtifact(
                        path=str(destination),
                        filename=filename,
                        node_id=str(node_id),
                        sequence=sequence,
                        size_bytes=len(content),
                    )
                )
        if not artifacts:
            raise ProviderResponseError(
                "ComfyUI completed without discoverable image outputs.",
                request_id=request.request_id,
            )
        return artifacts

    async def close(self) -> None:
        """Close the ComfyUI HTTP client."""
        await self.client.close()
