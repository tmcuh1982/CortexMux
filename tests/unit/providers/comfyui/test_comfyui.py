"""ComfyUI workflow and provider tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from cortexmux.core.exceptions import ProviderResponseError, WorkflowValidationError
from cortexmux.core.types import ProgressStage
from cortexmux.providers.comfyui.catalog import WorkflowCatalog
from cortexmux.providers.comfyui.client import ComfyUIClient
from cortexmux.providers.comfyui.provider import ComfyUIProvider
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition
from cortexmux.schemas.progress import ProgressCallback, ProgressEvent, emit_progress
from cortexmux.schemas.requests import ImageGenerationRequest


def test_workflow_deep_copy_binding_and_validation(api_graph: dict[str, Any]) -> None:
    original = copy.deepcopy(api_graph)
    definition = WorkflowDefinition(
        name="test",
        workflow=api_graph,
        input_bindings={"prompt": InputBinding(node_id="6", input="text")},
        expected_output_nodes=["9"],
    )
    bound, ignored = definition.bind({"prompt": "new"}, strict=True)
    assert bound["6"]["inputs"]["text"] == "new"
    assert api_graph == original
    assert ignored == []
    with pytest.raises(WorkflowValidationError):
        definition.bind({"seed": 1}, strict=True)
    _, ignored = definition.bind({"seed": 1}, strict=False)
    assert ignored == ["seed"]


def test_invalid_workflow_shapes(tmp_path: Path, api_graph: dict[str, Any]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    with pytest.raises(WorkflowValidationError):
        WorkflowDefinition(name="bad", workflow=bad).load_graph()
    with pytest.raises(WorkflowValidationError):
        WorkflowDefinition(name="bad", workflow={"1": {"inputs": {}}}).load_graph()
    with pytest.raises(WorkflowValidationError):
        WorkflowDefinition(
            name="bad",
            workflow=api_graph,
            input_bindings={"x": InputBinding(node_id="missing", input="x")},
        ).load_graph()


class FakeComfyClient:
    """In-memory ComfyUI transport."""

    graph: dict[str, Any] | None = None
    closed = False

    async def get_json(self, path: str, *, request_id: str | None = None) -> Any:
        if path == "/system_stats":
            return {"system": {"comfyui_version": "1"}}
        return ["checkpoint.safetensors"]

    async def submit(self, graph: dict[str, Any], *, client_id: str, request_id: str) -> str:
        self.graph = graph
        return "prompt-1"

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        client_id: str,
        request_id: str,
        timeout: float,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        await emit_progress(
            on_progress,
            ProgressEvent(
                request_id=request_id,
                provider="comfyui",
                stage=ProgressStage.NODE_PROGRESS,
                prompt_id=prompt_id,
                client_id=client_id,
                node_id="3",
                current=1,
                total=2,
                progress=0.5,
            ),
        )
        return {
            "outputs": {
                "9": {
                    "images": [{"filename": "../../unsafe.png", "subfolder": "", "type": "output"}]
                }
            }
        }

    async def download_image(self, **kwargs: Any) -> bytes:
        return b"png"

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_provider_submission_and_safe_download(
    tmp_path: Path, api_graph: dict[str, Any]
) -> None:
    client = FakeComfyClient()
    provider = ComfyUIProvider(
        client,  # type: ignore[arg-type]
        workflow_dir=tmp_path,
        output_dir=tmp_path / "out",
        model_folders=["checkpoints"],
        timeout=5,
    )
    assert (await provider.healthcheck()).available
    assert (await provider.list_models())[0].name == "checkpoint.safetensors"
    response = await provider.execute(
        ImageGenerationRequest(
            provider="comfyui",
            prompt="new",
            workflow=api_graph,
            bindings={"prompt": {"node_id": "6", "input": "text"}},
            expected_output_nodes=["9"],
        )
    )
    assert client.graph is not None and client.graph["6"]["inputs"]["text"] == "new"
    artifact = response.images[0]
    assert Path(artifact.path).is_relative_to((tmp_path / "out").resolve())
    assert Path(artifact.path).read_bytes() == b"png"
    await provider.close()
    assert client.closed


@pytest.mark.asyncio
async def test_provider_reports_typed_progress(tmp_path: Path, api_graph: dict[str, Any]) -> None:
    client = FakeComfyClient()
    provider = ComfyUIProvider(
        client,  # type: ignore[arg-type]
        workflow_dir=tmp_path,
        output_dir=tmp_path / "out",
        model_folders=[],
        timeout=5,
    )
    events: list[ProgressEvent] = []
    await provider.execute_with_progress(
        ImageGenerationRequest(
            prompt="new",
            workflow=api_graph,
            bindings={"prompt": {"node_id": "6", "input": "text"}},
            expected_output_nodes=["9"],
        ),
        events.append,
    )
    assert [event.stage for event in events] == [
        ProgressStage.SUBMITTING,
        ProgressStage.QUEUED,
        ProgressStage.NODE_PROGRESS,
        ProgressStage.DOWNLOADING,
        ProgressStage.COMPLETED,
    ]
    assert events[2].progress == 0.5
    assert events[-1].prompt_id == "prompt-1"


def test_normalize_comfyui_progress_events() -> None:
    event = ComfyUIClient.normalize_progress_event(
        {
            "type": "progress",
            "data": {"prompt_id": "p", "node": "3", "value": 4, "max": 10},
        },
        request_id="r",
        prompt_id="p",
        client_id="c",
    )
    assert event is not None
    assert event.stage is ProgressStage.NODE_PROGRESS
    assert event.node_id == "3"
    assert event.progress == 0.4
    status = ComfyUIClient.normalize_progress_event(
        {
            "type": "status",
            "data": {"status": {"exec_info": {"queue_remaining": 2}}},
        },
        request_id="r",
        prompt_id="p",
        client_id="c",
    )
    assert status is not None and status.queue_remaining == 2
    assert (
        ComfyUIClient.normalize_progress_event(
            {"type": "unknown", "data": {}},
            request_id="r",
            prompt_id="p",
            client_id="c",
        )
        is None
    )


def test_workflow_catalog_loads_named_manifest(tmp_path: Path, api_graph: dict[str, Any]) -> None:
    workflow = tmp_path / "graph.json"
    workflow.write_text(json.dumps(api_graph), encoding="utf-8")
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps({"prompt": {"node_id": "6", "input": "text"}}),
        encoding="utf-8",
    )
    manifest = tmp_path / "text-to-image.cortexmux.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "text-to-image",
                "description": "Reusable test workflow",
                "workflow": "graph.json",
                "bindings": "bindings.json",
                "expected_output_nodes": ["9"],
            }
        ),
        encoding="utf-8",
    )

    catalog = WorkflowCatalog(tmp_path)
    assert catalog.list()[0].binding_fields == ["prompt"]
    definition = catalog.get("text-to-image")
    bound, ignored = definition.bind({"prompt": "catalog prompt"})
    assert bound["6"]["inputs"]["text"] == "catalog prompt"
    assert ignored == []


@pytest.mark.asyncio
async def test_provider_executes_catalog_workflow_by_name(
    tmp_path: Path, api_graph: dict[str, Any]
) -> None:
    (tmp_path / "graph.json").write_text(json.dumps(api_graph), encoding="utf-8")
    (tmp_path / "catalog.cortexmux.json").write_text(
        json.dumps(
            {
                "name": "catalog-image",
                "workflow": "graph.json",
                "bindings": {"prompt": {"node_id": "6", "input": "text"}},
                "expected_output_nodes": ["9"],
            }
        ),
        encoding="utf-8",
    )
    client = FakeComfyClient()
    provider = ComfyUIProvider(
        client,  # type: ignore[arg-type]
        workflow_dir=tmp_path,
        output_dir=tmp_path / "out",
        model_folders=[],
        timeout=5,
    )
    response = await provider.execute(
        ImageGenerationRequest(prompt="from catalog", workflow="catalog-image")
    )
    assert response.raw_metadata == {"workflow": "catalog-image"}
    assert client.graph is not None
    assert client.graph["6"]["inputs"]["text"] == "from catalog"


def test_workflow_catalog_rejects_escape_and_duplicates(
    tmp_path: Path, api_graph: dict[str, Any]
) -> None:
    outside = tmp_path.parent / "outside-workflow.json"
    outside.write_text(json.dumps(api_graph), encoding="utf-8")
    (tmp_path / "escape.cortexmux.json").write_text(
        json.dumps({"name": "escape", "workflow": "../outside-workflow.json"}),
        encoding="utf-8",
    )
    with pytest.raises(WorkflowValidationError):
        WorkflowCatalog(tmp_path)

    (tmp_path / "escape.cortexmux.json").unlink()
    workflow = tmp_path / "graph.json"
    workflow.write_text(json.dumps(api_graph), encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    for parent in (tmp_path, nested):
        (parent / f"{parent.name}.cortexmux.json").write_text(
            json.dumps({"name": "duplicate", "workflow": str(workflow)}),
            encoding="utf-8",
        )
    with pytest.raises(WorkflowValidationError):
        WorkflowCatalog(tmp_path)


@pytest.mark.asyncio
async def test_http_client_queue_history_and_download() -> None:
    history_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal history_calls
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        if request.url.path == "/history/p1":
            history_calls += 1
            return httpx.Response(
                200,
                json=({} if history_calls == 1 else {"p1": {"outputs": {"9": {"images": []}}}}),
            )
        if request.url.path == "/view":
            assert request.url.params["filename"] == "image.png"
            return httpx.Response(200, content=b"image")
        return httpx.Response(200, json={"ok": True})

    http_client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8188", transport=httpx.MockTransport(handler)
    )
    client = ComfyUIClient("http://127.0.0.1:8188", timeout=2, client=http_client)
    assert await client.get_json("/") == {"ok": True}
    prompt_id = await client.submit({"1": {}}, client_id="c", request_id="r")
    assert prompt_id == "p1"
    entry = await client.wait_for_history(prompt_id, request_id="r", timeout=2)
    assert "outputs" in entry
    image = await client.download_image(
        filename="image.png",
        subfolder="",
        image_type="output",
        request_id="r",
    )
    assert image == b"image"
    assert client._websocket_url("id") == "ws://127.0.0.1:8188/ws?clientId=id"
    await client.close()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_http_client_rejects_bad_json_and_queue_response() -> None:
    responses = iter(
        [
            httpx.Response(200, content=b"not-json"),
            httpx.Response(200, json={}),
            httpx.Response(500, json={"error": "bad"}),
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    http_client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8188", transport=httpx.MockTransport(handler)
    )
    client = ComfyUIClient("http://127.0.0.1:8188", timeout=2, client=http_client)
    with pytest.raises(ProviderResponseError):
        await client.get_json("/")
    with pytest.raises(ProviderResponseError):
        await client.submit({}, client_id="c", request_id="r")
    with pytest.raises(ProviderResponseError):
        await client.get_json("/")
    await http_client.aclose()
