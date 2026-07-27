# ComfyUI workflows

Export a workflow through ComfyUI's **Save (API Format)** option. A normal
browser workflow contains UI metadata and will fail validation.

A binding maps a normalized field to a node input:

```json
{"prompt": {"node_id": "6", "input": "text"}}
```

Configure expected output node IDs to ignore unrelated preview images. Strict
mode rejects supplied fields without bindings; lenient mode records them as
ignored. CortexMux deep-copies graphs before injection.

Common errors are a browser-format graph, missing node/input, missing required
binding, unexpected output node, queue failure, and completion timeout.
Checkpoint is just another binding; CortexMux does not install it.

## Reusable workflow catalog

The directory configured by `providers.comfyui.workflow_dir` is also a
catalog. CortexMux recursively discovers files ending in
`.cortexmux.json`. A manifest keeps the graph, bindings, expected outputs, and
metadata together:

```json
{
  "schema_version": 1,
  "name": "text-to-image",
  "description": "Standard local text-to-image workflow",
  "workflow": "text_to_image.api.json",
  "bindings": "text_to_image.bindings.json",
  "expected_output_nodes": ["9"],
  "metadata": {"category": "image-generation"}
}
```

`workflow` and `bindings` paths are relative to the manifest and must remain
inside the configured catalog root. Graphs and bindings may also be embedded as
JSON objects. Names are unique and may contain letters, numbers, dots, dashes,
and underscores.

```python
with CortexMux.from_env() as mux:
    for workflow in mux.list_workflows():
        print(workflow.name, workflow.binding_fields)

    result = mux.generate_image(
        prompt="A small robot tending a greenhouse",
        workflow="text-to-image",
        checkpoint="model.safetensors",
    )
```

Request-level bindings and output node IDs override the defaults from the
catalog entry. Existing direct graph paths and inline graph dictionaries remain
supported.

## Typed progress events

Pass `on_progress` to image generation to receive normalized queue, execution,
node, download, completion, and error events. The callback can be synchronous
or asynchronous:

```python
from cortexmux.schemas import ProgressEvent


def show_progress(event: ProgressEvent) -> None:
    percent = f"{event.progress:.0%}" if event.progress is not None else ""
    print(event.stage.value, event.node_id or "", percent)


result = mux.generate_image(
    prompt="A small robot tending a greenhouse",
    workflow="text-to-image",
    checkpoint="model.safetensors",
    on_progress=show_progress,
)
```

Events include the CortexMux request ID and, when known, the ComfyUI prompt ID,
client ID, node ID, current/total values, normalized progress from 0 to 1, queue
length, raw ComfyUI event type, and a timestamp. WebSocket monitoring remains
optional; history polling is still the completion authority.
