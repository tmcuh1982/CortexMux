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

