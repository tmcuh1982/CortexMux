# Example workflows

`text_to_image.example.json` is a complete, generic ComfyUI API-format
text-to-image graph using only standard nodes. The companion
`text_to_image.bindings.example.json` maps CortexMux fields to graph inputs.
`text_to_image.cortexmux.json` registers both files as the reusable catalog
workflow named `text-to-image`.

The demo script replaces `model.safetensors` with the first checkpoint reported
by ComfyUI, or with the value passed through `--checkpoint`. The checkpoint
must already be installed; CortexMux does not download model weights.

Point `providers.comfyui.workflow_dir` at this directory, then inspect or run
the entry without repeating bindings:

```bash
cortexmux workflows list
cortexmux image generate --workflow text-to-image \
  --prompt "A small robot tending a greenhouse" \
  --checkpoint model.safetensors
```
