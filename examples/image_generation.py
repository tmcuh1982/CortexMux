"""Generate an image from an API-format workflow."""

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    print(
        mux.generate_image(
            "An agricultural sensor",
            workflow="workflow.json",
            bindings={"prompt": {"node_id": "6", "input": "text"}},
        ).images
    )
