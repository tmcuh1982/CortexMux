"""Analyze a local image."""

from pathlib import Path

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    print(
        mux.vision(
            Path("equipment.jpg"),
            prompt="Describe the components.",
            provider="ollama",
            model="my-vision-model",
        ).content
    )
