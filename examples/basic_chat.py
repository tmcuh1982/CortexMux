"""Basic synchronous chat."""

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    print(mux.chat("Hello", provider="ollama", model="my-model").content)
