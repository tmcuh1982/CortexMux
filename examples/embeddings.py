"""Generate a batch of embeddings."""

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    print(len(mux.embed(["first", "second"], provider="ollama", model="my-embed-model").embeddings))
