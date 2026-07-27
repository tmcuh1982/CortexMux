"""Pydantic-validated structured output."""

from pydantic import BaseModel

from cortexmux import CortexMux


class Summary(BaseModel):
    """Example response contract."""

    name: str
    function: str


with CortexMux.from_env() as mux:
    print(
        mux.structured(
            "Describe an inverter.", provider="ollama", model="my-model", response_model=Summary
        ).parsed
    )
