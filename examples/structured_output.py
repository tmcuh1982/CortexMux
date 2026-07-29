"""JSON Schema-validated structured output with an Ollama reasoning model."""

from cortexmux import CortexMux

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "function": {"type": "string"},
    },
    "required": ["name", "function"],
}


with CortexMux.from_env() as mux:
    print(
        mux.structured(
            prompt="Describe an inverter.",
            json_schema=schema,
            provider="ollama",
            model="qwen3:4b",
            think=False,
        ).parsed
    )
