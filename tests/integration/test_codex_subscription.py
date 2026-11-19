"""Explicitly opted-in live subscription test; never starts login itself."""

import os
from pathlib import Path

import pytest

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.providers.codex import CodexConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("CORTEXMUX_CODEX_INTEGRATION") != "1",
        reason="Codex live test requires explicit opt-in",
    ),
]


async def test_connected_subscription():
    config = CortexMuxConfig()
    config.providers.ollama.enabled = False
    config.providers.comfyui.enabled = False
    config.providers.codex = CodexConfig(
        enabled=True,
        executable=os.getenv("CORTEXMUX_CODEX_EXECUTABLE", "codex"),
        auth_directory=Path(os.environ["CORTEXMUX_CODEX_AUTH_DIRECTORY"]),
    )
    async with CortexMux(config) as mux:
        assert (await mux.acodex_account()).connected
        model = os.environ["CORTEXMUX_CODEX_MODEL"]
        assert any(candidate.name == model for candidate in await mux.alist_models("codex"))
        response = await mux.astructured(
            "Return an object with answer equal to 42.",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            provider="codex",
            model=model,
            require_no_tools=False,
        )
        assert response.model == model
        assert response.parsed == {"answer": 42}
