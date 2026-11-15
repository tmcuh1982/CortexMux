"""Explicit deployment probes: passing syntax is not proof of measured budgets."""

import os

import pytest

from cortexmux import CortexMux
from cortexmux.core.exceptions import CortexMuxError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("CORTEXMUX_BEHAVIOR_LIVE_CONFIG") is None,
        reason="Set CORTEXMUX_BEHAVIOR_LIVE_CONFIG to an approved deployment config",
    ),
]


def test_live_structured_contract_and_cap() -> None:
    """Require actual usage to attest the measured completion ceiling."""
    with CortexMux.from_env(config_path=os.environ["CORTEXMUX_BEHAVIOR_LIVE_CONFIG"]) as mux:
        result = mux.structured(
            'Return {"answer":"ok"}.',
            max_tokens=512,
            json_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"const": "ok"}},
            },
        )
        assert result.raw_metadata["behavior"]["finish_reason"] == "stop"
        if result.usage is None or result.usage.completion_tokens is None:
            pytest.skip("No completion counter: cap enforcement remains unqualified")
        assert result.usage.completion_tokens <= 512


def test_live_truncation() -> None:
    """Verify finish refusal on an intentionally insufficient real output budget."""
    with CortexMux.from_env(config_path=os.environ["CORTEXMUX_BEHAVIOR_LIVE_CONFIG"]) as mux:
        with pytest.raises(CortexMuxError) as caught:
            mux.structured(
                "Return a JSON object with an array of integers from 1 to 200.", max_tokens=1
            )
        assert caught.value.context["behavior"]["category"] == "truncation"


@pytest.mark.parametrize("think", [False, True])
def test_live_reasoning_controls(think: bool) -> None:
    """Probe native reasoning intent; do not infer measured effort from valid JSON."""
    with CortexMux.from_env(config_path=os.environ["CORTEXMUX_BEHAVIOR_LIVE_CONFIG"]) as mux:
        result = mux.structured(
            'Compute 17 times 19 and return {"answer":323}.',
            think=think,
            max_tokens=512,
            json_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"const": 323}},
            },
        )
        diagnostic = result.raw_metadata["behavior"]
        assert diagnostic["requested_parameters"]["think"] is think
        assert diagnostic["finish_reason"] == "stop"


def test_live_mlx_reasoning_budget() -> None:
    """Require the actual reasoning count before attesting a 128-token budget."""
    with CortexMux.from_env(config_path=os.environ["CORTEXMUX_BEHAVIOR_LIVE_CONFIG"]) as mux:
        provider = mux.config.routing.defaults.structured_output_provider
        route_profile = mux.config.routing.profile_for()
        if route_profile and route_profile.structured_output:
            provider = route_profile.structured_output.provider
        if provider not in {"mlx", "mlx_lm", "mlx_vlm"}:
            pytest.skip("This deployment does not expose an MLX token reasoning budget")
        result = mux.structured(
            'Compute 17 times 19 and return {"answer":323}.',
            think=True,
            thinking_budget=128,
            max_tokens=512,
            json_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"const": 323}},
            },
        )
        reasoning = result.raw_metadata["behavior"]["reasoning_tokens"]
        if reasoning is None:
            pytest.skip("No reasoning counter: budget enforcement remains unqualified")
        assert reasoning <= 128
