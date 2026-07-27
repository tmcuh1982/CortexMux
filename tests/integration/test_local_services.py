"""Opt-in local-service integration smoke tests."""

from __future__ import annotations

import os

import pytest

from cortexmux import CortexMux

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("CORTEXMUX_RUN_INTEGRATION_TESTS") != "1",
    reason="set CORTEXMUX_RUN_INTEGRATION_TESTS=1 to enable local network tests",
)
def test_local_provider_health() -> None:
    """Check configured local providers without requiring models or a GPU."""
    with CortexMux.from_env() as mux:
        assert mux.health()
