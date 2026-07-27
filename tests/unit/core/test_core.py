"""Core registry, routing, configuration, and security tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ModelSelectionError,
    OutputPathError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    RemoteHostNotAllowedError,
)
from cortexmux.core.registry import ProviderRegistry
from cortexmux.core.router import Router
from cortexmux.core.security import safe_output_path, validate_provider_url
from cortexmux.core.types import TaskType
from cortexmux.facade import CortexMux
from cortexmux.schemas.requests import TextGenerationRequest
from tests.conftest import StubProvider


def test_registry_lifecycle(stub_provider: StubProvider) -> None:
    registry = ProviderRegistry()
    registry.register(stub_provider)
    assert registry.get("stub") is stub_provider
    assert registry.list() == [stub_provider]
    with pytest.raises(ProviderRegistrationError):
        registry.register(stub_provider)
    replacement = StubProvider("stub")
    registry.register(replacement, replace=True)
    assert registry.unregister("stub") is replacement
    with pytest.raises(ProviderNotFoundError):
        registry.get("stub")


@pytest.mark.asyncio
async def test_registry_health_and_cleanup(stub_provider: StubProvider) -> None:
    registry = ProviderRegistry()
    registry.register(stub_provider)
    assert (await registry.health())[0].available
    await registry.close()
    assert stub_provider.closed


@pytest.mark.asyncio
async def test_explicit_and_default_routing() -> None:
    config = CortexMuxConfig.model_validate(
        {
            "providers": {"ollama": {"defaults": {"text_generation": "default"}}},
            "routing": {"defaults": {"text_generation_provider": "ollama"}},
        }
    )
    registry = ProviderRegistry()
    one = StubProvider("ollama", models={"default", "chosen"})
    registry.register(one)
    router = Router(registry, config)
    explicit = await router.route(
        TextGenerationRequest(provider="ollama", model="chosen", prompt="hello")
    )
    assert explicit.routing is not None
    assert explicit.routing.routing_reason == "explicit_provider_and_model"
    default = await router.route(TextGenerationRequest(prompt="hello"))
    assert default.model == "default"
    assert default.routing is not None
    assert default.routing.routing_reason == "configured_task_default"


def test_explicit_provider_missing_default_fails() -> None:
    registry = ProviderRegistry()
    registry.register(StubProvider("one"))
    router = Router(registry, CortexMuxConfig())
    with pytest.raises(ModelSelectionError):
        router.select(TextGenerationRequest(provider="one", prompt="hello"))


def test_unique_and_ambiguous_model_routing() -> None:
    registry = ProviderRegistry()
    registry.register(StubProvider("one", models={"shared", "unique"}))
    registry.register(StubProvider("two", models={"shared"}))
    router = Router(registry, CortexMuxConfig())
    provider, model, reason = router.select(TextGenerationRequest(model="unique", prompt="hello"))
    assert (provider.name, model, reason) == (
        "one",
        "unique",
        "explicit_model_unique_provider",
    )
    with pytest.raises(ModelSelectionError):
        router.select(TextGenerationRequest(model="shared", prompt="hello"))


def test_configuration_precedence_and_empty_models(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[core]\noutput_dir="toml"\n[providers.ollama.defaults]\nchat="toml-model"\n',
        encoding="utf-8",
    )
    config = CortexMuxConfig.load(
        path=path,
        environ={
            "CORTEXMUX_OUTPUT_DIR": "env",
            "CORTEXMUX_DEFAULT_CHAT_MODEL": "",
        },
        overrides={"core": {"output_dir": "python"}},
    )
    assert config.core.output_dir == Path("python")
    assert config.providers.ollama.defaults.chat is None
    assert config.config_path == path


def test_network_policy_and_safe_paths(tmp_path: Path) -> None:
    assert validate_provider_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert validate_provider_url("http://[::1]:8188") == "http://[::1]:8188"
    with pytest.raises(RemoteHostNotAllowedError):
        validate_provider_url("https://example.com")
    assert (
        validate_provider_url("https://example.com", approved_hosts={"example.com"})
        == "https://example.com"
    )
    path = safe_output_path(tmp_path, "result.txt")
    path.write_text("x", encoding="utf-8")
    with pytest.raises(OutputPathError):
        safe_output_path(tmp_path, "result.txt")
    with pytest.raises(OutputPathError):
        safe_output_path(tmp_path, "../escape.txt")


@pytest.mark.asyncio
async def test_sync_api_rejects_running_loop() -> None:
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    with pytest.raises(InvalidRequestError):
        mux.run(TaskType.TEXT_GENERATION, prompt="hello")
    await mux.aclose()


def test_exception_serialization() -> None:
    error = ModelSelectionError("ambiguous", task="chat")
    assert error.to_dict() == {
        "error": "ModelSelectionError",
        "message": "ambiguous",
        "context": {"task": "chat"},
    }
