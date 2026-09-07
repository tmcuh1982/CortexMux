"""Core registry, routing, configuration, and security tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    ModelSelectionError,
    OutputPathError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    RemoteHostNotAllowedError,
)
from cortexmux.core.json_schema import validate_json_schema
from cortexmux.core.registry import ProviderRegistry
from cortexmux.core.router import Router
from cortexmux.core.security import safe_output_path, validate_provider_url, validate_web_url
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


@pytest.mark.asyncio
async def test_project_model_profiles_select_installed_models() -> None:
    config = CortexMuxConfig.model_validate(
        {
            "routing": {
                "active_profile": "fast",
                "profiles": {
                    "fast": {
                        "text_generation": {
                            "provider": "ollama",
                            "model": "fast-model",
                            "options": {"num_ctx": 8192, "temperature": 0.2},
                        }
                    },
                    "balanced": {
                        "text_generation": {
                            "provider": "ollama",
                            "model": "balanced-model",
                        }
                    },
                },
            }
        }
    )
    registry = ProviderRegistry()
    registry.register(StubProvider("ollama", models={"fast-model", "balanced-model"}))
    router = Router(registry, config)

    default = await router.route(TextGenerationRequest(prompt="hello"))
    selected = await router.route(
        TextGenerationRequest(
            prompt="hello",
            model_profile="balanced",
            options={"temperature": 0.1},
        )
    )

    assert default.model == "fast-model"
    assert default.routing is not None
    assert default.routing.routing_reason == "configured_model_profile"
    assert registry.get("ollama").requests[0].options == {
        "num_ctx": 8192,
        "temperature": 0.2,
    }
    assert selected.model == "balanced-model"
    assert registry.get("ollama").requests[1].options == {"temperature": 0.1}


@pytest.mark.asyncio
async def test_router_rejects_uninstalled_model_before_execution() -> None:
    registry = ProviderRegistry()
    provider = StubProvider("ollama", models=None)
    registry.register(provider)
    router = Router(registry, CortexMuxConfig())

    with pytest.raises(ModelNotFoundError) as exc_info:
        await router.route(
            TextGenerationRequest(provider="ollama", model="not-installed", prompt="hello")
        )

    assert exc_info.value.context["model"] == "not-installed"
    assert provider.requests == []


@pytest.mark.asyncio
async def test_router_can_disable_installed_model_validation() -> None:
    config = CortexMuxConfig.model_validate({"routing": {"validate_model_availability": False}})
    registry = ProviderRegistry()
    provider = StubProvider("ollama", models=None)
    registry.register(provider)

    response = await Router(registry, config).route(
        TextGenerationRequest(provider="ollama", model="not-listed", prompt="hello")
    )

    assert response.model == "not-listed"
    assert len(provider.requests) == 1


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


def test_example_model_profiles_use_exact_installed_tags() -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "cortexmux.example.toml"
    config = CortexMuxConfig.load(path=config_path)
    fast = config.routing.profile_for("fast")
    balanced = config.routing.profile_for("balanced")
    quality = config.routing.profile_for("quality")

    assert fast is not None and fast.embedding is not None
    assert balanced is not None and balanced.embedding is not None
    assert quality is not None and quality.chat is not None
    assert fast.embedding.model == "nomic-embed-text:latest"
    assert balanced.embedding.model == "nomic-embed-text:latest"
    assert quality.chat.model == "qwen3.6:27b"
    assert quality.chat.options["num_ctx"] == 8192


def test_web_environment_configuration() -> None:
    config = CortexMuxConfig.load(
        environ={
            "CORTEXMUX_WEB_ENABLED": "true",
            "CORTEXMUX_WEB_ALLOWED_HOSTS": "one.example, two.example",
            "CORTEXMUX_WEB_ALLOW_PRIVATE_HOSTS": "false",
        }
    )
    assert config.web.enabled
    assert config.web.allowed_hosts == {"one.example", "two.example"}
    assert not config.web.allow_private_hosts


def test_openai_environment_configuration_is_explicit() -> None:
    config = CortexMuxConfig.load(
        environ={
            "CORTEXMUX_OPENAI_ENABLED": "true",
            "CORTEXMUX_OPENAI_BASE_URL": "https://api.openai.com/v1",
        }
    )
    assert config.providers.openai.enabled
    assert config.providers.openai.base_url == "https://api.openai.com/v1"
    assert config.providers.openai.api_key is None


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


def test_web_url_policy() -> None:
    assert (
        validate_web_url(
            "https://93.184.216.34/page#fragment",
            allowed_hosts={"93.184.216.34"},
        )
        == "https://93.184.216.34/page"
    )
    with pytest.raises(RemoteHostNotAllowedError):
        validate_web_url("http://127.0.0.1/admin")
    with pytest.raises(RemoteHostNotAllowedError):
        validate_web_url("https://93.184.216.34", allowed_hosts={"example.com"})


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


@pytest.mark.parametrize("json_type", ["integer", "number"])
def test_json_schema_numeric_types_reject_booleans(json_type: str) -> None:
    with pytest.raises(ValueError, match=f"must be {json_type}"):
        validate_json_schema(True, {"type": json_type})


def test_json_schema_rejects_unsupported_validation_keywords() -> None:
    with pytest.raises(ValueError, match="Unsupported JSON Schema keyword: format"):
        validate_json_schema("2026-09-07", {"type": "string", "format": "date"})
