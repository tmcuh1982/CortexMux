"""Typed configuration loaded from defaults, TOML, environment, and Python."""

from __future__ import annotations

import os
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from cortexmux.core.exceptions import ConfigurationError
from cortexmux.core.types import TaskType
from cortexmux.mcp.schemas import MCPServerConfig


class CoreConfig(BaseModel):
    """General process configuration."""

    output_dir: Path = Path("./outputs")
    allow_remote_hosts: bool = False
    approved_hosts: set[str] = Field(default_factory=set)
    log_level: str = "INFO"


class OllamaDefaults(BaseModel):
    """Configured model names by Ollama task."""

    chat: str | None = None
    text_generation: str | None = None
    structured_output: str | None = None
    vision: str | None = None
    embedding: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def empty_is_unset(cls, value: object) -> object:
        """Treat empty strings as an absent default."""
        return None if value == "" else value


class OllamaConfig(BaseModel):
    """Ollama endpoint configuration."""

    enabled: bool = True
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=120, gt=0)
    headers: dict[str, SecretStr] = Field(default_factory=dict, repr=False)
    defaults: OllamaDefaults = Field(default_factory=OllamaDefaults)


class ComfyUIConfig(BaseModel):
    """ComfyUI endpoint and workflow configuration."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:8188"
    timeout_seconds: float = Field(default=600, gt=0)
    workflow_dir: Path = Path("./workflows")
    default_workflow: str | None = None
    model_folders: list[str] = Field(default_factory=lambda: ["checkpoints"])
    headers: dict[str, SecretStr] = Field(default_factory=dict, repr=False)

    @field_validator("default_workflow", mode="before")
    @classmethod
    def empty_workflow_is_unset(cls, value: object) -> object:
        """Treat an empty workflow setting as absent."""
        return None if value == "" else value


class CapitalForgeMCPConfig(MCPServerConfig):
    """Opt-in configuration for CapitalForge's local read-only MCP process."""

    @model_validator(mode="after")
    def enabled_server_requires_command(self) -> CapitalForgeMCPConfig:
        """Require an explicit executable when the integration is enabled."""
        if self.enabled and self.command is None:
            raise ValueError("mcp.capitalforge.command is required when enabled")
        return self


class MCPConfig(BaseModel):
    """Local MCP integrations kept independent from individual providers."""

    capitalforge: CapitalForgeMCPConfig = Field(default_factory=CapitalForgeMCPConfig)


class ProvidersConfig(BaseModel):
    """Built-in provider configurations."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)


class RoutingDefaults(BaseModel):
    """Default provider selection by task."""

    chat_provider: str | None = "ollama"
    text_generation_provider: str | None = "ollama"
    structured_output_provider: str | None = "ollama"
    vision_provider: str | None = "ollama"
    embedding_provider: str | None = "ollama"
    image_generation_provider: str | None = "comfyui"
    data_analysis_provider: str | None = "data"

    def provider_for(self, task: TaskType) -> str | None:
        """Return the configured provider for a task."""
        return getattr(self, f"{task.value}_provider", None)


class TaskModelRoute(BaseModel):
    """Explicit provider and installed model for one task in a project profile."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class ModelProfile(BaseModel):
    """Named project-specific model choices grouped by CortexMux task."""

    model_config = ConfigDict(extra="forbid")

    chat: TaskModelRoute | None = None
    text_generation: TaskModelRoute | None = None
    structured_output: TaskModelRoute | None = None
    vision: TaskModelRoute | None = None
    embedding: TaskModelRoute | None = None

    def route_for(self, task: TaskType) -> TaskModelRoute | None:
        """Return the configured route for a model-backed task."""
        value = getattr(self, task.value, None)
        return value if isinstance(value, TaskModelRoute) else None


class RoutingConfig(BaseModel):
    """Routing configuration namespace."""

    defaults: RoutingDefaults = Field(default_factory=RoutingDefaults)
    active_profile: str | None = None
    profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    validate_model_availability: bool = True

    @model_validator(mode="after")
    def active_profile_must_exist(self) -> RoutingConfig:
        """Reject a selected profile that is not defined by this project."""
        if self.active_profile is not None and self.active_profile not in self.profiles:
            raise ValueError("routing.active_profile must name a configured routing profile")
        return self

    def profile_for(self, name: str | None = None) -> ModelProfile | None:
        """Return the requested profile, or this project's active profile."""
        selected = name if name is not None else self.active_profile
        if selected is None:
            return None
        try:
            return self.profiles[selected]
        except KeyError as exc:
            raise ConfigurationError(
                "Routing profile is not configured.", profile=selected
            ) from exc


class DataConfig(BaseModel):
    """Safe deterministic analysis limits."""

    engine: str = "auto"
    max_file_size_mb: int = Field(default=500, gt=0)
    max_sample_rows: int = Field(default=50, ge=0, le=1000)
    max_result_rows: int = Field(default=200, gt=0, le=10_000)
    max_plan_steps: int = Field(default=20, gt=0, le=100)
    max_charts: int = Field(default=5, ge=0, le=20)
    max_prompt_characters: int = Field(default=20_000, gt=0)
    max_calculation_claims: int = Field(default=20, gt=0, le=100)
    math_absolute_tolerance: Decimal = Field(default=Decimal("1e-9"), ge=0)
    math_relative_tolerance: Decimal = Field(default=Decimal("1e-6"), ge=0)
    sensitive_column_patterns: list[str] = Field(
        default_factory=lambda: ["password", "token", "secret", "api_key", "private_key"]
    )
    auto_large_file_mb: int = Field(default=100, gt=0)


class WebConfig(BaseModel):
    """Opt-in, bounded web-page retrieval settings."""

    enabled: bool = False
    allowed_hosts: set[str] = Field(default_factory=set)
    allowed_ports: set[int] = Field(default_factory=lambda: {80, 443})
    allow_private_hosts: bool = False
    timeout_seconds: float = Field(default=15, gt=0, le=120)
    max_response_bytes: int = Field(default=2_000_000, gt=0, le=20_000_000)
    max_text_characters: int = Field(default=100_000, gt=0, le=1_000_000)
    max_redirects: int = Field(default=3, ge=0, le=10)
    max_tables: int = Field(default=20, ge=0, le=100)
    max_table_rows: int = Field(default=1_000, ge=0, le=10_000)
    user_agent: str = Field(default="CortexMux-web-extraction/1", min_length=1)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def normalize_allowed_hosts(cls, value: object) -> object:
        """Accept comma-separated hosts and normalize surrounding whitespace."""
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        return value


class CortexMuxConfig(BaseModel):
    """Complete CortexMux configuration."""

    core: CoreConfig = Field(default_factory=CoreConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    config_path: Path | None = Field(default=None, exclude=True)

    def model_for(self, task: TaskType, provider: str) -> str | None:
        """Return a provider's configured default model for a task."""
        if provider == "ollama" and hasattr(self.providers.ollama.defaults, task.value):
            value = getattr(self.providers.ollama.defaults, task.value)
            return value if isinstance(value, str) else None
        return None

    @classmethod
    def load(
        cls,
        *,
        path: Path | str | None = None,
        environ: dict[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> CortexMuxConfig:
        """Load configuration with Python > environment > TOML > defaults precedence."""
        env = dict(os.environ if environ is None else environ)
        explicit_path = Path(path).expanduser() if path is not None else None
        env_path = (
            Path(env["CORTEXMUX_CONFIG"]).expanduser() if env.get("CORTEXMUX_CONFIG") else None
        )
        selected_path = explicit_path or env_path
        if selected_path is None:
            candidate = Path(user_config_dir("cortexmux")) / "config.toml"
            selected_path = candidate if candidate.exists() else None
        values: dict[str, Any] = {}
        if selected_path is not None:
            if not selected_path.is_file():
                raise ConfigurationError(
                    "Configuration file does not exist.", path=str(selected_path)
                )
            try:
                values = tomllib.loads(selected_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigurationError(
                    "Configuration file could not be read.", path=str(selected_path)
                ) from exc
        _deep_merge(values, _environment_values(env))
        if overrides:
            _deep_merge(values, overrides)
        values["config_path"] = selected_path
        try:
            return cls.model_validate(values)
        except ValueError as exc:
            raise ConfigurationError("Configuration validation failed.", details=str(exc)) from exc


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _environment_values(env: dict[str, str]) -> dict[str, Any]:
    mapping: dict[str, tuple[str, ...]] = {
        "CORTEXMUX_OLLAMA_BASE_URL": ("providers", "ollama", "base_url"),
        "CORTEXMUX_COMFYUI_BASE_URL": ("providers", "comfyui", "base_url"),
        "CORTEXMUX_OUTPUT_DIR": ("core", "output_dir"),
        "CORTEXMUX_ALLOW_REMOTE_HOSTS": ("core", "allow_remote_hosts"),
        "CORTEXMUX_LOG_LEVEL": ("core", "log_level"),
        "CORTEXMUX_DEFAULT_CHAT_MODEL": ("providers", "ollama", "defaults", "chat"),
        "CORTEXMUX_DEFAULT_VISION_MODEL": ("providers", "ollama", "defaults", "vision"),
        "CORTEXMUX_DEFAULT_EMBEDDING_MODEL": ("providers", "ollama", "defaults", "embedding"),
        "CORTEXMUX_DEFAULT_COMFYUI_WORKFLOW": (
            "providers",
            "comfyui",
            "default_workflow",
        ),
        "CORTEXMUX_WEB_ENABLED": ("web", "enabled"),
        "CORTEXMUX_WEB_ALLOWED_HOSTS": ("web", "allowed_hosts"),
        "CORTEXMUX_WEB_ALLOW_PRIVATE_HOSTS": ("web", "allow_private_hosts"),
    }
    result: dict[str, Any] = {}
    for variable, path in mapping.items():
        if variable not in env:
            continue
        value: Any = env[variable]
        if variable in {
            "CORTEXMUX_ALLOW_REMOTE_HOSTS",
            "CORTEXMUX_WEB_ENABLED",
            "CORTEXMUX_WEB_ALLOW_PRIVATE_HOSTS",
        }:
            value = value.lower() in {"1", "true", "yes", "on"}
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return result
