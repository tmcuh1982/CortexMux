"""Typed, provider-independent MCP stdio schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCPServerConfig(BaseModel):
    """Configuration for one local MCP server started with stdio."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    command: str | None = None
    arguments: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=20, gt=0, le=300)
    shutdown_timeout_seconds: float = Field(default=5, gt=0, le=60)
    max_result_characters: int = Field(default=200_000, gt=0, le=1_000_000)

    @field_validator("command")
    @classmethod
    def command_must_not_be_blank(cls, value: str | None) -> str | None:
        """Reject blank process commands while retaining an opt-in default."""
        if value is not None and not value.strip():
            raise ValueError("command must not be blank")
        if value is not None and not Path(value).is_absolute():
            raise ValueError("command must be an absolute local executable path")
        return value


class MCPTool(BaseModel):
    """One safe MCP tool description translated for a model client."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^capitalforge_[a-z0-9_]+$")
    description: str = Field(min_length=1, max_length=4_000)
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """Validated structured result from one MCP tool invocation."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(pattern=r"^capitalforge_[a-z0-9_]+$")
    data: dict[str, Any]
