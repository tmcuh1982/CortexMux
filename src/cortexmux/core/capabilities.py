"""Provider capability description."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortexmux.core.types import TaskType


class TemperatureRange(BaseModel):
    """Inclusive sampling-temperature bounds declared by a provider for a model."""

    model_config = ConfigDict(frozen=True)

    minimum: float = Field(ge=0, allow_inf_nan=False)
    maximum: float = Field(ge=0, allow_inf_nan=False)
    default: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def bounds_are_consistent(self) -> TemperatureRange:
        """Require an ordered range and keep its optional default inside it."""
        if self.maximum < self.minimum:
            raise ValueError("temperature maximum must be greater than or equal to minimum")
        if self.default is not None and not self.minimum <= self.default <= self.maximum:
            raise ValueError("temperature default must be within the declared range")
        return self


class TemperatureSetting(BaseModel):
    """Validated instance-level temperature default for one provider/model pair."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    minimum: float = Field(ge=0, allow_inf_nan=False)
    maximum: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def value_is_within_bounds(self) -> TemperatureSetting:
        """Keep direct construction as safe as facade-created settings."""
        if self.maximum < self.minimum:
            raise ValueError("temperature maximum must be greater than or equal to minimum")
        if not self.minimum <= self.value <= self.maximum:
            raise ValueError("temperature value must be within the declared range")
        return self


class ProviderCapability(BaseModel):
    """A provider/model capability declared without name-based inference."""

    model_config = ConfigDict(frozen=True)

    provider: str
    task_types: frozenset[TaskType]
    model: str | None = None
    streaming: bool = False
    structured_output: bool = False
    image_input: bool = False
    batch: bool = False
    locally_hosted: bool = True
    temperature: TemperatureRange | None = None
    limits: dict[str, int | float | str] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
