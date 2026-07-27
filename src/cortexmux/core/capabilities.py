"""Provider capability description."""

from pydantic import BaseModel, ConfigDict, Field

from cortexmux.core.types import TaskType


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
    limits: dict[str, int | float | str] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
