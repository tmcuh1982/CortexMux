"""Validated options for the xAI Responses API adapter."""

from pydantic import BaseModel, ConfigDict, Field


class GrokRequestOptions(BaseModel):
    """Whitelisted sampling options; remote tools and storage are not configurable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
