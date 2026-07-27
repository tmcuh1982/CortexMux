"""Validated whitelist schemas for deterministic analysis plans."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperationType(StrEnum):
    """Safe operations implemented by CortexMux engines."""

    INSPECT_SCHEMA = "inspect_schema"
    SHAPE = "shape"
    DESCRIBE = "describe"
    MISSING_VALUES = "missing_values"
    UNIQUE_COUNTS = "unique_counts"
    VALUE_COUNTS = "value_counts"
    CORRELATIONS = "correlations"
    GROUP_BY = "group_by"
    TOP_N = "top_n"
    SORT = "sort"
    FILTER = "filter"
    DATE_PARSE = "date_parse"
    TIME_SERIES = "time_series"
    IQR_OUTLIERS = "iqr_outliers"
    HISTOGRAM = "histogram"
    BAR_SERIES = "bar_series"
    LINE_SERIES = "line_series"


class AggregationFunction(StrEnum):
    """Allowed aggregation functions."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    NUNIQUE = "nunique"


class FilterOperator(StrEnum):
    """Bounded filter comparison operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class FilterCondition(BaseModel):
    """A single validated column predicate."""

    column: str
    operator: FilterOperator
    value: str | int | float | bool | list[str | int | float | bool] | None = None


class AnalysisStep(BaseModel):
    """One whitelisted deterministic operation."""

    operation: OperationType
    columns: list[str] = Field(default_factory=list, max_length=20)
    group_by: list[str] = Field(default_factory=list, max_length=5)
    aggregations: dict[str, AggregationFunction] = Field(default_factory=dict)
    filters: list[FilterCondition] = Field(default_factory=list, max_length=10)
    n: int = Field(default=10, gt=0, le=1000)
    ascending: bool = False
    frequency: str | None = None
    title: str | None = Field(default=None, max_length=120)


class AnalysisPlan(BaseModel):
    """A bounded list of deterministic operations."""

    steps: list[AnalysisStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def steps_are_bounded(self) -> AnalysisPlan:
        """Apply a hard safety ceiling independent of configuration."""
        if len(self.steps) > 100:
            raise ValueError("analysis plans cannot exceed 100 steps")
        return self

    def bounded(self, maximum: int) -> AnalysisPlan:
        """Validate a configured plan-step limit."""
        if len(self.steps) > maximum:
            raise ValueError(f"analysis plan exceeds configured limit of {maximum}")
        return self


class DataProfile(BaseModel):
    """Deterministic baseline dataset profile."""

    model_config = ConfigDict(populate_by_name=True)

    rows: int
    columns: int
    schema_info: dict[str, str] = Field(alias="schema")
    missing_values: dict[str, int]
    numeric_summary: dict[str, dict[str, Any]]
    sample: list[dict[str, Any]]
    redacted_columns: list[str]
