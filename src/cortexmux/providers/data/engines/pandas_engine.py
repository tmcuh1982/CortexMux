"""Pandas deterministic analysis engine."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from cortexmux.core.exceptions import DataAnalysisError, DataSourceError, OptionalDependencyError
from cortexmux.providers.data.engines.base import BaseDataEngine
from cortexmux.providers.data.loader import validate_source_path
from cortexmux.providers.data.schemas import (
    AnalysisPlan,
    AnalysisStep,
    DataProfile,
    FilterOperator,
    OperationType,
)


class PandasEngine(BaseDataEngine):
    """Baseline engine for safe, whitelisted dataframe operations."""

    name = "pandas"

    @staticmethod
    def _pd() -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError("pandas", "data") from exc
        return pd

    def load(self, source: Any, *, max_file_size_mb: int) -> Any:
        """Load supported paths or copy dataframe-like input."""
        pd = self._pd()
        if isinstance(source, pd.DataFrame):
            return source.copy(deep=True)
        if source.__class__.__module__.startswith("polars") and hasattr(source, "to_dicts"):
            return pd.DataFrame(source.to_dicts())
        if not isinstance(source, str | Path):
            raise DataSourceError("Data source must be a path or supported dataframe.")
        path = validate_source_path(source, max_file_size_mb=max_file_size_mb)
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix == ".json":
                return pd.read_json(path)
            if suffix in {".jsonl", ".ndjson"}:
                return pd.read_json(path, lines=True)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path)
        except (OSError, ValueError, ImportError) as exc:
            raise DataSourceError("Data source could not be loaded.", path=str(path)) from exc
        raise DataSourceError("Unsupported data source format.", path=str(path))

    def profile(
        self,
        frame: Any,
        *,
        max_sample_rows: int,
        sensitive_patterns: list[str],
    ) -> DataProfile:
        """Build schema, shape, missingness, statistics, and a redacted sample."""
        redacted = _sensitive_columns(frame.columns, sensitive_patterns)
        safe = frame.drop(columns=redacted, errors="ignore")
        numeric = safe.select_dtypes(include="number")
        numeric_summary: dict[str, dict[str, Any]] = {}
        if not numeric.empty:
            described = numeric.describe().replace({float("nan"): None})
            numeric_summary = {
                str(column): {
                    str(stat): _json_value(value)
                    for stat, value in described[column].to_dict().items()
                }
                for column in described.columns
            }
        sample = [
            {str(key): _json_value(value) for key, value in row.items()}
            for row in safe.head(max_sample_rows).to_dict(orient="records")
        ]
        return DataProfile(
            rows=len(frame),
            columns=len(frame.columns),
            schema_info={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            missing_values={
                str(column): int(count) for column, count in frame.isna().sum().items()
            },
            numeric_summary=numeric_summary,
            sample=sample,
            redacted_columns=redacted,
        )

    def execute(
        self, frame: Any, plan: AnalysisPlan, *, max_result_rows: int
    ) -> list[dict[str, Any]]:
        """Execute validated operations using direct Pandas calls only."""
        results: list[dict[str, Any]] = []
        working = frame.copy(deep=True)
        for index, step in enumerate(plan.steps):
            _validate_columns(working, step)
            try:
                payload, working = self._execute_step(working, step, max_result_rows)
            except (KeyError, TypeError, ValueError) as exc:
                raise DataAnalysisError(
                    "A deterministic analysis step failed.",
                    step=index,
                    operation=step.operation.value,
                ) from exc
            results.append(
                {
                    "step": index,
                    "operation": step.operation.value,
                    "data": _records(payload, max_result_rows),
                }
            )
        return results

    def _execute_step(self, frame: Any, step: AnalysisStep, limit: int) -> tuple[Any, Any]:
        pd = self._pd()
        operation = step.operation
        if operation is OperationType.INSPECT_SCHEMA:
            return [{"column": str(c), "dtype": str(t)} for c, t in frame.dtypes.items()], frame
        if operation is OperationType.SHAPE:
            return [{"rows": len(frame), "columns": len(frame.columns)}], frame
        if operation is OperationType.DESCRIBE:
            columns = step.columns or list(frame.select_dtypes(include="number").columns)
            return frame[columns].describe().reset_index(), frame
        if operation is OperationType.MISSING_VALUES:
            data = frame.isna().sum().rename_axis("column").reset_index(name="missing")
            return data, frame
        if operation is OperationType.UNIQUE_COUNTS:
            data = frame.nunique(dropna=False).rename_axis("column").reset_index(name="unique")
            return data, frame
        if operation is OperationType.VALUE_COUNTS:
            column = _one_column(step)
            return frame[column].value_counts(dropna=False).head(step.n).reset_index(), frame
        if operation is OperationType.CORRELATIONS:
            columns = step.columns or list(frame.select_dtypes(include="number").columns)
            return frame[columns].corr().reset_index(), frame
        if operation is OperationType.GROUP_BY:
            grouped = frame.groupby(step.group_by, dropna=False)
            if step.aggregations:
                aggregated = grouped.agg(
                    {key: function.value for key, function in step.aggregations.items()}
                ).reset_index()
                return aggregated, frame
            return grouped.size().rename("count").reset_index(), frame
        if operation in {OperationType.TOP_N, OperationType.SORT}:
            columns = step.columns
            return frame.sort_values(columns, ascending=step.ascending).head(step.n), frame
        if operation is OperationType.FILTER:
            filtered = frame
            for condition in step.filters:
                series = filtered[condition.column]
                op = condition.operator
                value = condition.value
                if op is FilterOperator.EQ:
                    mask = series == value
                elif op is FilterOperator.NE:
                    mask = series != value
                elif op is FilterOperator.GT:
                    mask = series > value
                elif op is FilterOperator.GTE:
                    mask = series >= value
                elif op is FilterOperator.LT:
                    mask = series < value
                elif op is FilterOperator.LTE:
                    mask = series <= value
                elif op is FilterOperator.IN:
                    mask = series.isin(value if isinstance(value, list) else [value])
                elif op is FilterOperator.CONTAINS:
                    mask = series.astype(str).str.contains(str(value), regex=False, na=False)
                elif op is FilterOperator.IS_NULL:
                    mask = series.isna()
                else:
                    mask = series.notna()
                filtered = filtered[mask]
            return filtered.head(limit), filtered
        if operation is OperationType.DATE_PARSE:
            column = _one_column(step)
            converted = frame.copy()
            converted[column] = pd.to_datetime(converted[column], errors="coerce")
            return [{"column": column, "parsed": int(converted[column].notna().sum())}], converted
        if operation is OperationType.TIME_SERIES:
            date_column = step.columns[0]
            value_column = step.columns[1] if len(step.columns) > 1 else None
            converted = frame.copy()
            converted[date_column] = pd.to_datetime(converted[date_column], errors="coerce")
            indexed = converted.dropna(subset=[date_column]).set_index(date_column)
            frequency = step.frequency or "ME"
            data = (
                indexed[value_column].resample(frequency).sum().reset_index()
                if value_column
                else indexed.resample(frequency).size().rename("count").reset_index()
            )
            return data, frame
        if operation is OperationType.IQR_OUTLIERS:
            column = _one_column(step)
            q1, q3 = frame[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (frame[column] < q1 - 1.5 * iqr) | (frame[column] > q3 + 1.5 * iqr)
            return frame.loc[mask, [column]].head(limit), frame
        if operation is OperationType.HISTOGRAM:
            column = _one_column(step)
            counts, edges = _numpy_histogram(frame[column].dropna(), bins=min(step.n, 100))
            return [
                {"left": float(edges[i]), "right": float(edges[i + 1]), "count": int(counts[i])}
                for i in range(len(counts))
            ], frame
        if operation in {OperationType.BAR_SERIES, OperationType.LINE_SERIES}:
            columns = step.columns
            return frame[columns].head(limit), frame
        raise DataAnalysisError("Unsupported deterministic operation.", operation=operation.value)


def _sensitive_columns(columns: Any, patterns: list[str]) -> list[str]:
    expressions = [re.compile(pattern.replace("*", ".*"), re.IGNORECASE) for pattern in patterns]
    return [str(column) for column in columns if any(p.search(str(column)) for p in expressions)]


def _validate_columns(frame: Any, step: AnalysisStep) -> None:
    referenced = set(step.columns) | set(step.group_by) | set(step.aggregations)
    referenced.update(condition.column for condition in step.filters)
    missing = sorted(referenced - {str(column) for column in frame.columns})
    if missing:
        raise DataAnalysisError("Analysis plan references missing columns.", columns=missing)
    if step.operation is OperationType.GROUP_BY and not step.group_by:
        raise DataAnalysisError("group_by requires at least one grouping column.")


def _one_column(step: AnalysisStep) -> str:
    if len(step.columns) != 1:
        raise DataAnalysisError(
            "Operation requires exactly one column.", operation=step.operation.value
        )
    return step.columns[0]


def _records(value: Any, limit: int) -> list[dict[str, Any]]:
    if hasattr(value, "head") and hasattr(value, "to_dict"):
        rows = value.head(limit).to_dict(orient="records")
    elif isinstance(value, list):
        rows = value[:limit]
    elif isinstance(value, dict):
        rows = [value]
    else:
        rows = [{"value": value}]
    return [{str(key): _json_value(item) for key, item in row.items()} for row in rows]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    try:
        if value != value:
            return None
    except TypeError:
        pass
    return value if isinstance(value, str | int | float | bool) else str(value)


def _numpy_histogram(values: Any, bins: int) -> tuple[Any, Any]:
    try:
        np = import_module("numpy")
    except ImportError as exc:
        raise OptionalDependencyError("numpy", "data") from exc
    return cast(tuple[Any, Any], np.histogram(values, bins=bins))
