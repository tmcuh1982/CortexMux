"""Deterministic data-analysis tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from cortexmux.core.config import DataConfig
from cortexmux.core.exceptions import DataAnalysisError, DataSourceError
from cortexmux.providers.data.engines import DuckDBEngine, PandasEngine, PolarsEngine
from cortexmux.providers.data.provider import DataAnalysisProvider
from cortexmux.providers.data.schemas import AnalysisPlan
from cortexmux.schemas.requests import ChatRequest, DataAnalysisRequest, StructuredOutputRequest
from cortexmux.schemas.responses import ChatResponse, StructuredResponse


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Create a compact dataset with missing and sensitive values."""
    path = tmp_path / "sales.csv"
    path.write_text(
        "category,value,password,date\nA,10,secret,2026-01-01\n"
        "A,20,hidden,2026-02-01\nB,,private,2026-03-01\nB,1000,x,2026-04-01\n",
        encoding="utf-8",
    )
    return path


def test_load_profile_and_redaction(csv_file: Path) -> None:
    engine = PandasEngine()
    frame = engine.load(csv_file, max_file_size_mb=1)
    profile = engine.profile(frame, max_sample_rows=2, sensitive_patterns=["password", "token"])
    assert (profile.rows, profile.columns) == (4, 4)
    assert profile.missing_values["value"] == 1
    assert profile.redacted_columns == ["password"]
    assert "password" not in profile.sample[0]


def test_json_jsonl_and_invalid_file(tmp_path: Path) -> None:
    engine = PandasEngine()
    json_path = tmp_path / "data.json"
    json_path.write_text('[{"a":1},{"a":2}]', encoding="utf-8")
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    assert len(engine.load(json_path, max_file_size_mb=1)) == 2
    assert len(engine.load(jsonl_path, max_file_size_mb=1)) == 2
    bad = tmp_path / "data.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(DataSourceError):
        engine.load(bad, max_file_size_mb=1)
    with pytest.raises(DataSourceError):
        engine.load(json_path, max_file_size_mb=0)


def test_whitelisted_operations(csv_file: Path) -> None:
    engine = PandasEngine()
    frame = engine.load(csv_file, max_file_size_mb=1)
    plan = AnalysisPlan.model_validate(
        {
            "steps": [
                {"operation": "shape"},
                {"operation": "missing_values"},
                {
                    "operation": "group_by",
                    "group_by": ["category"],
                    "aggregations": {"value": "sum"},
                },
                {"operation": "correlations", "columns": ["value"]},
                {"operation": "iqr_outliers", "columns": ["value"]},
                {
                    "operation": "filter",
                    "filters": [{"column": "value", "operator": "gt", "value": 10}],
                },
                {"operation": "value_counts", "columns": ["category"]},
                {"operation": "histogram", "columns": ["value"], "n": 3},
            ]
        }
    )
    results = engine.execute(frame, plan, max_result_rows=20)
    assert len(results) == 8
    assert results[0]["data"] == [{"rows": 4, "columns": 4}]
    assert results[2]["data"][0]["value"] == 30.0


def test_plan_rejects_unknown_or_too_many_steps() -> None:
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate({"steps": [{"operation": "execute_python"}]})
    plan = AnalysisPlan.model_validate({"steps": [{"operation": "shape"} for _ in range(21)]})
    with pytest.raises(ValueError):
        plan.bounded(20)


@pytest.mark.asyncio
async def test_provider_fallback_and_charts(csv_file: Path, tmp_path: Path) -> None:
    provider = DataAnalysisProvider(DataConfig(), output_dir=tmp_path)
    response = await provider.execute(
        DataAnalysisRequest(
            source=csv_file,
            provider="data",
            create_charts=True,
        )
    )
    assert response.engine == "pandas"
    assert response.interpretation is None
    assert response.chart_paths
    assert all(Path(path).is_file() for path in response.chart_paths)
    assert response.disclosure["redacted_columns"] == ["password"]


@pytest.mark.asyncio
async def test_invalid_plan_is_wrapped(csv_file: Path, tmp_path: Path) -> None:
    provider = DataAnalysisProvider(DataConfig(), output_dir=tmp_path)
    with pytest.raises(DataAnalysisError):
        await provider.execute(
            DataAnalysisRequest(
                source=csv_file,
                provider="data",
                plan={"steps": [{"operation": "nope"}]},
            )
        )


def test_optional_engines_load(csv_file: Path) -> None:
    assert len(PolarsEngine().load(csv_file, max_file_size_mb=1)) == 4
    assert len(DuckDBEngine().load(csv_file, max_file_size_mb=1)) == 4
    frame = pd.DataFrame({"a": [1]})
    assert len(PandasEngine().load(frame, max_file_size_mb=1)) == 1


def test_additional_safe_operations(csv_file: Path) -> None:
    engine = PandasEngine()
    frame = engine.load(csv_file, max_file_size_mb=1)
    plan = AnalysisPlan.model_validate(
        {
            "steps": [
                {"operation": "unique_counts"},
                {"operation": "sort", "columns": ["value"], "n": 2},
                {"operation": "top_n", "columns": ["value"], "n": 2},
                {"operation": "date_parse", "columns": ["date"]},
                {
                    "operation": "time_series",
                    "columns": ["date", "value"],
                    "frequency": "ME",
                },
                {"operation": "bar_series", "columns": ["category", "value"]},
                {"operation": "line_series", "columns": ["date", "value"]},
            ]
        }
    )
    results = engine.execute(frame, plan, max_result_rows=10)
    assert len(results) == 7
    assert results[3]["data"][0]["parsed"] == 4


@pytest.mark.asyncio
async def test_bounded_llm_planning_and_interpretation(csv_file: Path, tmp_path: Path) -> None:
    prompts: list[str] = []

    async def executor(request: object) -> object:
        if isinstance(request, StructuredOutputRequest):
            prompts.append(request.prompt)
            return StructuredResponse(
                provider="ollama",
                model=request.model,
                request_id=request.request_id,
                content='{"steps":[{"operation":"shape"}]}',
                parsed={"steps": [{"operation": "shape"}]},
            )
        assert isinstance(request, ChatRequest)
        prompts.append(request.messages[0].content)
        return ChatResponse(
            provider="ollama",
            model=request.model,
            request_id=request.request_id,
            content="The dataset contains four rows.",
        )

    provider = DataAnalysisProvider(
        DataConfig(max_prompt_characters=5000),
        output_dir=tmp_path,
        request_executor=executor,  # type: ignore[arg-type]
    )
    response = await provider.execute(
        DataAnalysisRequest(
            source=csv_file,
            provider="data",
            instruction="Summarize.",
            interpretation_provider="ollama",
            interpretation_model="model",
        )
    )
    assert response.interpretation == "The dataset contains four rows."
    assert response.plan is not None
    assert response.plan["steps"][0]["operation"] == "shape"
    assert response.disclosure["sample_rows"] == 4
    assert all("hidden" not in prompt and "private" not in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_data_provider_introspection_and_explicit_engines(tmp_path: Path) -> None:
    provider = DataAnalysisProvider(DataConfig(), output_dir=tmp_path)
    assert (await provider.healthcheck()).available
    assert {model.name for model in await provider.list_models()} >= {
        "pandas",
        "polars",
        "duckdb",
    }
    assert provider.supports(DataAnalysisRequest(source=[]).task)
    assert (await provider.get_capabilities())[0].batch
    assert provider._engine_named("polars").name == "polars"
    assert provider._engine_named("duckdb").name == "duckdb"
    with pytest.raises(DataAnalysisError):
        provider._engine_named("unknown")


@pytest.mark.asyncio
async def test_planner_failure_falls_back_or_raises(csv_file: Path, tmp_path: Path) -> None:
    async def failing_executor(request: object) -> object:
        if isinstance(request, StructuredOutputRequest):
            raise DataAnalysisError("planner failed")
        assert isinstance(request, ChatRequest)
        return ChatResponse(
            provider="ollama",
            model=request.model,
            request_id=request.request_id,
            content="Fallback interpretation.",
        )

    provider = DataAnalysisProvider(
        DataConfig(),
        output_dir=tmp_path,
        request_executor=failing_executor,  # type: ignore[arg-type]
    )
    request = DataAnalysisRequest(
        source=csv_file,
        instruction="Summarize.",
        interpretation_provider="ollama",
        interpretation_model="model",
    )
    response = await provider.execute(request)
    assert "baseline plan" in response.warnings[0]
    with pytest.raises(DataAnalysisError):
        await provider.execute(request.model_copy(update={"strict_planning": True}))
