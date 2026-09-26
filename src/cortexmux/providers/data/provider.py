"""Deterministic data-analysis provider with optional bounded LLM assistance."""

from __future__ import annotations

import importlib.util
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.config import DataConfig
from cortexmux.core.exceptions import (
    CortexMuxError,
    DataAnalysisError,
    OptionalDependencyError,
    UnsupportedTaskError,
)
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.data.engines.base import BaseDataEngine
from cortexmux.providers.data.engines.duckdb_engine import DuckDBEngine
from cortexmux.providers.data.engines.pandas_engine import PandasEngine
from cortexmux.providers.data.engines.polars_engine import PolarsEngine
from cortexmux.providers.data.math_verification import MathVerifier
from cortexmux.providers.data.planner import (
    asks_for_calculation,
    baseline_plan,
    interpretation_prompt,
    planning_prompt,
)
from cortexmux.providers.data.reporting import markdown_report
from cortexmux.providers.data.schemas import AnalysisPlan
from cortexmux.providers.data.visualization import create_baseline_charts
from cortexmux.schemas.calculations import (
    CalculationClaim,
    CalculationVerification,
    InterpretationWithCalculations,
    VerificationStatus,
)
from cortexmux.schemas.common import ChatMessage, HealthStatus, ModelInfo
from cortexmux.schemas.requests import (
    ChatRequest,
    CortexRequest,
    DataAnalysisRequest,
    StructuredOutputRequest,
)
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    DataAnalysisResponse,
    StructuredResponse,
)

RequestExecutor = Callable[[CortexRequest], Awaitable[CortexResponse]]


class DataAnalysisProvider(BaseProvider):
    """Load local tabular data and execute only validated operations."""

    name = "data"

    def __init__(
        self,
        config: DataConfig,
        *,
        output_dir: Path,
        request_executor: RequestExecutor | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.request_executor = request_executor

    async def healthcheck(self) -> HealthStatus:
        """Report whether the baseline Pandas engine is installed."""
        available = importlib.util.find_spec("pandas") is not None
        return HealthStatus(
            provider=self.name,
            available=available,
            message=(
                "Deterministic data analysis is available."
                if available
                else "Install cortexmux[data] to enable data analysis."
            ),
        )

    async def list_models(self) -> list[ModelInfo]:
        """List installed deterministic engines as normalized models."""
        packages = {"pandas": "pandas", "polars": "polars", "duckdb": "duckdb"}
        return [
            ModelInfo(name=name, provider=self.name, family="deterministic-engine")
            for name, package in packages.items()
            if importlib.util.find_spec(package) is not None
        ]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe deterministic data-analysis support."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset({TaskType.DATA_ANALYSIS}),
                batch=True,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this is a data-analysis request."""
        return task is TaskType.DATA_ANALYSIS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Run the safe analysis pipeline and optional bounded LLM assistance."""
        if not isinstance(request, DataAnalysisRequest):
            raise UnsupportedTaskError("Data provider supports only data analysis.")
        engine = self._select_engine(request)
        frame = engine.load(request.source, max_file_size_mb=self.config.max_file_size_mb)
        profile = engine.profile(
            frame,
            max_sample_rows=self.config.max_sample_rows,
            sensitive_patterns=self.config.sensitive_column_patterns,
        )
        model_enabled = bool(request.interpretation_provider and request.interpretation_model)
        safe_frame = (
            frame.drop(columns=profile.redacted_columns)
            if profile.redacted_columns and model_enabled
            else frame
        )
        model_profile = (
            engine.profile(
                safe_frame,
                max_sample_rows=self.config.max_sample_rows,
                sensitive_patterns=[],
            )
            if profile.redacted_columns and model_enabled
            else profile
        )
        warnings: list[str] = []
        plan = await self._resolve_plan(request, model_profile, warnings)
        try:
            plan.bounded(self.config.max_plan_steps)
        except ValueError as exc:
            raise DataAnalysisError(str(exc), request_id=request.request_id) from exc
        try:
            results = engine.execute(frame, plan, max_result_rows=self.config.max_result_rows)
        except DataAnalysisError:
            can_fallback = (
                request.plan is None
                and request.interpretation_provider is not None
                and request.interpretation_model is not None
                and not request.strict_planning
            )
            if not can_fallback:
                raise
            warnings.append(
                "The LLM plan referenced invalid data or operations; "
                "the deterministic baseline plan was used."
            )
            plan = baseline_plan()
            results = engine.execute(frame, plan, max_result_rows=self.config.max_result_rows)
        chart_paths = (
            create_baseline_charts(
                frame,
                output_dir=self.output_dir / "charts",
                request_id=request.request_id,
                max_charts=self.config.max_charts,
            )
            if request.create_charts
            else []
        )
        model_results = results
        can_interpret = True
        if profile.redacted_columns and model_enabled:
            try:
                model_results = engine.execute(
                    safe_frame, plan, max_result_rows=self.config.max_result_rows
                )
            except DataAnalysisError:
                can_interpret = False
                model_results = []
                warnings.append(
                    "AI interpretation was skipped because the analysis plan uses "
                    "sensitive columns."
                )
        if can_interpret:
            interpretation, calculation_claims, character_count = await self._interpret(
                request, model_profile, model_results, warnings
            )
        else:
            interpretation, calculation_claims, character_count = None, [], 0
        verifications, math_verification_passed = self._verify_calculations(
            request,
            profile=model_profile,
            results=model_results,
            claims=calculation_claims,
            warnings=warnings,
        )
        accepted_interpretation = interpretation if math_verification_passed is not False else None
        unverified_interpretation = interpretation if math_verification_passed is False else None
        disclosure = {
            "schema_columns": [
                column for column in profile.schema_info if column not in profile.redacted_columns
            ],
            "sample_rows": len(profile.sample),
            "aggregate_items": len(results),
            "redacted_columns": profile.redacted_columns,
            "character_count": character_count,
            "calculation_claims": len(calculation_claims),
        }
        report = markdown_report(
            profile,
            plan,
            engine=engine.name,
            warnings=warnings,
            interpretation=accepted_interpretation,
        )
        return DataAnalysisResponse(
            provider=self.name,
            model=engine.name,
            request_id=request.request_id,
            engine=engine.name,
            profile=profile.model_dump(by_alias=True),
            results=results,
            report_markdown=report,
            warnings=warnings,
            plan=plan.model_dump(mode="json"),
            chart_paths=chart_paths,
            interpretation=accepted_interpretation,
            unverified_interpretation=unverified_interpretation,
            calculation_verifications=verifications,
            math_verification_passed=math_verification_passed,
            disclosure=disclosure,
        )

    def _select_engine(self, request: DataAnalysisRequest) -> BaseDataEngine:
        requested = request.engine if request.engine != "auto" else self.config.engine
        if requested != "auto":
            return self._engine_named(requested)
        suffix = (
            Path(request.source).suffix.lower() if isinstance(request.source, str | Path) else ""
        )
        size = PandasEngine().source_size_mb(request.source) or 0
        if (suffix == ".parquet" or size >= self.config.auto_large_file_mb) and _installed(
            "duckdb"
        ):
            return DuckDBEngine()
        if size >= self.config.auto_large_file_mb and _installed("polars"):
            return PolarsEngine()
        return PandasEngine()

    @staticmethod
    def _engine_named(name: str) -> BaseDataEngine:
        if name == "pandas":
            return PandasEngine()
        if name == "polars":
            if not _installed("polars"):
                raise OptionalDependencyError("polars", "polars")
            return PolarsEngine()
        if name == "duckdb":
            if not _installed("duckdb"):
                raise OptionalDependencyError("duckdb", "duckdb")
            return DuckDBEngine()
        raise DataAnalysisError("Unknown data engine.", engine=name)

    async def _resolve_plan(
        self, request: DataAnalysisRequest, profile: Any, warnings: list[str]
    ) -> AnalysisPlan:
        if request.plan is not None:
            try:
                return AnalysisPlan.model_validate(request.plan)
            except ValidationError as exc:
                raise DataAnalysisError("Analysis plan is invalid.") from exc
        if (
            not request.interpretation_provider
            or not request.interpretation_model
            or not request.instruction
            or self.request_executor is None
        ):
            return baseline_plan()
        prompt = planning_prompt(
            request.instruction,
            profile,
            maximum_characters=self.config.max_prompt_characters,
        )
        planner_request = StructuredOutputRequest(
            provider=request.interpretation_provider,
            model=request.interpretation_model,
            prompt=prompt,
            json_schema=AnalysisPlan.model_json_schema(),
        )
        try:
            response = await self.request_executor(planner_request)
            if not isinstance(response, StructuredResponse):
                raise DataAnalysisError("Planner provider returned an unexpected response.")
            return AnalysisPlan.model_validate(response.parsed)
        except (CortexMuxError, ValidationError, ValueError) as exc:
            if request.strict_planning:
                raise DataAnalysisError("LLM planning failed in strict mode.") from exc
            warnings.append("LLM planning failed; the deterministic baseline plan was used.")
            return baseline_plan()

    async def _interpret(
        self,
        request: DataAnalysisRequest,
        profile: Any,
        results: list[dict[str, Any]],
        warnings: list[str],
    ) -> tuple[str | None, list[CalculationClaim], int]:
        if not request.interpretation_provider or not request.interpretation_model:
            return None, [], 0
        if self.request_executor is None:
            warnings.append("LLM interpretation was requested but no executor is configured.")
            return None, [], 0
        prompt, character_count = interpretation_prompt(
            request.instruction,
            profile,
            results,
            maximum_characters=self.config.max_prompt_characters,
            verify_calculations=request.verify_calculations,
        )
        if request.verify_calculations:
            try:
                response = await self.request_executor(
                    StructuredOutputRequest(
                        provider=request.interpretation_provider,
                        model=request.interpretation_model,
                        prompt=prompt,
                        json_schema=InterpretationWithCalculations.model_json_schema(),
                    )
                )
                if not isinstance(response, StructuredResponse):
                    raise DataAnalysisError(
                        "Interpretation provider returned an unexpected response."
                    )
                parsed = InterpretationWithCalculations.model_validate(response.parsed)
            except (CortexMuxError, ValidationError, ValueError) as exc:
                if request.require_verified_calculations:
                    raise DataAnalysisError(
                        "Structured AI interpretation failed in strict verification mode."
                    ) from exc
                warnings.append(
                    "Structured AI interpretation failed; no unverified mathematical "
                    "answer was accepted."
                )
                return None, [], character_count
            return parsed.content, parsed.calculations, character_count
        response = await self.request_executor(
            ChatRequest(
                provider=request.interpretation_provider,
                model=request.interpretation_model,
                messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            )
        )
        if not isinstance(response, ChatResponse):
            raise DataAnalysisError("Interpretation provider returned an unexpected response.")
        return response.content, [], character_count

    def _verify_calculations(
        self,
        request: DataAnalysisRequest,
        *,
        profile: Any,
        results: list[dict[str, Any]],
        claims: list[CalculationClaim],
        warnings: list[str],
    ) -> tuple[list[CalculationVerification], bool | None]:
        if not request.verify_calculations:
            return [], None
        if len(claims) > self.config.max_calculation_claims:
            raise DataAnalysisError(
                "AI returned too many calculation claims.",
                count=len(claims),
                maximum=self.config.max_calculation_claims,
            )
        verifier = MathVerifier(
            absolute_tolerance=self.config.math_absolute_tolerance,
            relative_tolerance=self.config.math_relative_tolerance,
        )
        context = {
            "profile": profile.model_dump(by_alias=True, exclude={"sample"}),
            "results": results,
        }
        verifications = [verifier.verify(claim, context) for claim in claims]
        requested = asks_for_calculation(request.instruction)
        passed: bool | None
        if verifications:
            passed = all(result.status is VerificationStatus.VERIFIED for result in verifications)
        elif requested:
            passed = False
            warnings.append(
                "A mathematical result was requested, but the AI returned no auditable "
                "calculation claim."
            )
        else:
            passed = None
        if passed is False and verifications:
            invalid = sum(
                result.status is not VerificationStatus.VERIFIED for result in verifications
            )
            warnings.append(
                f"{invalid} AI calculation claim(s) failed deterministic verification; "
                "the interpretation was withheld."
            )
        if request.require_verified_calculations and passed is not True:
            raise DataAnalysisError(
                "Strict mathematical verification failed.",
                request_id=request.request_id,
            )
        return verifications, passed


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None
