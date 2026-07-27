"""Common deterministic tabular engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cortexmux.providers.data.schemas import AnalysisPlan, DataProfile


class BaseDataEngine(ABC):
    """Load, profile, and execute safe operations without generated code."""

    name: str

    @abstractmethod
    def load(self, source: Any, *, max_file_size_mb: int) -> Any:
        """Load a supported local source."""

    @abstractmethod
    def profile(
        self,
        frame: Any,
        *,
        max_sample_rows: int,
        sensitive_patterns: list[str],
    ) -> DataProfile:
        """Build a bounded deterministic profile."""

    @abstractmethod
    def execute(
        self, frame: Any, plan: AnalysisPlan, *, max_result_rows: int
    ) -> list[dict[str, Any]]:
        """Execute only validated plan operations."""

    def source_size_mb(self, source: Any) -> float | None:
        """Return local source size when it is a path."""
        if isinstance(source, str | Path):
            path = Path(source).expanduser()
            return path.stat().st_size / (1024 * 1024) if path.is_file() else None
        return None
