"""Optional Polars loader backed by the common safe Pandas operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cortexmux.core.exceptions import OptionalDependencyError
from cortexmux.providers.data.engines.pandas_engine import PandasEngine
from cortexmux.providers.data.loader import validate_source_path


class PolarsEngine(PandasEngine):
    """Load with Polars, then use the shared validated operation implementation."""

    name = "polars"

    def load(self, source: Any, *, max_file_size_mb: int) -> Any:
        """Load a supported source through Polars and convert to Pandas."""
        try:
            import polars as pl
        except ImportError as exc:
            raise OptionalDependencyError("polars", "polars") from exc
        if isinstance(source, pl.DataFrame):
            return self._pd().DataFrame(source.to_dicts())
        if not isinstance(source, str | Path):
            return super().load(source, max_file_size_mb=max_file_size_mb)
        path = validate_source_path(source, max_file_size_mb=max_file_size_mb)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._pd().DataFrame(pl.read_csv(path).to_dicts())
        if suffix in {".json", ".jsonl", ".ndjson"}:
            return self._pd().DataFrame(pl.read_ndjson(path).to_dicts())
        if suffix == ".parquet":
            return self._pd().DataFrame(pl.read_parquet(path).to_dicts())
        return super().load(path, max_file_size_mb=max_file_size_mb)
