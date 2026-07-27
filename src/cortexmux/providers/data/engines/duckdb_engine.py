"""Optional DuckDB loader with fixed, non-generated queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cortexmux.core.exceptions import OptionalDependencyError
from cortexmux.providers.data.engines.pandas_engine import PandasEngine
from cortexmux.providers.data.loader import validate_source_path


class DuckDBEngine(PandasEngine):
    """Load large CSV or Parquet data with fixed DuckDB reader calls."""

    name = "duckdb"

    def load(self, source: Any, *, max_file_size_mb: int) -> Any:
        """Load through safe DuckDB reader functions without arbitrary SQL."""
        try:
            import duckdb
        except ImportError as exc:
            raise OptionalDependencyError("duckdb", "duckdb") from exc
        if not isinstance(source, str | Path):
            return super().load(source, max_file_size_mb=max_file_size_mb)
        path = validate_source_path(source, max_file_size_mb=max_file_size_mb)
        if path.suffix.lower() == ".csv":
            return duckdb.read_csv(str(path)).df()
        if path.suffix.lower() == ".parquet":
            return duckdb.read_parquet(str(path)).df()
        return super().load(path, max_file_size_mb=max_file_size_mb)
