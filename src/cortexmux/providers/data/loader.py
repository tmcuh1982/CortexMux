"""Safe local data-source validation helpers."""

from __future__ import annotations

from pathlib import Path

from cortexmux.core.exceptions import DataSourceError

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".xlsx", ".xls"}


def validate_source_path(source: str | Path, *, max_file_size_mb: int) -> Path:
    """Resolve and validate a supported regular local data file."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise DataSourceError("Data source does not exist.", path=str(path))
    if not path.is_file():
        raise DataSourceError("Data source is not a regular file.", path=str(path))
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DataSourceError(
            "Unsupported data source format.", path=str(path), suffix=path.suffix.lower()
        )
    size = path.stat().st_size
    if size > max_file_size_mb * 1024 * 1024:
        raise DataSourceError(
            "Data source exceeds the configured size limit.",
            path=str(path),
            size_bytes=size,
        )
    return path
