"""Deterministic Matplotlib visualizations."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any

from cortexmux.core.exceptions import OptionalDependencyError
from cortexmux.core.security import safe_output_path


def create_baseline_charts(
    frame: Any,
    *,
    output_dir: Path,
    request_id: str,
    max_charts: int,
) -> list[str]:
    """Create bounded histograms and a correlation heatmap from numeric columns."""
    try:
        plt = import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise OptionalDependencyError("matplotlib", "visualization") from exc
    numeric = frame.select_dtypes(include="number")
    paths: list[str] = []
    for column in list(numeric.columns)[: max(0, max_charts - 1)]:
        figure, axis = plt.subplots()
        axis.hist(numeric[column].dropna(), bins=20)
        axis.set_title(_safe_title(f"Distribution of {column}"))
        axis.set_xlabel(str(column))
        axis.set_ylabel("Count")
        filename = f"{request_id[:8]}_hist_{_slug(str(column))}.png"
        destination = safe_output_path(output_dir, filename)
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(destination))
    if len(paths) < max_charts and len(numeric.columns) >= 2:
        figure, axis = plt.subplots()
        correlations = numeric.corr()
        image = axis.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=45)
        axis.set_yticks(range(len(correlations.columns)), correlations.columns)
        axis.set_title("Numeric correlation heatmap")
        figure.colorbar(image, ax=axis)
        destination = safe_output_path(output_dir, f"{request_id[:8]}_correlations.png")
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(destination))
    return paths


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:60] or "column"


def _safe_title(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", "", value)[:120]
