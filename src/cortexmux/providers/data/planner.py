"""Safe structured planning and bounded interpretation prompts."""

from __future__ import annotations

import json
from typing import Any

from cortexmux.providers.data.schemas import AnalysisPlan, DataProfile


def baseline_plan() -> AnalysisPlan:
    """Return a useful provider-independent deterministic baseline."""
    return AnalysisPlan.model_validate(
        {
            "steps": [
                {"operation": "inspect_schema"},
                {"operation": "shape"},
                {"operation": "missing_values"},
                {"operation": "describe"},
            ]
        }
    )


def planning_prompt(instruction: str, profile: DataProfile, *, maximum_characters: int) -> str:
    """Create a bounded prompt containing schema and aggregates, never the full dataset."""
    safe = {
        "instruction": instruction,
        "schema": profile.schema_info,
        "rows": profile.rows,
        "columns": profile.columns,
        "missing_values": profile.missing_values,
        "numeric_summary": profile.numeric_summary,
        "sample": profile.sample,
        "redacted_columns": profile.redacted_columns,
    }
    prefix = (
        "Create a safe deterministic analysis plan using only the supplied JSON schema. "
        "Do not request code or SQL.\n"
    )
    return (prefix + json.dumps(safe, ensure_ascii=False, default=str))[:maximum_characters]


def interpretation_prompt(
    instruction: str | None,
    profile: DataProfile,
    results: list[dict[str, Any]],
    *,
    maximum_characters: int,
) -> tuple[str, int]:
    """Create a bounded summary-only interpretation prompt."""
    payload = {
        "instruction": instruction,
        "profile": profile.model_dump(exclude={"sample"}),
        "bounded_sample": profile.sample,
        "results": results,
    }
    text = (
        "Explain these deterministic analysis results concisely. State uncertainty and do "
        "not invent facts. The payload contains only bounded summaries and a redacted sample.\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    bounded = text[:maximum_characters]
    return bounded, len(bounded)
