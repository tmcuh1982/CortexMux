"""Safe structured planning and bounded interpretation prompts."""

from __future__ import annotations

import json
import re
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
    verify_calculations: bool = False,
) -> tuple[str, int]:
    """Create a bounded summary-only interpretation prompt."""
    payload = {
        "instruction": instruction,
        "profile": profile.model_dump(by_alias=True, exclude={"sample"}),
        "bounded_sample": profile.sample,
        "results": results,
    }
    instructions = (
        "Explain these deterministic analysis results concisely. State uncertainty and do "
        "not invent facts. The payload contains only bounded summaries and a redacted sample. "
    )
    if verify_calculations:
        instructions += (
            "Return the required structured object with `content` and `calculations`. "
            "Every numerical conclusion that you calculate must have one calculation entry. "
            "Use source_path JSON Pointers rooted at /profile or /results for every value "
            "taken from the payload; use literal only for genuine constants. Supported "
            "operations are sum, subtract, product, divide, mean, minimum, maximum, "
            "percentage, and percentage_change. Never hide arithmetic only in `content`. "
        )
    text = instructions + "\n" + json.dumps(payload, ensure_ascii=False, default=str)
    bounded = text[:maximum_characters]
    return bounded, len(bounded)


def asks_for_calculation(instruction: str | None) -> bool:
    """Conservatively detect common French and English calculation requests."""
    if not instruction:
        return False
    normalized = instruction.casefold()
    patterns = (
        r"\bcalcul\w*",
        r"\bsomme\b",
        r"\btotal\b",
        r"\bmoyenne\b",
        r"\bm[ée]diane\b",
        r"\bpourcentage\b",
        r"\btaux\b",
        r"\bratio\b",
        r"\bdiff[ée]rence\b",
        r"\b[ée]volution\b",
        r"\bcroissance\b",
        r"\bcombien\b",
        r"\bcompute\b",
        r"\bcalculate\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bmean\b",
        r"\bmedian\b",
        r"\bpercentage\b",
        r"\bpercent\b",
        r"\bdifference\b",
        r"\bgrowth\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)
