"""Deterministic Markdown reporting."""

from __future__ import annotations

from cortexmux.providers.data.schemas import AnalysisPlan, DataProfile


def markdown_report(
    profile: DataProfile,
    plan: AnalysisPlan,
    *,
    engine: str,
    warnings: list[str],
    interpretation: str | None,
) -> str:
    """Render a concise Markdown report without model-generated formatting code."""
    lines = [
        "# CortexMux Data Analysis",
        "",
        f"- Engine: `{engine}`",
        f"- Rows: {profile.rows:,}",
        f"- Columns: {profile.columns:,}",
        f"- Plan steps: {len(plan.steps)}",
        f"- Redacted columns: {', '.join(profile.redacted_columns) or 'none'}",
        "",
        "## Schema",
        "",
        "| Column | Type | Missing |",
        "|---|---:|---:|",
    ]
    for column, dtype in profile.schema_info.items():
        display = "[REDACTED]" if column in profile.redacted_columns else column
        lines.append(f"| {display} | {dtype} | {profile.missing_values.get(column, 0)} |")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    if interpretation:
        lines.extend(["", "## Interpretation", "", interpretation])
    else:
        lines.extend(
            ["", "## Interpretation", "", "No language-model interpretation was requested."]
        )
    return "\n".join(lines) + "\n"
