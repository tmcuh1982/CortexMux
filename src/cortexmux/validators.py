"""Reusable deterministic decisions for application-declared fields."""

from cortexmux.harness import HarnessDecision
from cortexmux.measurements import MeasurementComparison, ScalarMeasurement, compare_measurements


def exact_copy(source: str, candidate: str) -> HarnessDecision:
    """Require an entire literal field to match, including case and whitespace."""
    if not source.strip():
        return HarnessDecision(action="review", reason="Missing source evidence for exact copy.")
    if source == candidate:
        return HarnessDecision(action="accept", reason="Exact source copy.")
    return HarnessDecision(
        action="retry", reason=f"Copy {source!r} exactly; received {candidate!r}."
    )


def verified_quantity(source: ScalarMeasurement, candidate: ScalarMeasurement) -> HarnessDecision:
    """Compare declared scalar measurements with CortexMux's exact unit engine."""
    proof = compare_measurements(MeasurementComparison(left=source, right=candidate))
    if proof.status == "equal":
        return HarnessDecision(action="accept", reason="Exact quantities match.")
    if proof.status == "unsupported":
        return HarnessDecision(action="review", reason="Unsupported scalar measurement.")
    return HarnessDecision(
        action="retry",
        reason=f"Quantity mismatch ({proof.status}). Preserve {source.value} {source.unit}.",
    )
