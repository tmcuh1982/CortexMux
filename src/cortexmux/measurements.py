"""Exact scalar conversions and comparisons independent of model providers."""

from __future__ import annotations

import re
from fractions import Fraction
from types import MappingProxyType
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

_MAX_VALUE_CHARS = 64
_MAX_UNIT_CHARS = 16
_MAX_DISPLAY_CHARS = 128
_DECIMAL = re.compile(r"[+\-\u2212]?[0-9]+(?:[.,][0-9]+)?")

DisplayFormat = Literal["decimal", "rational"]
IssueCode = Literal[
    "unsupported_kind",
    "unsupported_unit",
    "invalid_number",
    "below_absolute_zero",
    "output_limit",
]


class _Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class ScalarMeasurement(_Record):
    """Original scalar value, unit and explicitly declared quantity kind."""

    quantity_kind: str = Field(min_length=1, max_length=32, repr=False)
    value: str = Field(max_length=_MAX_VALUE_CHARS, repr=False)
    unit: str = Field(max_length=_MAX_UNIT_CHARS, repr=False)


class MeasurementComparison(_Record):
    """Two independently declared scalar measurements."""

    left: ScalarMeasurement
    right: ScalarMeasurement


class MeasurementPresentation(_Record):
    """Whitelisted display policy for exact values."""

    format_version: Literal[1] = 1
    style: Literal["decimal_if_finite", "rational"] = "decimal_if_finite"


class MeasurementContext(_Record):
    """Explicit structured-call enrichment using original inputs and a display policy."""

    comparison: MeasurementComparison
    presentation: MeasurementPresentation = Field(default_factory=MeasurementPresentation)
    capture_private: bool = False


class MeasurementProvenance(_Record):
    """Independent revisions of the arithmetic, unit registry and display algorithms."""

    calculator_version: Literal[1] = 1
    unit_registry_version: Literal[1] = 1
    presentation_version: Literal[1] = 1


class ExactRatio(_Record):
    """Reduced fraction with a positive denominator, serialized as integer strings."""

    numerator: str = Field(repr=False)
    denominator: str = Field(repr=False)


class ExactDisplay(_Record):
    """Exact display text in the named notation."""

    text: str = Field(repr=False)
    format: DisplayFormat
    exact: Literal[True] = True


class ExactMeasurement(_Record):
    """Exact value in a specified unit and its display form."""

    unit: str
    exact: ExactRatio
    display: ExactDisplay


class MeasurementIssue(_Record):
    """Safe machine-readable reason for an unsupported input or output."""

    operand: Literal["left", "right", "source", "target"]
    code: IssueCode


class MeasurementComparisonResult(_Record):
    """Versioned numerical comparison with original operands and optional proof."""

    contract: Literal["exact-measurement-comparison/1"] = "exact-measurement-comparison/1"
    status: Literal["equal", "different", "incompatible_dimensions", "unsupported"]
    original: MeasurementComparison
    presentation: MeasurementPresentation
    quantity_kind: str | None = None
    canonical_unit: str | None = None
    left: ExactMeasurement | None = None
    right: ExactMeasurement | None = None
    tolerance_applied: Literal[False] = False
    issue: MeasurementIssue | None = None
    provenance: MeasurementProvenance = Field(default_factory=MeasurementProvenance)


class MeasurementConversionResult(_Record):
    """Versioned scalar conversion retaining the input and requested target unit."""

    contract: Literal["exact-measurement-conversion/1"] = "exact-measurement-conversion/1"
    status: Literal["converted", "incompatible_dimensions", "unsupported"]
    original: ScalarMeasurement
    target_unit: str = Field(max_length=_MAX_UNIT_CHARS, repr=False)
    presentation: MeasurementPresentation
    result: ExactMeasurement | None = None
    issue: MeasurementIssue | None = None
    provenance: MeasurementProvenance = Field(default_factory=MeasurementProvenance)


class _Unit(NamedTuple):
    quantity_kind: Literal["absolute_temperature", "linear_speed"]
    canonical_unit: Literal["K", "m/s"]
    scale: Fraction
    offset: Fraction = Fraction(0)


# canonical = original * scale + offset
_ICE_POINT = Fraction(27315, 100)
_UNITS = MappingProxyType(
    {
        "K": _Unit("absolute_temperature", "K", Fraction(1)),
        "°C": _Unit("absolute_temperature", "K", Fraction(1), _ICE_POINT),
        "°F": _Unit("absolute_temperature", "K", Fraction(5, 9), _ICE_POINT - 32 * Fraction(5, 9)),
        "m/s": _Unit("linear_speed", "m/s", Fraction(1)),
        "km/h": _Unit("linear_speed", "m/s", Fraction(5, 18)),
        "cm/s": _Unit("linear_speed", "m/s", Fraction(1, 100)),
        "mm/s": _Unit("linear_speed", "m/s", Fraction(1, 1000)),
    }
)


class _Unsupported(ValueError):
    def __init__(self, code: IssueCode) -> None:
        self.code = code


def _parse_decimal(value: str) -> Fraction:
    if not _DECIMAL.fullmatch(value):
        raise _Unsupported("invalid_number")
    return Fraction(value.replace("\u2212", "-").replace(",", "."))


def _canonical(measurement: ScalarMeasurement) -> tuple[_Unit, Fraction]:
    if measurement.quantity_kind not in {"absolute_temperature", "linear_speed"}:
        raise _Unsupported("unsupported_kind")
    unit = _UNITS.get(measurement.unit)
    if unit is None or unit.quantity_kind != measurement.quantity_kind:
        raise _Unsupported("unsupported_unit")
    value = _parse_decimal(measurement.value) * unit.scale + unit.offset
    if unit.quantity_kind == "absolute_temperature" and value < 0:
        raise _Unsupported("below_absolute_zero")
    return unit, value


def _finite_decimal(value: Fraction) -> str | None:
    remaining = value.denominator
    twos = fives = 0
    while remaining % 2 == 0:
        twos += 1
        remaining //= 2
    while remaining % 5 == 0:
        fives += 1
        remaining //= 5
    if remaining != 1:
        return None
    places = max(twos, fives)
    if places == 0:
        return str(value.numerator)
    scaled = abs(value.numerator) * 2 ** (places - twos) * 5 ** (places - fives)
    digits = str(scaled).zfill(places + 1)
    sign = "-" if value < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _exact(value: Fraction, unit: str, policy: MeasurementPresentation) -> ExactMeasurement:
    numerator, denominator = str(value.numerator), str(value.denominator)
    rational = f"{numerator}/{denominator}"
    decimal = _finite_decimal(value)
    if max(len(rational), len(decimal or "")) > _MAX_DISPLAY_CHARS:
        raise _Unsupported("output_limit")
    return ExactMeasurement(
        unit=unit,
        exact=ExactRatio(numerator=numerator, denominator=denominator),
        display=(
            ExactDisplay(text=decimal, format="decimal")
            if decimal is not None and policy.style == "decimal_if_finite"
            else ExactDisplay(text=rational, format="rational")
        ),
    )


def compare_measurements(
    comparison: MeasurementComparison,
    *,
    presentation: MeasurementPresentation = MeasurementPresentation(),
) -> MeasurementComparisonResult:
    """Recalculate both operands and compare compatible quantities without tolerance."""
    comparison = MeasurementComparison.model_validate(comparison)
    presentation = MeasurementPresentation.model_validate(presentation)
    side: Literal["left", "right"] = "left"
    try:
        unit, left = _canonical(comparison.left)
        side = "right"
        right_unit, right = _canonical(comparison.right)
        if unit.quantity_kind != right_unit.quantity_kind:
            return MeasurementComparisonResult(
                status="incompatible_dimensions", original=comparison, presentation=presentation
            )
        side = "left"
        left_proof = _exact(left, unit.canonical_unit, presentation)
        side = "right"
        right_proof = _exact(right, unit.canonical_unit, presentation)
    except _Unsupported as error:
        return MeasurementComparisonResult(
            status="unsupported",
            original=comparison,
            presentation=presentation,
            issue=MeasurementIssue(operand=side, code=error.code),
        )
    return MeasurementComparisonResult(
        status="equal" if left == right else "different",
        original=comparison,
        presentation=presentation,
        quantity_kind=unit.quantity_kind,
        canonical_unit=unit.canonical_unit,
        left=left_proof,
        right=right_proof,
    )


def convert_measurement(
    source: ScalarMeasurement,
    target_unit: str,
    *,
    presentation: MeasurementPresentation = MeasurementPresentation(),
) -> MeasurementConversionResult:
    """Convert a scalar directly to a supported target unit using exact arithmetic."""
    source = ScalarMeasurement.model_validate(source)
    presentation = MeasurementPresentation.model_validate(presentation)
    if not isinstance(target_unit, str) or len(target_unit) > _MAX_UNIT_CHARS:
        raise ValueError(f"Target unit must be a string of at most {_MAX_UNIT_CHARS} characters.")
    side: Literal["source", "target"] = "source"
    try:
        unit, canonical = _canonical(source)
        side = "target"
        target = _UNITS.get(target_unit)
        if target is None:
            raise _Unsupported("unsupported_unit")
        if target.quantity_kind != unit.quantity_kind:
            return MeasurementConversionResult(
                status="incompatible_dimensions",
                original=source,
                target_unit=target_unit,
                presentation=presentation,
            )
        result = _exact((canonical - target.offset) / target.scale, target_unit, presentation)
    except _Unsupported as error:
        return MeasurementConversionResult(
            status="unsupported",
            original=source,
            target_unit=target_unit,
            presentation=presentation,
            issue=MeasurementIssue(operand=side, code=error.code),
        )
    return MeasurementConversionResult(
        status="converted",
        original=source,
        target_unit=target_unit,
        presentation=presentation,
        result=result,
    )
