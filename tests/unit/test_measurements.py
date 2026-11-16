"""Exact measurement contracts, including rejection and presentation boundaries."""

from decimal import localcontext
from fractions import Fraction

import pytest
from pydantic import ValidationError

from cortexmux.measurements import (
    MeasurementComparison,
    MeasurementContext,
    MeasurementPresentation,
    ScalarMeasurement,
    compare_measurements,
    convert_measurement,
)

TEMPERATURE = "absolute_temperature"
SPEED = "linear_speed"


@pytest.mark.parametrize(
    "kind,left,left_unit,right,right_unit,status,expected",
    [
        (TEMPERATURE, "-15", "°C", "5", "°F", "equal", ("258.15", "258.15")),
        (TEMPERATURE, "-25", "°C", "-13", "°F", "equal", ("248.15", "248.15")),
        (TEMPERATURE, "-40", "°C", "-40", "°F", "equal", ("233.15", "233.15")),
        (TEMPERATURE, "0", "°C", "273.15", "K", "equal", ("273.15", "273.15")),
        (TEMPERATURE, "10", "°C", "50", "°F", "equal", ("283.15", "283.15")),
        (TEMPERATURE, "-15", "°C", "5.18", "°F", "different", ("258.15", "258.25")),
        (TEMPERATURE, "\u221215", "°C", "-15,0", "°C", "equal", ("258.15", "258.15")),
        (TEMPERATURE, "-273.15", "°C", "0", "K", "equal", ("0", "0")),
        (SPEED, "2", "m/s", "7.2", "km/h", "equal", ("2", "2")),
        (SPEED, "1.25", "m/s", "125", "cm/s", "equal", ("1.25", "1.25")),
        (SPEED, "0.8", "m/s", "800", "mm/s", "equal", ("0.8", "0.8")),
        (SPEED, "2", "m/s", "7.3", "km/h", "different", ("2", "73/36")),
        (SPEED, "0.2", "m/s", "20", "cm/s", "equal", ("0.2", "0.2")),
        (SPEED, "-0.125", "m/s", "-125", "mm/s", "equal", ("-0.125", "-0.125")),
        (SPEED, "-0.00", "m/s", "+000,00", "km/h", "equal", ("0", "0")),
    ],
)
def test_exact_comparison(kind, left, left_unit, right, right_unit, status, expected):
    a = ScalarMeasurement(quantity_kind=kind, value=left, unit=left_unit)
    b = ScalarMeasurement(quantity_kind=kind, value=right, unit=right_unit)
    with localcontext() as decimal_context:
        decimal_context.prec = 1
        pair = MeasurementComparison(left=a, right=b)
        result = compare_measurements(pair)
        reversed_result = compare_measurements(MeasurementComparison(left=b, right=a))
    assert result.original.model_dump() == pair.model_dump()
    assert result.status == reversed_result.status == status
    assert result.left == reversed_result.right and result.right == reversed_result.left
    assert (result.left.display.text, result.right.display.text) == expected
    assert result.tolerance_applied is False
    for item in (result.left, result.right):
        assert item.display.exact is True
        assert item.display.format == ("rational" if "/" in item.display.text else "decimal")
        assert Fraction(item.display.text) == Fraction(
            int(item.exact.numerator), int(item.exact.denominator)
        )


@pytest.mark.parametrize(
    "kind,value,unit,target,converted",
    [
        (TEMPERATURE, "-15", "°C", "°F", "5"),
        (TEMPERATURE, "-25", "°C", "K", "248.15"),
        (TEMPERATURE, "-40", "°F", "°C", "-40"),
        (SPEED, "2", "m/s", "km/h", "7.2"),
        (SPEED, "1.25", "m/s", "cm/s", "125"),
        (SPEED, "800", "mm/s", "m/s", "0.8"),
    ],
)
def test_exact_conversion_round_trip(kind, value, unit, target, converted):
    source = ScalarMeasurement(quantity_kind=kind, value=value, unit=unit)
    forward = convert_measurement(source, target)
    assert forward.status == "converted"
    assert forward.original == source
    assert forward.result.display.text == converted
    reverse = convert_measurement(
        ScalarMeasurement(quantity_kind=kind, value=converted, unit=target), unit
    )
    assert reverse.status == "converted"
    assert Fraction(reverse.result.display.text) == Fraction(value)


def test_presentation_policy_changes_display_only():
    pair = MeasurementComparison(
        left=ScalarMeasurement(quantity_kind=TEMPERATURE, value="-15", unit="°C"),
        right=ScalarMeasurement(quantity_kind=TEMPERATURE, value="5", unit="°F"),
    )
    result = compare_measurements(pair)
    ratio = compare_measurements(pair, presentation=MeasurementPresentation(style="rational"))
    assert ratio.left.exact == result.left.exact
    assert ratio.right.exact == result.right.exact
    assert ratio.status == result.status
    assert ratio.left.exact.numerator == "5163"
    assert ratio.left.exact.denominator == "20"
    assert ratio.left.display.text == "5163/20"
    assert ratio.left.display.format == "rational"
    assert result.provenance.model_dump() == {
        "calculator_version": 1,
        "unit_registry_version": 1,
        "presentation_version": 1,
    }


def test_incompatible_and_unknown_conversions():
    speed = ScalarMeasurement(quantity_kind=SPEED, value="2", unit="m/s")
    temperature = ScalarMeasurement(quantity_kind=TEMPERATURE, value="2", unit="K")
    pair = MeasurementComparison(left=temperature, right=speed)
    result = compare_measurements(pair)
    assert result.status == "incompatible_dimensions"
    assert result.left is result.right is result.canonical_unit is None
    assert convert_measurement(temperature, "m/s").status == "incompatible_dimensions"
    assert convert_measurement(speed, "MPH").issue.code == "unsupported_unit"
    recurring = convert_measurement(
        ScalarMeasurement(quantity_kind=SPEED, value="7.3", unit="km/h"), "m/s"
    )
    assert recurring.result.display.text == "73/36"
    assert recurring.result.display.format == "rational"


@pytest.mark.parametrize(
    "kind,value,unit,issue",
    [
        (TEMPERATURE, "10", "", "unsupported_unit"),
        (TEMPERATURE, "10", "C", "unsupported_unit"),
        (TEMPERATURE, "10", " °F", "unsupported_unit"),
        (TEMPERATURE, "10", "m/s", "unsupported_unit"),
        (TEMPERATURE, "-0.01", "K", "below_absolute_zero"),
        (TEMPERATURE, "-274", "°C", "below_absolute_zero"),
        (TEMPERATURE, "-459.68", "°F", "below_absolute_zero"),
        ("temperature_delta", "10", "°C", "unsupported_kind"),
        ("angular_speed", "2", "rad/s", "unsupported_kind"),
        (TEMPERATURE, "\u221220° to 40°C", "°C", "invalid_number"),
        (SPEED, "about 2 m/s", "m/s", "invalid_number"),
        (SPEED, "2 ± 0.1 m/s", "m/s", "invalid_number"),
        *[
            (SPEED, value, "m/s", "invalid_number")
            for value in (
                "",
                "--2",
                "2.",
                ".2",
                "1_000",
                "1,234.5",
                "1 000",
                " 2",
                "2\n",
                "½",
                "٢",
                "2/3",
                "1e2",
                "NaN",
                "inf",
                "-Infinity",
            )
        ],
    ],
)
def test_unsupported_is_never_a_different_number(kind, value, unit, issue):
    valid = ScalarMeasurement(quantity_kind=SPEED, value="2", unit="m/s")
    invalid = ScalarMeasurement(quantity_kind=kind, value=value, unit=unit)
    result = compare_measurements(MeasurementComparison(left=valid, right=invalid))
    assert result.status == "unsupported"
    assert result.issue.model_dump() == {"operand": "right", "code": issue}
    assert result.original.right.value == value
    assert result.left is result.right is None


@pytest.mark.parametrize("style", ["decimal_if_finite", "rational"])
def test_output_budget_is_independent_of_style(style):
    pair = MeasurementComparison(
        left=ScalarMeasurement(quantity_kind=TEMPERATURE, value="10", unit="K"),
        right=ScalarMeasurement(quantity_kind=TEMPERATURE, value="0." + "3" * 62, unit="°F"),
    )
    result = compare_measurements(pair, presentation=MeasurementPresentation(style=style))
    assert result.status == "unsupported"
    assert result.issue.model_dump() == {"operand": "right", "code": "output_limit"}


@pytest.mark.parametrize(
    "data",
    [
        {"quantity_kind": SPEED, "value": 2.0, "unit": "m/s"},
        {"quantity_kind": SPEED, "value": True, "unit": "m/s"},
        {"quantity_kind": SPEED, "value": 2, "unit": "m/s"},
        {"quantity_kind": SPEED, "value": "2"},
        {"quantity_kind": SPEED, "value": "9" * 65, "unit": "m/s"},
        {"value": "2", "unit": "m/s"},
    ],
)
def test_structural_validation_is_strict(data):
    with pytest.raises(ValidationError):
        ScalarMeasurement.model_validate(data)


def test_untrusted_precomputed_result_cannot_be_supplied():
    with pytest.raises(ValidationError):
        MeasurementContext.model_validate(
            {
                "comparison": {
                    side: {"quantity_kind": SPEED, "value": "2", "unit": "m/s"}
                    for side in ("left", "right")
                },
                "result": {"status": "equal"},
            }
        )
