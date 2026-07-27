"""Regression checks for the deterministic example CSV fixtures."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "examples" / "data"
FOUR_PLACES = Decimal("0.0001")


def read_rows(name: str) -> list[dict[str, str]]:
    """Read a bundled CSV fixture without optional dataframe dependencies."""
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize(
    ("name", "expected_rows"),
    [
        ("revenue_monthly_synthetic.csv", 24),
        ("population_statistics_synthetic.csv", 32),
        ("weather_observations_synthetic.csv", 90),
        ("visa_stock_prices_synthetic.csv", 260),
    ],
)
def test_fixture_is_present_complete_and_synthetic(name: str, expected_rows: int) -> None:
    rows = read_rows(name)
    assert len(rows) == expected_rows
    assert all(row["is_synthetic"] == "true" for row in rows)
    assert all(all(value is not None for value in row.values()) for row in rows)


def test_revenue_components_and_reference_total() -> None:
    rows = read_rows("revenue_monthly_synthetic.csv")
    months = [row["month"] for row in rows]

    assert months == sorted(months)
    assert len(months) == len(set(months))
    assert sum(Decimal(row["total_revenue_eur"]) for row in rows) == Decimal("4893900")

    for row in rows:
        components = (
            Decimal(row["subscription_revenue_eur"])
            + Decimal(row["transaction_revenue_eur"])
            + Decimal(row["services_revenue_eur"])
        )
        assert components == Decimal(row["total_revenue_eur"])
        assert Decimal("0") <= Decimal(row["churn_rate"]) <= Decimal("1")


def test_population_components_trends_and_reference_total() -> None:
    rows = read_rows("population_statistics_synthetic.csv")
    by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
        assert int(row["urban_population"]) + int(row["rural_population"]) == int(
            row["population_total"]
        )

    assert set(by_region) == {"Auroria", "Borealia", "Cyrenia", "Demeria"}
    for region_rows in by_region.values():
        assert [int(row["year"]) for row in region_rows] == list(range(2018, 2026))
        populations = [int(row["population_total"]) for row in region_rows]
        assert populations == sorted(populations)

    total_2025 = sum(int(row["population_total"]) for row in rows if row["year"] == "2025")
    assert total_2025 == 27_443_368


def test_weather_ranges_coverage_and_reference_precipitation() -> None:
    rows = read_rows("weather_observations_synthetic.csv")
    station_counts = Counter(row["station"] for row in rows)

    assert station_counts == {"Nordville": 30, "Lac-Central": 30, "Solaria": 30}
    assert sum(Decimal(row["precipitation_mm"]) for row in rows) == Decimal("318.5")

    for row in rows:
        minimum = Decimal(row["temp_min_c"])
        maximum = Decimal(row["temp_max_c"])
        mean = Decimal(row["temp_mean_c"])
        assert minimum <= mean <= maximum
        assert mean == ((minimum + maximum) / 2).quantize(Decimal("0.1"))
        assert Decimal(row["precipitation_mm"]) >= 0
        assert Decimal("0") <= Decimal(row["humidity_pct"]) <= Decimal("100")


def moving_average(closes: list[Decimal], index: int, window: int) -> Decimal | None:
    """Return a four-decimal SMA once a complete window is available."""
    if index + 1 < window:
        return None
    values = closes[index + 1 - window : index + 1]
    return (sum(values) / Decimal(window)).quantize(FOUR_PLACES, rounding=ROUND_HALF_UP)


def test_stock_ohlcv_and_moving_averages_are_reproducible() -> None:
    rows = read_rows("visa_stock_prices_synthetic.csv")
    dates = [date.fromisoformat(row["date"]) for row in rows]
    closes = [Decimal(row["close_usd"]) for row in rows]

    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))
    assert all(item.weekday() < 5 for item in dates)
    assert all(row["ticker"] == "V" for row in rows)

    for index, row in enumerate(rows):
        open_price = Decimal(row["open_usd"])
        high = Decimal(row["high_usd"])
        low = Decimal(row["low_usd"])
        close = closes[index]
        assert low <= open_price <= high
        assert low <= close <= high
        assert int(row["volume"]) > 0

        expected_sma_50 = moving_average(closes, index, 50)
        expected_sma_200 = moving_average(closes, index, 200)
        assert (None if row["sma_50"] == "" else Decimal(row["sma_50"])) == expected_sma_50
        assert (None if row["sma_200"] == "" else Decimal(row["sma_200"])) == expected_sma_200

    assert rows[-1] == {
        "date": "2025-12-31",
        "ticker": "V",
        "open_usd": "293.9",
        "high_usd": "295.48",
        "low_usd": "292.15",
        "close_usd": "293.35",
        "volume": "7360529",
        "sma_50": "292.9594",
        "sma_200": "279.1324",
        "is_synthetic": "true",
    }
