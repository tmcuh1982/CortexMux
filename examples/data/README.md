# Regression CSV fixtures

This directory contains deterministic CSV fixtures for examples and regression
tests. Every file whose name ends in `_synthetic.csv` contains invented data and
marks each row with `is_synthetic=true`.

## Fixtures

| File | Rows | Intended checks |
| --- | ---: | --- |
| `revenue_monthly_synthetic.csv` | 24 | Monthly revenue components, totals, customers, and churn |
| `population_statistics_synthetic.csv` | 32 | Four fictional regions over eight years |
| `weather_observations_synthetic.csv` | 90 | Three fictional stations over 30 days |
| `visa_stock_prices_synthetic.csv` | 260 | Synthetic weekday OHLCV series with 50- and 200-day moving averages |

`demo_sales.csv` is the smaller end-to-end demo input used by
`examples/full_local_demo.py`.

## Stable reference values

The regression tests intentionally pin these values:

- Revenue grand total: `4,893,900 EUR`.
- Total fictional population in 2025: `27,443,368`.
- Total precipitation: `318.5 mm`.
- Final synthetic Visa row: `2025-12-31`, close `293.35 USD`,
  SMA50 `292.9594`, and SMA200 `279.1324`.

The stock fixture uses `V` only as a familiar analysis scenario. It is not real
Visa market data, does not follow an exchange holiday calendar, and must not be
used for financial decisions. Dates include weekdays only. `sma_50` and
`sma_200` are simple arithmetic means of the current and preceding 49 or 199
`close_usd` observations; cells remain empty until a complete window exists.
