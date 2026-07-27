# Data analysis

Supported files are CSV, JSON, JSONL/NDJSON, Parquet, and Excel, plus installed
Pandas or Polars dataframes. Pandas is the baseline. In `auto`, Pandas handles
normal inputs; installed DuckDB handles Parquet or files at/above the configured
large-file threshold, and installed Polars is the next large-file option.

Plans can inspect schema/shape, describe values, summarize missingness and
uniqueness, count values, correlate, group and aggregate, sort/top-N, apply
bounded filters, parse dates, aggregate time series, detect IQR outliers, and
prepare histogram/bar/line series.

Every plan is validated for operation, columns, operators, aggregations, steps,
and output limits. Reports include engine, profile, results, warnings, plan,
chart paths, and disclosure metadata.

