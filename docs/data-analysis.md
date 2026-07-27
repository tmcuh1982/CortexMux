# Data analysis

Supported inputs are CSV, JSON, JSONL/NDJSON, Parquet, Excel, extracted
`list[dict]`/`dict` records, and installed Pandas or Polars dataframes. This
allows a website ingestion layer to pass already-extracted text and numbers
without granting CortexMux arbitrary network access. CortexMux does not fetch
web pages itself.

Pandas is the baseline. In `auto`, Pandas handles normal inputs; installed
DuckDB handles Parquet or files at/above the configured large-file threshold,
and installed Polars is the next large-file option.

Plans can inspect schema/shape, describe values, summarize missingness and
uniqueness, count values, correlate, group and aggregate, sort/top-N, apply
bounded filters, parse dates, aggregate time series, detect IQR outliers, and
prepare histogram/bar/line series.

Every plan is validated for operation, columns, operators, aggregations, steps,
and output limits. The primary outputs are structured results, optional model
interpretation, warnings, disclosure metadata, and mathematical verifications.
The existing Markdown report and chart fields remain for 0.1 compatibility,
but they are not the focus of the 0.2 web-data workflow.

## Mathematical verification

When `verify_calculations=True` (the default), model interpretation uses a
structured response. Every AI-generated numerical conclusion must declare:

- a whitelisted operation;
- operands referenced by JSON Pointer under `/profile` or `/results`;
- any genuine literal constants;
- the result claimed by the model;
- optional decimal rounding.

CortexMux independently resolves the operands and recalculates with bounded
`Decimal` arithmetic. It never evaluates a model-provided expression. Each
claim is returned as `verified`, `incorrect`, or `unverifiable`.

```python
response = mux.analyze_data(
    extracted_records,
    instruction="Calcule le pourcentage d'évolution du prix moyen.",
    interpretation_provider="ollama",
    interpretation_model="qwen2.5-coder:7b",
    require_verified_calculations=True,
)

print(response.interpretation)
print(response.math_verification_passed)
print(response.calculation_verifications)
```

Strict mode rejects the AI interpretation unless at least one calculation was
declared and every claim was verified. Without strict mode, incorrect or
unverifiable claims remain explicit in the response and generate warnings. The
unsafe text is withheld from `interpretation` and retained only in
`unverified_interpretation` for auditing.

The checker is also available directly for an AI response produced elsewhere:

```python
verification = mux.verify_calculation(
    {
        "label": "Évolution",
        "operation": "percentage_change",
        "operands": [
            {"source_path": "/results/0/current"},
            {"source_path": "/results/0/previous"},
        ],
        "claimed_result": 20,
    },
    {"profile": {}, "results": [{"current": 120, "previous": 100}]},
)
```
