# Migrating from 0.5.1 to 0.5.2

Version 0.5.2 is source-compatible for existing supported OpenAI options and
provider-neutral requests. It intentionally turns previously ignored invalid
OpenAI options and unsupported JSON Schema constraints into explicit errors.

## OpenAI request options

Existing calls using `max_output_tokens`, `reasoning_effort`, and `verbosity`
continue to work when their values are valid. `service_tier` is new. Use
`service_tier="default"` to request Standard processing, including deployments
that require the non-Fast path for EU data residency.

The convenience APIs now accept request controls directly:

```python
response = mux.generate(
    "Summarize this record.",
    provider="openai",
    model="gpt-6-astra",
    system="Be concise and factual.",
    service_tier="default",
    reasoning_effort="low",
    timeout=30,
)
```

`structured()` accepts the same provider options through keyword arguments.
`timeout` is a request field, not an OpenAI payload option. It overrides the
provider timeout for every retry attempt.

## Intentional incompatibilities

- Unknown OpenAI option keys now raise `InvalidRequestError` before network I/O.
- Invalid enum values for `service_tier`, `reasoning_effort`, or `verbosity` now
  raise `InvalidRequestError` instead of being ignored.
- GPT-6 Astra now always sends an effective reasoning effort, defaulting to
  `low`; `none` and `minimal` are rejected.
- Unsupported JSON Schema validation keywords now fail explicitly. Review
  [the supported subset](providers.md#locally-validated-json-schema-subset).
- JSON boolean values no longer pass `integer` or `number` schemas.

Pydantic response models remain the final application-level validation layer.
No OpenAI tools/function calling, streaming, PostgreSQL storage, or
UniversRobot sensitivity routing is added by this release.
