# Exact scalar measurements

## Architecture and activation

```text
Validated operands → cortexmux.measurements → versioned exact result
                                            ↘ exact display
CortexMux.structured / astructured → append numerical evidence to prompt
                                   → existing router → selected provider
                                   → existing response and audit
```

`compare_measurements` and `convert_measurement` are local, synchronous functions
with no provider or network dependency. They use the same unit registry,
rational arithmetic and presentation for every model. Call them directly when
the application only needs a numerical result.

Pass `measurement_context` explicitly to the non-streaming `structured` or
`astructured` facade to append one comparison. The facade recalculates from the
supplied original operands on every call, before routing. There is no field for
a client-supplied proof, status or formula. Original prompt text stays an exact
prefix of the enriched prompt. Model-facing evidence contains the display values,
units, status and tolerance flag. The reduced fractions are available in the
local result and optional private audit. The existing router, response schema,
model selection, endpoint and request options apply normally.

```python
from cortexmux.measurements import (
    MeasurementComparison,
    MeasurementContext,
    ScalarMeasurement,
    compare_measurements,
    convert_measurement,
)

pair = MeasurementComparison(
    left=ScalarMeasurement(quantity_kind="absolute_temperature", value="-15", unit="°C"),
    right=ScalarMeasurement(quantity_kind="absolute_temperature", value="5", unit="°F"),
)

proof = compare_measurements(pair)
assert proof.status == "equal"
assert proof.left is not None and proof.left.display.text == "258.15"
fahrenheit = convert_measurement(pair.left, "°F")
assert fahrenheit.result is not None and fahrenheit.result.display.text == "5"

# Use the application's configured mux, prompt with source quotes, and schema.
response = mux.structured(
    original_prompt,
    json_schema=business_schema,
    measurement_context=MeasurementContext(comparison=pair, capture_private=True),
    **existing_route_and_options,
)
```

Structured streaming and generic `run` / `arun` are outside this integration.
The standalone calculator can still be called before either flow. The enriched
facade itself is provider-neutral. Transport parity and prompt hash fidelity
were exercised against MLX-VLM with MockTransport.

## Inputs, statuses and exact presentation

Each operand declares its own kind. Supported kinds and exact unit tokens are
`absolute_temperature` with `°C`, `°F` or `K`, and `linear_speed` with `m/s`,
`km/h`, `cm/s` or `mm/s`. Unit tokens are case sensitive and have no aliases.
Comparison canonicalizes to `K` and `m/s` respectively. Conversion takes a target
unit explicitly. Temperature below absolute zero is outside the supported domain.

Values must be strings containing ASCII digits with an optional initial `+`,
ASCII `-` or U+2212 minus sign. A single dot or comma may separate integer and
fractional digits. Each side of the separator requires a digit. A comma always
means a decimal separator. Thousands separators, whitespace, exponents,
rational input, NaN and infinity are rejected. Keep original quotes and their
source attribution in the application. The calculator receives structured
operands, so text in a quote cannot change a factor or policy. Ranges, deltas,
angular speed, inferred units, approximations and tolerances are outside scope.

| Status | Meaning |
| --- | --- |
| `equal` / `different` | Compatible inputs compared as reduced fractions |
| `converted` | Exact conversion succeeded |
| `incompatible_dimensions` | The valid inputs or source and target have different dimensions |
| `unsupported` | A bounded input or output failed a numerical rule, with an operand and issue code |

The first unsupported operand in processing order supplies the issue.
`unsupported_kind`, `unsupported_unit`, `invalid_number`,
`below_absolute_zero` and `output_limit` are defined in the contract.
For structurally invalid operands, the Pydantic input models raise
`ValidationError`. Required fields, string types and lengths are checked at
that boundary, including numeric values of at most 64 characters, unit tokens
of at most 16 and kinds of at most 32. The standalone conversion target also
raises `ValueError` for a wrong type or oversized token. Proof notation is capped
at 128 characters per value in either representation. Both display policies
apply the same bound before returning a proof.

The common policy, `decimal_if_finite`, uses a decimal when the reduced
denominator contains only factors 2 and 5. Otherwise it emits a reduced
`numerator/denominator` tagged as `rational`. For example, `5163/20` becomes
`258.15` and `7.3 km/h` becomes `73/36 m/s`. The `rational` policy displays
`numerator/denominator` for every value, including integers. Both policies use
integer arithmetic. Exact zero displays as `0` or `0/1`, with the original
input still preserved. JSON ratio components are strings. All successful results
carry the original operands and presentation policy. Comparison explicitly
sets `tolerance_applied=false`. Standalone conversion and comparison use
`exact-measurement-conversion/1` and `exact-measurement-comparison/1`.

## Profiles and evidence

A reviewed [behavior profile](behavior-profiles.md) can declare
`rules.measurement_presentation`, for example
`{"format_version":1,"style":"decimal_if_finite"}`. On a call that opts in,
the application can explicitly pass
`presentation=profile.rules.measurement_presentation or MeasurementPresentation()`
to `MeasurementContext`. The binding continues to enforce the existing identity
and options. Declarations participate in both profile fingerprints. Existing
manifests with no declaration retain their canonical export and fingerprints.
Increment the profile revision and obtain new attestations when changing rules.
The bundled Qwen and Magistral revision 2 manifests keep their existing
offline syntax qualification. A live deployment still needs a suitable
attestation under its existing binding policy.

The calculator, registry and presentation each have an independent integer
version in the result's `provenance`. Preserve these versions and the contract
identifier with exported proof. Recalculate a serialized result from the
original inputs and pinned policy to verify it. A version or a status alone
does not establish proof.

The structured facade records `raw_metadata["measurement"]` on success and
`CortexMuxError.context["measurement"]` on failures after preparation. It
includes versions, status, presentation, effective formats, any safe issue code
and the SHA-256 of the UTF-8 enriched prompt passed to the adapter. For MLX-VLM
this is exactly the decoded user message content. The hash covers no schema,
system message or HTTP envelope, and a hash on a routing or preflight error
identifies the prepared text without attesting transmission. For any additional
adapter, verify its prompt handling before using the hash as transport evidence.

The trace omits source text and values. With `MeasurementContext(capture_private=True, ...)`,
the exact proof and originals are available via `response.private_audit["measurement"]`
or `error.private_audit["measurement"]`. Protect this audit in the application.
MLX raw HTTP capture remains separately controlled by the existing
`capture_private=True` call option on a behavior-profile call. CortexMux itself
does not persist either audit. If the model disagrees with the calculation, its
response and the Python proof remain independently accessible.

For `unsupported` or `incompatible_dimensions`, enrichment communicates an
unknown outcome without canonical values. Consumers must handle these states
explicitly. Invalid context at the facade yields `InvalidRequestError` with
`invalid_context`. An unexpected preparation failure yields
`MeasurementCalculationError` with `preparation_error` and stops before provider
I/O. These errors retain safe version information.

## Requalification

Start with the eight exportable examples in
[`exact_measurements_synthetic.json`](../examples/data/exact_measurements_synthetic.json).
The fixtures pin exact displays, statuses and separate synthetic business
expectations. Run the offline checks with:

```bash
.venv/bin/pytest -q tests/unit/test_measurements.py \
  tests/unit/providers/mlx/test_behavior_profiles.py
.venv/bin/python examples/exact_measurements.py
```

On a separately approved MLX-VLM deployment, prepare a local config with the
exact server version, model and behavior binding. Enable the existing OpenAPI
version check if supported and retain any observed version separately from the
declared identity. Run the following with a new, private output directory.
To test a new authorized deployment, update its reviewed binding and model ID.
The probe takes its endpoint and settings from the supplied TOML. An API key
may still come from the credential environment variable named in that config.

```bash
.venv/bin/python examples/exact_measurements.py \
  --config /path/to/approved-mlx-vlm.toml \
  --model lmstudio-community/Magistral-Small-2509-MLX-8bit \
  --repeats 2 \
  --output-dir /private/path/to/new-diagnostic-directory
```

The probe alternates arm order across repeats. It keeps the citations,
instruction, model, profile, schema and transport parameters fixed and varies
only the exact presentation. Private artifacts retain raw HTTP response bytes
when available, parsed output or error, tokens, latency, finish reason,
fingerprints and versions. The test covers the two negative temperatures,
equivalent and different speeds, different temperatures, and equal values
describing distinct business facts. Evaluate numerical consistency and the
business response separately. Review unknown usage or finish information as
unknown. Repeat offline and live checks after any calculator, registry,
presentation, profile, model or server change. Compare results to a separately
maintained holdout and record it independently of these diagnostic cases.
Successful synthetic runs are not general validation of a model.

## Pin the revision from UniversRobot

Once the reviewed commit is available in the
[CortexMux repository on GitHub](https://github.com/tmcuh1982/CortexMux),
replace the CortexMux wheel entry in UniversRobot's `requirements/llm.txt`
with a direct Git dependency on the full commit SHA.

```text
cortexmux @ git+https://github.com/tmcuh1982/CortexMux.git@REVIEWED_FULL_COMMIT_SHA
```

Install in a clean test environment before adoption. Check
`vcs_info.commit_id` in the installed package's `direct_url.json` against the
reviewed SHA. Record the calculator, registry and presentation versions with
the qualification results.

Map two validated source operands to `MeasurementComparison` and pass the
opted-in context through the consumer's `TracedCortexMux` structured call.
Record source IDs and verified quotations alongside the proof and prompt hash.
In UniversRobot, reject a pair with Python status `different` before handling
missing business evidence. Use `equal` as numerical evidence and validate the
exact product reference, bound and conditions independently. Send approximate
measurements or an unverified comparison to review when the available evidence
does not establish a rejection.
