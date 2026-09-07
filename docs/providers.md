# Providers

Providers implement health, model listing, capabilities, support checks,
execution, and cleanup. The registry rejects duplicate names unless replacement
is explicit. Capabilities are introspected where reliable or configured; names
are never treated as proof that a model supports vision or structured output.

Ollama uses `/api/version`, `/api/tags`, `/api/generate`, `/api/chat`, and
`/api/embed`. ComfyUI uses `/system_stats`, `/models/{folder}`, `/prompt`,
`/history/{prompt_id}`, `/view`, and optional `/ws`. Its provider exposes
reusable workflow catalogs and normalized progress callbacks without adding
ComfyUI-specific behavior to the router.

Structured Ollama generation supports both validated non-streaming responses
and typed streaming. `astream_structured()` and `stream_structured()` send the
JSON Schema as `format`, set `stream` to `true`, and send `think` at the payload
root when specified. Each `StructuredStreamChunk` is an unvalidated display-only
fragment. CortexMux concatenates all fragments after Ollama's completion marker,
parses the complete JSON, applies the same schema validation used by the
non-streaming API, and only then emits `StructuredStreamCompleted`. Premature
stream termination and final validation failure raise typed CortexMux errors;
no partial fragment is represented as validated application data.

Custom providers subclass `BaseProvider` and can be registered through
`mux.register_provider(provider)`.

## OpenAI Responses API

OpenAI is opt-in and uses `POST /responses` with `store: false`. For text and
structured requests, `system` is sent as the top-level `instructions` field;
it is never concatenated with `input`.

The adapter accepts only these typed request options and rejects every unknown
key before network I/O:

- `max_output_tokens`: positive integer;
- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or
  `max`, further restricted by model capabilities;
- `service_tier`: `auto`, `default`, `flex`, `scale`, `priority`, `fast`, or
  `ultrafast`; `default` explicitly selects Standard processing;
- `verbosity`: `low`, `medium`, or `high`.

`gpt-6-astra` and identifiers beginning with `gpt-6-astra-` accept only `low`,
`medium`, `high`, `xhigh`, and `max`, and default to `low`. `none` and `minimal`
are rejected before the request is sent. Other models use a separate capability
record so their accepted values can evolve independently.

`request.timeout`, when set, overrides `providers.openai.timeout_seconds` for
each HTTP attempt. Otherwise the provider timeout applies. OpenAI calls retry
HTTP 408, 409, 429, 5xx, timeouts, and transport errors at most
`max_retries + 1` times. Backoff is exponential, bounded by
`retry_max_delay_seconds`, and uses full jitter. Validation, authentication,
404, and other non-transient response errors are not retried. Current Responses
calls are side-effect-free (`store: false`, no tools or function calls), so the
adapter does not retry any manifestly non-replayable operation.

Successful responses retain token usage and add JSON-serializable
`raw_metadata`: effective provider/model, requested and returned service tier
when available, effective reasoning effort, attempt count, provider
`x-request-id` when available, and total provider-call duration. Router metadata
continues to report the complete routing decision and end-to-end route duration.

## Locally validated JSON Schema subset

CortexMux validates structured results deterministically and then preserves the
facade's final Pydantic `response_model.model_validate()` check. The local JSON
Schema validator supports:

- types `object`, `array`, `string`, `integer`, `number`, `boolean`, and `null`,
  including arrays of types;
- `enum`, `const`, `allOf`, `anyOf`, `oneOf`, and `not`;
- `properties`, `required`, boolean or schema-valued `additionalProperties`,
  `minProperties`, and `maxProperties`;
- `items`, `minItems`, `maxItems`, and `uniqueItems`;
- `minLength`, `maxLength`, and Python regular-expression `pattern` matching;
- `minimum`, `maximum`, `exclusiveMinimum`, and `exclusiveMaximum`;
- acyclic local references of the form `#/$defs/name` and `$defs`;
- annotation-only `$id`, `$schema`, `title`, `description`, `default`,
  `examples`, `deprecated`, `readOnly`, and `writeOnly`.

No other keyword is claimed or silently ignored. Unsupported keywords such as
`format`, `multipleOf`, `contains`, conditional schemas, tuple/prefix items,
unevaluated properties/items, pattern properties, and remote references are
rejected explicitly. JSON booleans never satisfy `integer` or `number`.
