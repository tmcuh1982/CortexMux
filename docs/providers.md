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
