# Providers

Providers implement health, model listing, capabilities, support checks,
execution, and cleanup. The registry rejects duplicate names unless replacement
is explicit. Capabilities are introspected where reliable or configured; names
are never treated as proof that a model supports vision or structured output.

Ollama uses `/api/version`, `/api/tags`, `/api/generate`, `/api/chat`, and
`/api/embed`. ComfyUI uses `/system_stats`, `/models/{folder}`, `/prompt`,
`/history/{prompt_id}`, `/view`, and optional `/ws`.

Custom providers subclass `BaseProvider` and can be registered through
`mux.register_provider(provider)`.

