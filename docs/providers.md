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

## Gemini API

The `gemini` provider uses Google AI Studio's native `generateContent` API for
text, chat and structured JSON. It is opt-in and requires explicit remote-host
approval. System instructions are separate from user content, and outputs are
normalized into existing CortexMux response models. See
[Gemini configuration, options and limitations](gemini.md).

## Grok API

The `grok` provider uses xAI's Responses API for text, chat and JSON, with
`store=false` and no tools. Language models and aliases come from the
authenticated xAI catalog. Local JSON validation, normalized token usage and
the existing routing metadata also apply. See
[Grok configuration, options and limitations](grok.md).

## DeepSeek API

The `deepseek` provider uses DeepSeek's Chat Completions API for non-streaming
text, chat, and locally validated JSON. DeepSeek manages context caching
automatically. CortexMux preserves message order and normalizes the provider's
cache-hit and cache-miss token counters. See
[DeepSeek configuration and cache behavior](deepseek.md).

## TypeSafe System One API

The `typesafe` provider evaluates application-supplied state against typed
Noul, Choice or Score questions. It returns the original probabilities and
usage without applying a threshold or making application decisions. See
[TypeSafe configuration and examples](typesafe.md).

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

### Asynchronous OpenAI WebSocket sessions

Install `cortexmux[openai]` and explicitly enable the OpenAI provider before
calling `mux.openai_async_session()`. Use it with `async with`, then await its
session methods. It uses the Responses WebSocket API with `store: false` and an
in-memory response chain. Keep the
connection open for the whole conversation; a closed connection cannot resume
its `previous_response_id` when responses are not stored.

Pass the `model` argument to select a model whose WebSocket session capabilities
are registered in the OpenAI provider. Currently `gpt-6-astra` supports all
three features below. Other models are rejected before connecting until their
capabilities are confirmed and registered.

`mux.openai_async_session()` accepts these keyword arguments:

| Argument | Meaning |
| --- | --- |
| `model` | OpenAI model ID; defaults to `gpt-6-astra`. |
| `instructions` | Optional instructions sent with each response. |
| `tools` | Optional list of typed `OpenAIFunctionTool` definitions. |
| `reasoning_effort` | Optional initial effort; otherwise uses the model default. |

Inside the context, call `await session.start(prompt)` once and read events
with `await session.next_event()`. A completed event contains
`event.response.content` and any `event.response.tool_calls`. The session also
provides `steer(instruction)`, `continue_with_tool_results(results, prompt=None)`,
and `follow_up(prompt, reasoning_effort=None)`. `pending_call_ids` and
`latest_response_id` expose the current response chain state. The context
closes the WebSocket when it exits.

The session supports:

- Function tools can be declared with `async_=True`. CortexMux returns typed
  `OpenAIToolCall` records; application code starts and manages each job, then
  supplies `OpenAIToolResult` with the original `call_id` through
  `continue_with_tool_results()`. The model may produce independent text while
  that job runs. CortexMux never executes model-requested functions.
- After a `response.created` event, `steer()` queues a user correction on that
  response. Continue reading events after `response.steer.accepted`: it signals
  that the update is queued. The following `response.created` is when the
  server applies it. A `response.steer.pending` event means a tool result is
  needed; supply it once with `continue_with_tool_results()`.
- `follow_up(..., reasoning_effort=...)` inserts a `configuration_update`
  before the next user message. The request-level `reasoning.effort` stays at
  its original value so the original prompt prefix remains cacheable.

```python
import asyncio
import json

from cortexmux import CortexMux
from cortexmux.providers.openai import (
    OpenAIFunctionTool,
    OpenAIReasoningEffort,
    OpenAIToolResult,
)


async def lookup(key: str) -> dict[str, int]:
    # Application-owned synthetic data; replace with a validated local lookup.
    await asyncio.sleep(0)
    return {"value": 7}


async def main() -> None:
    async with CortexMux.from_env() as mux:
        tool = OpenAIFunctionTool(
            name="lookup",
            description="Look up a value in application data.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
            async_=True,
        )
        async with mux.openai_async_session(model="gpt-6-astra", tools=[tool]) as session:
            await session.start("Look up the value and draft the independent steps.")
            jobs: dict[str, asyncio.Task[dict[str, int]]] = {}
            while True:
                event = await session.next_event()
                if event.tool_call is not None:
                    args = json.loads(event.tool_call.arguments)
                    if event.tool_call.name != "lookup" or set(args) != {"key"}:
                        raise ValueError("Unexpected tool call")
                    jobs[event.tool_call.call_id] = asyncio.create_task(lookup(args["key"]))
                if event.response is not None:
                    print(event.response.content)
                    if not event.response.tool_calls:
                        break
                    results = [
                        OpenAIToolResult(
                            call_id=call.call_id, output=json.dumps(await jobs[call.call_id])
                        )
                        for call in event.response.tool_calls
                    ]
                    await session.continue_with_tool_results(results)


asyncio.run(main())
```

The example starts an application task when `response.output_item.done` delivers
the call, allowing Astra to keep working before the tool result is ready. In a
running response, `await session.steer("Keep the scope small.")` can be called
after its `response.created` event. After a completed response,
`await session.follow_up("Analyze risks.", reasoning_effort=OpenAIReasoningEffort.HIGH)`
changes effective effort without changing the request-level prefix. Steering
does not undo earlier output or cancel an already started tool. See the
[OpenAI async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling),
[steering](https://developers.openai.com/api/docs/guides/steering), and
[reasoning update](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation)
guides for the underlying API behavior.

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

## Codex subscription provider

The opt-in `codex` provider supports ChatGPT-managed login, text/structured
responses and account limits through local App Server stdio. See [Codex setup and limitations](codex.md).
