# Validate an engine output

Use `mux.harness(source, generate, validate)` or `await mux.aharness(...)`.
Choose the input and output types for the task. The async generator can call
any configured CortexMux engine or an application engine. A synchronous Python check or asynchronous reviewer
receives the original source and the candidate on every pass.

For example, extract a total from a list of integer quantities and check it
against the records before accepting it:

```python
import json

from pydantic import BaseModel
from cortexmux import CortexMux
from cortexmux.harness import HarnessAttempt, HarnessDecision


class Total(BaseModel):
    total: int


with CortexMux() as mux:

    async def generate(rows: list[int], previous: HarnessAttempt[Total] | None) -> Total:
        payload = {"quantities": rows}
        if previous is not None:
            payload.update(previous=previous.candidate.total, issue=previous.decision.reason)
        response = await mux.astructured(
            json.dumps(payload),
            system="Return the sum of the supplied quantities.",
            response_model=Total,
        )
        return Total.model_validate(response.parsed)

    def validate(rows: list[int], candidate: Total) -> HarnessDecision:
        if candidate.total == sum(rows):
            return HarnessDecision(action="accept", reason="Matches the source sum.")
        return HarnessDecision(action="retry", reason="Recalculate from all original rows.")

    result = mux.harness([12, 7], generate, validate)
    if result.status == "accepted":
        print(result.output)
```

Return `review` when the evidence is insufficient to decide. It stops the loop
immediately. Return `retry` for a correctable failure. The default budget is two
passes in total; set `max_passes=1` for a single attempt. When the budget runs
out, the status is `review` and `output` is empty. Each entry in `attempts`
retains its candidate and verdict for local inspection. Exceptions propagate
to the caller. Use replayable operations when enabling a second pass.

The [translation adapter](translation.md) uses the same loop with its own
quantity checks, independent candidates and cited arbitration.

Pass `cortexmux.validators.exact_copy` directly as the validator for a literal
field (manufacturer, reference, case-sensitive ID). Pass `verified_quantity`
for two `ScalarMeasurement` values. It uses the exact measurement engine's
supported scalar temperature and speed units. Unsupported quantities go to
review. Applications select the rule and supply the authoritative source.
