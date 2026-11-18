"""Bounded generation and independent validation for application-defined tasks."""

from collections.abc import Awaitable, Callable
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class HarnessDecision(BaseModel):
    """Validator verdict. Retry requests a correction; review stops immediately."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    action: Literal["accept", "retry", "review"]
    reason: str = Field(min_length=1)


class HarnessAttempt(BaseModel, Generic[OutputT]):
    """One untrusted engine output paired with its validator's decision."""

    candidate: OutputT
    decision: HarnessDecision


class HarnessResult(BaseModel, Generic[OutputT]):
    """Ordered attempts; only a validated candidate is exposed as accepted output."""

    attempts: tuple[HarnessAttempt[OutputT], ...] = Field(min_length=1)

    @property
    def status(self) -> Literal["accepted", "review"]:
        """Report review when the last attempt failed or exhausted its retry budget."""
        return "accepted" if self.attempts[-1].decision.action == "accept" else "review"

    @property
    def output(self) -> OutputT | None:
        """Return the final candidate only if the validator accepted it."""
        return self.attempts[-1].candidate if self.status == "accepted" else None


async def run_harness(
    source: InputT,
    generate: Callable[[InputT, HarnessAttempt[OutputT] | None], Awaitable[OutputT]],
    validate: Callable[[InputT, OutputT], HarnessDecision | Awaitable[HarnessDecision]],
    *,
    max_passes: Literal[1, 2] = 2,
) -> HarnessResult[OutputT]:
    """Check each fresh output against the original source, with one optional retry."""
    if type(max_passes) is not int or max_passes not in (1, 2):
        raise ValueError("max_passes must be 1 or 2")
    attempts: list[HarnessAttempt[OutputT]] = []
    for _ in range(max_passes):
        candidate = await generate(source, attempts[-1] if attempts else None)
        verdict = validate(source, candidate)
        decision = verdict if isinstance(verdict, HarnessDecision) else await verdict
        attempts.append(HarnessAttempt(candidate=candidate, decision=decision))
        if decision.action != "retry":
            break
    return HarnessResult(attempts=tuple(attempts))
