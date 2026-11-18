import pytest

from cortexmux import CortexMux
from cortexmux.harness import HarnessDecision


@pytest.mark.parametrize(
    ("actions", "budget", "status", "count"),
    [
        (["accept"], 2, "accepted", 1),
        (["retry", "accept"], 2, "accepted", 2),
        (["review"], 2, "review", 1),
        (["retry", "retry"], 2, "review", 2),
        (["retry"], 1, "review", 1),
    ],
)
def test_harness_validates_each_engine_result(actions, budget, status, count):
    source = bytes([3, 7])
    calls = []

    async def engine(data, previous):
        assert data is source
        if previous is not None:
            assert previous.candidate == {"size": 1}
            assert previous.decision.reason == "check length"
        calls.append(previous)
        return {"size": len(calls)}

    def validate(data, candidate):
        assert data is source
        return HarnessDecision(action=actions[candidate["size"] - 1], reason="check length")

    with CortexMux() as mux:
        result = mux.harness(source, engine, validate, max_passes=budget)
    assert len(result.attempts) == len(calls) == count
    assert calls[0] is None
    assert result.status == status
    assert result.output == ({"size": count} if status == "accepted" else None)


@pytest.mark.asyncio
async def test_async_reviewer_can_stop_on_insufficient_evidence():
    async def engine(source, previous):
        return source.upper()

    async def review(source, candidate):
        assert candidate == source.upper()
        return HarnessDecision(action="review", reason="Missing source evidence")

    async with CortexMux() as mux:
        result = await mux.aharness("abc", engine, review)
    assert result.output is None and len(result.attempts) == 1


def test_reusable_field_checks():
    from cortexmux.measurements import ScalarMeasurement
    from cortexmux.validators import exact_copy, verified_quantity

    assert exact_copy("CB7", "CB7").action == "accept"
    assert exact_copy("CB7", "cb7").action == "retry"
    assert exact_copy("", "").action == "review"
    source = ScalarMeasurement(quantity_kind="linear_speed", value="2", unit="m/s")
    for value, unit, action in [
        ("7.2", "km/h", "accept"),
        ("7.3", "km/h", "retry"),
        ("2", "unknown", "review"),
    ]:
        target = source.model_copy(update=dict(value=value, unit=unit))
        assert verified_quantity(source, target).action == action
