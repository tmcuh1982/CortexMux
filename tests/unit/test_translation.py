"""Offline harness regression checks through the sync and async facades."""

import json

import pytest
from pydantic import ValidationError

from cortexmux import CortexMux
from cortexmux.schemas.responses import StructuredResponse
from cortexmux.translation import TranslationEngine, TranslationRequest
from cortexmux.translation._quantities import gate


def request():
    a, b = (TranslationEngine(provider="mlx", model=m) for m in ("first", "second"))
    return TranslationRequest(
        text="重量 10 公斤",
        citation="重量 10 公斤。",
        source_language="zh",
        translators=(a, b),
        arbiter=b,
    )


def stub(mux, monkeypatch, answers):
    calls = []

    async def respond(req):
        calls.append(req)
        parsed = answers[len(calls) - 1]
        return StructuredResponse(
            provider=req.provider,
            model=req.model,
            request_id=req.request_id,
            content=json.dumps(parsed),
            parsed=parsed,
        )

    monkeypatch.setattr(mux, "arun", respond)
    return calls


def test_select_only_eligible_and_require_source_evidence(monkeypatch):
    with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(
            mux,
            monkeypatch,
            [
                {"translation": "Poids 11 kg"},
                {"translation": "Poids 10 kg"},
                {"decision": "B", "quote": "10 公斤", "reason": "Matches the stated weight."},
            ],
        )
        req = request()
        result = mux.translate(req)
        assert result.status == "accepted" and result.selected_engine == req.translators[1]
        assert result.translation == "Poids 10 kg" and result.source_quote == "10 公斤"
        assert calls[0].prompt == calls[1].prompt
        assert json.loads(calls[2].prompt)["candidates"] == {"B": "Poids 10 kg"}
        assert [c.model for c in calls] == ["first", "second", "second"]
        assert all(c.think for c in calls)


@pytest.mark.asyncio
async def test_unknown_quantity_abstains_before_arbitration(monkeypatch):
    async with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(mux, monkeypatch, [{"translation": "22 watts"}] * 2)
        req = request().model_copy(update={"text": "22 W", "citation": "22 W"})
        result = await mux.atranslate(req)
        assert result.status == "review" and result.translation is None and len(calls) == 2


def test_unsupported_vote_or_unanchored_quote_cannot_select(monkeypatch):
    with CortexMux(register_builtin_providers=False) as mux:
        stub(
            mux,
            monkeypatch,
            [{"translation": "10 kg"}] * 2
            + [{"decision": "A", "quote": "absent", "reason": "seems fine"}],
        )
        assert mux.translate(request().model_copy(update={"max_passes": 1})).status == "review"


def test_unanchored_input_rejected():
    with pytest.raises(ValidationError):
        TranslationRequest.model_validate({**request().model_dump(), "text": "absent"})


@pytest.mark.parametrize(
    ("source", "valid", "invalid"),
    [
        ("20\u201325 公斤負載", "charge de 20 à 25 kg", "charge de 25 kg"),
        ("PDF:603KB", "PDF : 603 Ko", "PDF : 603 kg"),
        ("臂長可達 1750 毫米", "portée allant jusqu'à 1 750 mm", "portée de 1750 mm"),
        ("數分鐘內", "en quelques minutes", "en trois minutes"),
        ("百家即时零售仓", "cent entrepôts", "des centaines d'entrepôts"),
    ],
)
def test_quantity_gate(source, valid, invalid):
    assert gate(source, source, {"a": valid})[0] == ["a"]
    assert not gate(source, source, {"b": invalid})[0]


def test_quantity_failure_revises_independently_then_rechecks(monkeypatch):
    with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(
            mux,
            monkeypatch,
            [
                {"translation": "11 kg"},
                {"translation": "12 kg"},
                {"translation": "13 kg"},
                {"translation": "10 kg"},
                {"decision": "B", "quote": "10 公斤", "reason": "Poids conforme."},
            ],
        )
        result = mux.translate(request())
        assert (result.status, result.passes_used, result.translation) == ("accepted", 2, "10 kg")
        assert result.first_review_reason == "No candidate preserves the quantities."
        assert json.loads(calls[2].prompt)["previous_translation"] == "11 kg"
        assert json.loads(calls[3].prompt)["previous_translation"] == "12 kg"
        assert json.loads(calls[4].prompt)["candidates"] == {"B": "10 kg"}


@pytest.mark.parametrize("quote, status", [("10 公斤", "accepted"), ("encore absent", "review")])
def test_only_arbiter_retries_missing_proof_and_stops(monkeypatch, quote, status):
    with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(
            mux,
            monkeypatch,
            [{"translation": "10 kg"}] * 2
            + [
                {"decision": "A", "quote": "absent", "reason": "OK"},
                {"decision": "A", "quote": quote, "reason": "OK"},
            ],
        )
        result = mux.translate(request())
        assert result.passes_used == 2 and result.status == status
        assert [r.model for r in calls] == ["first", "second", "second", "second"]
        assert json.loads(calls[-1].prompt)["format_feedback"] == result.first_review_reason
        if status == "review":
            assert result.translation is None and result.selected_engine is None


def test_schema_failure_can_be_repaired(monkeypatch):
    with CortexMux(register_builtin_providers=False) as mux:
        stub(
            mux,
            monkeypatch,
            [
                {"wrong": "bad"},
                {"translation": "10 kg"},
                {"translation": "Poids 10 kg"},
                {"translation": "10 kg"},
                {"decision": "A", "quote": "重量", "reason": "Fidèle."},
            ],
        )
        result = mux.translate(request())
        assert result.status == "accepted" and result.passes_used == 2


def test_reference_and_adjacent_measure_are_independent(monkeypatch):
    req = request().model_copy(
        update=dict(text="CB7 10 公斤", citation="CB7 10 公斤", references=["CB7"])
    )
    with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(
            mux,
            monkeypatch,
            [
                {"translation": "CB7A 10 kg"},
                {"translation": "CB7 11 kg"},
                {"translation": "CB7 10 kg"},
                {"translation": "CB7A 10 kg"},
                {"decision": "A", "quote": "CB7", "reason": "Exact."},
            ],
        )
        result = mux.translate(req)
    assert result.translation == "CB7 10 kg" and result.passes_used == 2
    assert list(json.loads(calls[-1].prompt)["candidates"]) == ["A"]


def test_retry_names_the_missing_source_literal(monkeypatch):
    req = request().model_copy(
        update=dict(text="エプソン", citation="エプソン", preserve=["エプソン"])
    )
    with CortexMux(register_builtin_providers=False) as mux:
        calls = stub(mux, monkeypatch, [{"translation": "Epson"}] * 4)
        result = mux.translate(req)
    assert result.status == "review"
    retry = json.loads(calls[2].prompt)
    assert "エプソン" in retry["issue_to_check"] and retry["previous_translation"] == "Epson"


def test_require_decisive_source_span(monkeypatch):
    req = request().model_copy(update=dict(decisive_span="重量"))
    with CortexMux(register_builtin_providers=False) as mux:
        stub(
            mux,
            monkeypatch,
            [{"translation": "10 kg"}] * 2
            + [{"decision": "A", "quote": "10 公斤", "reason": "OK"}] * 2,
        )
        result = mux.translate(req)
    assert result.status == "review" and result.translation is None
