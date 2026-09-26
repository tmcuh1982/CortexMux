"""Network-free TypeSafe decision contract tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    ConfigurationError,
    ProviderResponseError,
    RemoteHostNotAllowedError,
)
from cortexmux.providers.typesafe import TypeSafeClient, TypeSafeProvider
from cortexmux.schemas.decisions import ChoiceQuestion, NoulAnswer, NoulQuestion, ScoreQuestion
from cortexmux.schemas.requests import DecisionRequest
from cortexmux.schemas.responses import DecisionResponse


@pytest.mark.asyncio
async def test_decision_request_preserves_typed_probabilities_and_usage() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
        assert request.url.path == "/v1/systemone"
        assert request.headers.get("Content-Type") == "application/json"
        body = request.read().decode()
        assert '"state":{"url":"https://example.org/robot"}' in body
        assert '"type":"noul"' in body
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "relevance": {"type": "noul", "noul": 0.82},
                    "extractable": {"type": "noul", "noul": 0.76},
                },
                "usage": {"input_tokens": 125, "output_tokens": 20},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(
        TypeSafeClient(
            "https://api.typesafe.ai", api_key="synthetic-key", timeout=5, client=transport
        )
    )
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    mux.register_provider(provider)
    async with mux:
        response = await mux.adecide(
            {"url": "https://example.org/robot"},
            {
                "relevance": NoulQuestion(instructions="Is this relevant?"),
                "extractable": NoulQuestion(instructions="Does it contain extractable facts?"),
            },
        )
    assert isinstance(response, DecisionResponse)
    assert isinstance(response.answers["relevance"], NoulAnswer)
    assert response.answers["relevance"].noul == 0.82
    assert response.answers["extractable"].noul == 0.76
    assert response.model == "jev-1.13.0"
    assert response.usage is not None and response.usage.prompt_tokens == 125
    assert response.routing is not None and response.routing.selected_provider == "typesafe"
    assert paths == ["/v1/models", "/v1/systemone"]
    await transport.aclose()


@pytest.mark.asyncio
async def test_pinned_model_is_passed_to_api_when_discovery_lists_only_aliases() -> None:
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
        requested_models.append(request.read().decode())
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"relevant": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    mux.register_provider(
        TypeSafeProvider(
            TypeSafeClient(
                "https://api.typesafe.ai", api_key="synthetic-key", timeout=5, client=transport
            )
        )
    )
    async with mux:
        response = await mux.adecide(
            "A robotics page",
            {"relevant": NoulQuestion(instructions="Is it relevant?")},
            model="jev-1.13.0",
        )
    assert response.model == "jev-1.13.0"
    assert len(requested_models) == 1
    assert '"model":"jev-1.13.0"' in requested_models[0]
    await transport.aclose()


@pytest.mark.asyncio
async def test_reject_mismatched_answer_and_option_distribution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "kind": {
                        "type": "choice",
                        "choice": "robot",
                        "probabilities": {"robot": 0.8, "wrong_option": 0.2},
                        "confidence": 0.7,
                    }
                },
                "usage": {"input_tokens": 20, "output_tokens": 8},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(
        TypeSafeClient(
            "https://api.typesafe.ai", api_key="synthetic-key", timeout=5, client=transport
        )
    )
    request = DecisionRequest(
        provider="typesafe",
        model="jev-latest",
        state="An industrial arm",
        questions={
            "kind": ChoiceQuestion(
                instructions="What is described?", criteria={"robot": None, "software": None}
            )
        },
    )
    with pytest.raises(ProviderResponseError):
        await provider.execute(request)
    await transport.aclose()


@pytest.mark.asyncio
async def test_choice_and_score_keep_full_distributions() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "kind": {
                        "type": "choice",
                        "choice": "robot",
                        "probabilities": {"robot": 0.8, "software": 0.2},
                        "confidence": 0.6,
                    },
                    "detail": {
                        "type": "score",
                        "score": 1.2,
                        "legend": {"0": "none", "1": "some", "2": "many"},
                        "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2},
                        "confidence": 0.7,
                    },
                },
                "usage": {"input_tokens": 50, "output_tokens": 30},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TypeSafeProvider(
        TypeSafeClient(
            "https://api.typesafe.ai", api_key="synthetic-key", timeout=5, client=transport
        )
    )
    request = DecisionRequest(
        provider="typesafe",
        model="jev-latest",
        state="A robot product description",
        questions={
            "kind": ChoiceQuestion(
                instructions="What is it?", criteria={"robot": None, "software": None}
            ),
            "detail": ScoreQuestion(
                instructions="How much detail?", criteria=["none", "some", "many"]
            ),
        },
    )
    response = await provider.execute(request)
    assert isinstance(response, DecisionResponse)
    assert response.answers["kind"].probabilities == {"robot": 0.8, "software": 0.2}
    assert response.answers["detail"].probabilities == {"0": 0.0, "1": 0.8, "2": 0.2}
    await transport.aclose()


def test_decision_schema_bounds_and_remote_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(
            state={"binary": b"private"},
            questions={"is_robot": NoulQuestion(instructions="Is this a robot?")},
        )
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="How useful?", criteria=["low"])
    with pytest.raises(ValidationError):
        DecisionRequest(
            state="short",
            questions={
                "kind": ChoiceQuestion(
                    instructions="Which kind?",
                    criteria={"robot": "x" * 120_000, "software": None},
                )
            },
        )
    assert not CortexMuxConfig().providers.typesafe.enabled
    config = CortexMuxConfig.model_validate(
        {
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "typesafe": {"enabled": True, "api_key": "synthetic-key"},
            }
        }
    )
    monkeypatch.setattr(
        "cortexmux.core.security.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    with pytest.raises(RemoteHostNotAllowedError):
        CortexMux(config)
    config.core.approved_hosts.add("api.typesafe.ai")
    config.providers.typesafe.api_key = None
    with pytest.raises(ConfigurationError):
        CortexMux(config)


def test_sync_facade_reuses_model_discovery() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"name": "jev-latest"}]})
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"relevant": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    mux = CortexMux(CortexMuxConfig(), register_builtin_providers=False)
    mux.register_provider(
        TypeSafeProvider(
            TypeSafeClient(
                "https://api.typesafe.ai", api_key="synthetic-key", timeout=5, client=transport
            )
        )
    )
    with mux:
        for _ in range(2):
            response = mux.decide(
                "A robotics page", {"relevant": NoulQuestion(instructions="Is it relevant?")}
            )
            assert response.answers["relevant"].noul == 0.8
    assert paths == ["/v1/models", "/v1/systemone", "/v1/systemone"]
    asyncio.run(transport.aclose())
