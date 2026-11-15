"""Offline behavioral contracts through the public facade and real HTTP adapter."""

import json

import httpx
import pytest
from pydantic import ValidationError

from cortexmux import CortexMux
from cortexmux.behavior import (
    BehaviorBinding,
    BehaviorProfile,
    Qualification,
    bundled_profile,
    qualify_syntax,
)
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import CortexMuxError, InvalidRequestError
from cortexmux.providers.mlx.client import MLXClient
from cortexmux.providers.mlx.provider import MLXProvider
from cortexmux.schemas.requests import StructuredOutputRequest


def binding(name="magistral", **updates):
    profile = bundled_profile(name)
    return BehaviorBinding(
        profile=profile, identity=profile.identity, qualification_policy="offline", **updates
    )


def reply(content='{"answer":"ok"}', finish="stop", **extra):
    return httpx.Response(
        200,
        json={
            "model": bundled_profile("magistral").identity.model,
            "choices": [
                {"finish_reason": finish, "message": {"role": "assistant", "content": content}}
            ],
            **extra,
        },
    )


def configured_mux(client, deployment=None):
    deployment = deployment or binding()
    model = deployment.identity.model
    config = CortexMuxConfig.model_validate(
        {
            "providers": {"ollama": {"enabled": False}, "comfyui": {"enabled": False}},
            "routing": {
                "active_profile": "tasks",
                "profiles": {
                    "tasks": {
                        "structured_output": {
                            "provider": "mlx_vlm",
                            "model": model,
                            "options": {"max_tokens": 512},
                        }
                    }
                },
            },
        }
    )
    mux = CortexMux(config)
    mux.register_provider(
        MLXProvider(
            MLXClient("http://127.0.0.1:8080", client=client),
            name="mlx_vlm",
            server_version="0.7.11",
            behavior_profiles={model: deployment},
        )
    )
    return mux


@pytest.mark.parametrize("name", ["magistral", "qwen"])
def test_portable_profiles_and_fixtures(name):
    profile = bundled_profile(name)
    restored = BehaviorProfile.import_json(profile.export_json())
    assert restored == profile
    assert restored.fingerprint == profile.fingerprint
    assert qualify_syntax(profile).passed
    assert all(q.kind == "offline" for q in profile.qualifications)
    assert "selector" not in profile.export_json()
    with pytest.raises(ValidationError):
        profile.revision = 2


def test_requalification_reuses_rules_without_implicit_inheritance():
    profile = bundled_profile("magistral")
    changed = profile.identity.model_copy(update={"server_version": "0.7.12"})
    deployment = BehaviorBinding(profile=profile, identity=changed)
    with pytest.raises(InvalidRequestError):
        deployment.validate_target("mlx_vlm", changed.model, "0.7.12")
    attestation = Qualification(
        identity=changed,
        rules_fingerprint=profile.rules_fingerprint,
        kind="live",
        checks=("identity", "structured_output", "max_tokens", "reasoning", "reasoning_budget"),
        passed=True,
        evidence="synthetic-test-attestation",
    )
    requalified = BehaviorProfile.model_validate(
        {**profile.model_dump(), "qualifications": [attestation]}
    )
    BehaviorBinding(profile=requalified, identity=changed).validate_target(
        "mlx_vlm", changed.model, "0.7.12"
    )
    assert requalified.rules_fingerprint == profile.rules_fingerprint
    assert requalified.fingerprint != profile.fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"server_version": "unknown"},
        {"server": "other"},
        {"model_revision": "new"},
        {"quantization": "4bit"},
    ],
)
def test_identity_mismatch(change):
    original = binding()
    deployment = original.model_copy(
        update={"identity": original.identity.model_copy(update=change)}
    )
    with pytest.raises(InvalidRequestError):
        deployment.validate_target("mlx_vlm", original.identity.model, "0.7.11")


async def test_public_explicit_and_routed_calls_keep_task_budgets():
    payloads = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return reply('[THINK]synthetic private reasoning[/THINK]\n```json\n{"answer":"ok"}\n```')

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        configured_mux(client) as mux,
    ):
        a = await mux.astructured(
            "task-a",
            provider="mlx_vlm",
            model=binding().identity.model,
            max_tokens=512,
            think=True,
            thinking_budget=128,
            capture_private=True,
            json_schema={"type": "object", "required": ["answer"]},
        )
        b = await mux.astructured(
            "task-b",
            max_tokens=256,
            json_schema={"type": "object", "properties": {"answer": {"const": "ok"}}},
        )
    assert a.parsed == b.parsed == {"answer": "ok"}
    assert [p["max_tokens"] for p in payloads] == [512, 256]
    assert payloads[0]["enable_thinking"] is True
    assert payloads[0]["thinking_budget"] == 128
    assert all("response_format" not in p and "behavior_profile" not in p for p in payloads)
    assert a.raw_metadata["behavior"]["fingerprint"] == b.raw_metadata["behavior"]["fingerprint"]
    assert a.raw_metadata["behavior"]["usage"] is None
    assert a.raw_metadata["behavior"]["transformations"] == ["think_prefix", "json_fence"]
    assert a.private_audit["normalized_response"] == '{"answer":"ok"}'
    assert "private reasoning" not in a.model_dump_json()
    assert b.private_audit is None


@pytest.mark.parametrize(
    "content,finish,category",
    [
        ('{"answer":"ok"}', "length", "truncation"),
        ('{"answer":"ok"}', None, "completion"),
        ('{"answer":"ok"}', "unknown", "completion"),
        ('{"answer":"ok","answer":"bad"}', "stop", "syntax"),
        ('{"answer":NaN}', "stop", "syntax"),
        ('{"answer":1e999}', "stop", "syntax"),
        ('{"answer":"wrong"}', "stop", "schema"),
        ('[{"answer":"ok"}]', "stop", "syntax"),
    ],
)
async def test_failure_diagnostics_private_capture_and_no_retry(content, finish, category):
    calls = []

    def handler(request):
        calls.append(request)
        return reply(content, finish)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        configured_mux(client) as mux,
    ):
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured(
                "private prompt",
                max_tokens=512,
                capture_private=True,
                json_schema={"properties": {"answer": {"const": "ok"}}},
            )
    assert len(calls) == 1
    assert caught.value.context["behavior"]["category"] == category
    assert caught.value.private_audit["response_body"]
    assert "private prompt" not in str(caught.value.to_dict())
    assert content not in str(caught.value)


@pytest.mark.parametrize(
    "options",
    [
        {"max_tokens": 0},
        {"thinking_budget": 128},
        {"reasoning_effort": "low"},
        {"required_capabilities": ["vision"]},
        {"behavior_profile": "wrong@1"},
    ],
)
async def test_preflight_never_sends_unsupported_options(options):
    def handler(request):
        pytest.fail("Preflight must not use network")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        configured_mux(client) as mux,
    ):
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured("task", **options)
    assert caught.value.context["behavior"]["category"] == "compatibility"
    assert caught.value.context["behavior"]["transmitted_parameters"] is None


async def test_alias_and_strict_normalization():
    deployment = binding(selector="/private/snapshot", reported_selectors=("local-alias",))

    def handler(request):
        assert json.loads(request.content)["model"] == "/private/snapshot"
        return reply('[THINK]private[/THINK]{"answer":"ok"}', model="local-alias")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        configured_mux(client, deployment) as mux,
    ):
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured("task", normalization="strict")
    assert caught.value.context["behavior"]["category"] == "syntax"
    assert "/private/snapshot" not in str(caught.value.to_dict())
    assert "local-alias" not in str(caught.value.to_dict())


async def test_bounded_response_body():
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 2_000_001))
        ) as client,
        configured_mux(client) as mux,
    ):
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured("task", capture_private=True)
    assert caught.value.context["behavior"]["category"] == "size"
    assert caught.value.private_audit["response_body_truncated"] is True
    assert len(caught.value.private_audit["response_body"]) == 2_000_000


def test_sync_public_parity():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: reply()))
    with configured_mux(client) as mux:
        result = mux.structured("task", capture_private=True)
        assert result.raw_metadata["behavior"]["category"] == "success"
        mux._sync(client.aclose())


async def test_unknown_server_version_refuses_bound_profile():
    deployment = binding()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("Unknown server must fail before I/O"))
    ) as client:
        provider = MLXProvider(
            MLXClient("http://127.0.0.1:8080", client=client),
            name="mlx_vlm",
            behavior_profiles={deployment.identity.model: deployment},
        )
        with pytest.raises(InvalidRequestError):
            await provider.execute(
                StructuredOutputRequest(
                    prompt="task", model=deployment.identity.model, options={"max_tokens": 512}
                )
            )


async def test_qwen_native_controls_without_response_format():
    deployment = binding("qwen")

    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == deployment.identity.model
        assert payload["max_tokens"] == 512
        assert payload["enable_thinking"] is True and payload["thinking_budget"] == 128
        assert "response_format" not in payload
        return reply(model=deployment.identity.model)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
        configured_mux(client, deployment) as mux,
    ):
        result = await mux.astructured("task", think=True, thinking_budget=128)
    assert result.parsed == {"answer": "ok"}


async def test_relations_and_literal_values_are_not_rewritten():
    value = {
        "relations": [
            {
                "left_id": "fact-a",
                "right_id": "fact-b",
                "relation": "same_fact",
                "reason": "Même preuve.",
            }
        ],
        "payload": "7",
        "product": "the range",
        "quote": "[THINK]literal[/THINK]",
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: reply(json.dumps(value)))
        ) as client,
        configured_mux(client) as mux,
    ):
        result = await mux.astructured("task")
    assert result.parsed == value


async def test_response_model_validator_failure_preserves_audit():
    from pydantic import BaseModel, field_validator

    class Contract(BaseModel):
        answer: str

        @field_validator("answer")
        @classmethod
        def reject(cls, value):
            raise ValueError("private validation details")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: reply())) as client,
        configured_mux(client) as mux,
    ):
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured("task", response_model=Contract, capture_private=True)
    assert caught.value.context["behavior"]["category"] == "schema"
    assert caught.value.private_audit["response_body"]
    assert "private validation" not in str(caught.value)
