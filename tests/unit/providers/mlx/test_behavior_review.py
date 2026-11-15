"""Regression evidence for the 1.2.0rc1 requirements review (offline only)."""

import gzip
import json

import httpx
import pytest

from cortexmux.behavior import BehaviorBinding, BehaviorProfile, bundled_profile, qualify_syntax
from cortexmux.core.behavior_runtime import BehaviorDiagnostic, validate_bindings
from cortexmux.core.exceptions import CortexMuxError, InvalidRequestError
from cortexmux.core.types import TaskType
from cortexmux.providers.deepseek.client import DeepSeekClient
from cortexmux.providers.deepseek.provider import DeepSeekProvider
from cortexmux.providers.mlx.client import MLXClient
from cortexmux.providers.mlx.provider import MLXProvider
from cortexmux.schemas.requests import StructuredOutputRequest


def profile_for(provider):
    data = bundled_profile("qwen").model_dump()
    if provider == "deepseek":
        data.update(provider="deepseek", profile_id="synthetic-deepseek-test")
        data["identity"].update(
            server="deepseek",
            server_version="synthetic-v1",
            model="deepseek-chat",
            quantization=None,
        )
        data["rules"].update(reasoning="deepseek", reasoning_budget=False)
    return BehaviorProfile.model_validate({**data, "qualifications": []})


def deployment(provider="mlx_vlm", **kwargs):
    profile = profile_for(provider)
    return BehaviorBinding(
        profile=profile, identity=profile.identity, qualification_policy="offline", **kwargs
    )


def answer(model, content='{"answer":"ok"}', finish="stop", **fields):
    return {
        "model": model,
        "choices": [
            {"finish_reason": finish, "message": {"role": "assistant", "content": content}}
        ],
        **fields,
    }


def create_provider(provider, client, binding):
    if provider == "deepseek":
        return DeepSeekProvider(
            DeepSeekClient("http://127.0.0.1:8080/v1", api_key="private-http-key", client=client),
            server_version=binding.identity.server_version,
            behavior_profiles={binding.identity.model: binding},
        )
    return MLXProvider(
        MLXClient("http://127.0.0.1:8080/v1", client=client),
        name="mlx_vlm",
        server_version=binding.identity.server_version,
        behavior_profiles={binding.identity.model: binding},
    )


def request(binding, **options):
    return StructuredOutputRequest(
        prompt="private business prompt",
        model=binding.identity.model,
        options={"max_tokens": 512, **options},
    )


def test_candidate_requires_explicit_offline_policy():
    profile = bundled_profile("magistral")
    default = BehaviorBinding(profile=profile, identity=profile.identity)
    with pytest.raises(InvalidRequestError):
        default.validate_target(profile.provider, profile.identity.model, "0.7.11")
    explicit = BehaviorBinding(
        profile=profile, identity=profile.identity, qualification_policy="offline"
    )
    assert explicit.validate_target(profile.provider, profile.identity.model, "0.7.11") == "offline"


@pytest.mark.parametrize(
    "checks,passed",
    [
        ((), True),
        (("identity",), True),
        (("identity", "structured_output", "max_tokens", "reasoning", "reasoning_budget"), False),
    ],
)
def test_incomplete_or_failed_live_attestations_cannot_be_used(checks, passed):
    profile = bundled_profile("qwen")
    data = profile.model_dump()
    data["qualifications"] = [
        {
            "identity": profile.identity.model_dump(),
            "kind": "live",
            "passed": passed,
            "checks": checks,
            "evidence": "synthetic-test",
            "rules_fingerprint": profile.rules_fingerprint,
        }
    ]
    p = BehaviorProfile.model_validate(data)
    b = BehaviorBinding(profile=p, identity=p.identity, qualification_policy="offline")
    with pytest.raises(InvalidRequestError):
        b.validate_target(p.provider, p.identity.model, "0.7.11")


def test_semantic_fixture_expectation_is_checked():
    data = bundled_profile("qwen").model_dump()
    data["qualifications"] = []
    data["fixtures"][0]["expected_json"] = '{"answer":"changed"}'
    assert not qualify_syntax(BehaviorProfile.model_validate(data)).passed


def test_conflicting_aliases_are_rejected():
    first = deployment(selector="/private/shared-snapshot")
    data = first.profile.model_dump()
    data["identity"]["model"] = "other/model"
    second_profile = BehaviorProfile.model_validate(data)
    second = BehaviorBinding(
        profile=second_profile,
        identity=second_profile.identity,
        selector="/private/shared-snapshot",
    )
    with pytest.raises(InvalidRequestError, match="ambiguous"):
        validate_bindings({first.identity.model: first, second.identity.model: second}, "mlx_vlm")


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
@pytest.mark.parametrize("think", [True, False])
async def test_controls_and_diagnostics_share_the_same_pipeline(provider, think):
    binding = deployment(provider)

    def handler(req):
        payload = json.loads(req.content)
        assert payload["model"] == binding.identity.model
        assert payload["max_tokens"] == 512
        assert "response_format" not in payload and "behavior_profile" not in payload
        assert "reasoning_effort" not in payload
        if provider == "deepseek":
            assert payload["thinking"] == {"type": "enabled" if think else "disabled"}
        else:
            assert payload["enable_thinking"] is think
        return httpx.Response(200, json=answer(binding.identity.model))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = create_provider(provider, client, binding)
        req = request(binding, capture_private=True)
        req.think = think
        result = await adapter.execute(req)
    diagnostic = BehaviorDiagnostic.model_validate(result.raw_metadata["behavior"])
    assert diagnostic.qualification_kind == "offline"
    assert diagnostic.usage is None
    assert diagnostic.requested_parameters["think"] is think
    assert result.private_audit["response_body"]
    assert "private-http-key" not in result.model_dump_json()
    assert "private business" not in result.model_dump_json()


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
@pytest.mark.parametrize(
    "content,finish,category",
    [
        ('{"answer":"ok"}', "length", "truncation"),
        ('{"answer":"private","answer":"other"}', "stop", "syntax"),
        ('{"value":1e999}', "stop", "syntax"),
        ('{"answer":"ok"}', None, "completion"),
    ],
)
async def test_provider_failures_have_common_private_diagnostics(
    provider, content, finish, category
):
    binding = deployment(provider)
    count = 0

    def handler(_):
        nonlocal count
        count += 1
        return httpx.Response(200, json=answer(binding.identity.model, content, finish))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = create_provider(provider, client, binding)
        with pytest.raises(CortexMuxError) as caught:
            await adapter.execute(request(binding, capture_private=True))
    assert count == 1
    assert caught.value.context["behavior"]["category"] == category
    assert caught.value.private_audit["response_body"]
    assert "private-http-key" not in str(caught.value.to_dict())
    assert '"answer":"private"' not in str(caught.value.to_dict())


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
async def test_gzip_response_is_decoded_once(provider):
    binding = deployment(provider)
    compressed = gzip.compress(json.dumps(answer(binding.identity.model)).encode())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=compressed, headers={"content-encoding": "gzip"})
        )
    ) as client:
        result = await create_provider(provider, client, binding).execute(request(binding))
    assert result.parsed == {"answer": "ok"}


async def test_openapi_is_bounded_before_completion():
    seen = []

    def handler(req):
        seen.append(req.method)
        return httpx.Response(200, content=b"x" * 2_000_001)

    binding = deployment()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = create_provider("mlx_vlm", client, binding)
        adapter._check_openapi_version = True
        with pytest.raises(CortexMuxError) as caught:
            await adapter.execute(request(binding, capture_private=True))
    assert seen == ["GET"]
    assert caught.value.context["behavior"]["category"] == "size"
    assert caught.value.context["behavior"]["transmitted_parameters"] is None


async def test_metadata_and_capabilities_do_not_infer_vision():
    binding = deployment()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("no IO"))
    ) as client:
        adapter = create_provider("mlx_vlm", client, binding)
        capabilities = await adapter.get_capabilities(binding.identity.model)
    assert capabilities[0].task_types == {TaskType.STRUCTURED_OUTPUT}
    assert not adapter.supports(TaskType.VISION)


async def test_transformed_invalid_json_remains_in_private_audit():
    profile = bundled_profile("magistral")
    b = BehaviorBinding(profile=profile, identity=profile.identity, qualification_policy="offline")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json=answer(b.identity.model, "[THINK]private[/THINK]```json\nnot-json\n```")
            )
        )
    ) as client:
        with pytest.raises(CortexMuxError) as caught:
            await create_provider("mlx_vlm", client, b).execute(request(b, capture_private=True))
    assert caught.value.private_audit["normalized_response"] == "not-json"
    assert caught.value.context["behavior"]["transformations"] == ["think_prefix", "json_fence"]


class InterruptedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"x" * 65536
        raise httpx.ReadTimeout("private server detail")


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
async def test_partial_body_survives_transport_failure(provider):
    b = deployment(provider)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=InterruptedStream()))
    ) as client:
        with pytest.raises(CortexMuxError) as caught:
            await create_provider(provider, client, b).execute(request(b, capture_private=True))
    assert caught.value.context["behavior"]["category"] == "transport"
    assert len(caught.value.private_audit["response_body"]) == 65536
    assert "private server detail" not in str(caught.value.to_dict())


async def test_sampling_override_cannot_enable_unsupported_control():
    b = deployment()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("no IO"))
    ) as client:
        with pytest.raises(CortexMuxError) as caught:
            await create_provider("mlx_vlm", client, b).execute(request(b, temperature=0.5))
    assert caught.value.context["behavior"]["transmitted_parameters"] is None


async def test_reported_reasoning_budget_violation_is_rejected():
    b = deployment()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json=answer(
                    b.identity.model,
                    usage={
                        "completion_tokens": 256,
                        "completion_tokens_details": {"reasoning_tokens": 129},
                    },
                ),
            )
        )
    ) as client:
        req = request(b, thinking_budget=128)
        req.think = True
        with pytest.raises(CortexMuxError, match="reasoning count") as caught:
            await create_provider("mlx_vlm", client, b).execute(req)
    assert caught.value.context["behavior"]["reasoning_tokens"] == 129


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
async def test_schema_errors_share_safe_diagnostics(provider):
    b = deployment(provider)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=answer(b.identity.model)))
    ) as client:
        req = request(b, capture_private=True)
        req.json_schema = {"type": "object", "required": ["missing"]}
        with pytest.raises(CortexMuxError) as caught:
            await create_provider(provider, client, b).execute(req)
    assert caught.value.context["behavior"]["category"] == "schema"
    assert caught.value.private_audit["normalized_response"] == '{"answer":"ok"}'


def test_same_revision_cannot_have_conflicting_rules():
    first = deployment()
    data = first.profile.model_dump()
    data["identity"]["model"] = "other/model"
    data["rules"]["json_fence"] = True
    second_profile = BehaviorProfile.model_validate(data)
    second = BehaviorBinding(profile=second_profile, identity=second_profile.identity)
    with pytest.raises(InvalidRequestError, match="conflicting rules"):
        validate_bindings({first.identity.model: first, second.identity.model: second}, "mlx_vlm")


async def test_deepseek_facade_two_tasks_use_one_binding():
    from cortexmux import CortexMux
    from cortexmux.core.config import CortexMuxConfig

    b = deployment("deepseek")
    payloads = []

    def handler(req):
        payloads.append(json.loads(req.content))
        return httpx.Response(200, json=answer(b.identity.model))

    config = CortexMuxConfig.model_validate(
        {
            "providers": {"ollama": {"enabled": False}, "comfyui": {"enabled": False}},
            "routing": {
                "active_profile": "tasks",
                "profiles": {
                    "tasks": {
                        "structured_output": {
                            "provider": "deepseek",
                            "model": b.identity.model,
                            "options": {"max_tokens": 512},
                        }
                    }
                },
            },
        }
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
        CortexMux(config) as mux,
    ):
        mux.register_provider(create_provider("deepseek", http, b))
        response = await mux.astructured(
            "synthetic task A",
            think=True,
            json_schema={"required": ["answer"]},
            capture_private=True,
        )
        with pytest.raises(CortexMuxError) as caught:
            await mux.astructured(
                "synthetic task B",
                provider="deepseek",
                model=b.identity.model,
                think=False,
                max_tokens=256,
                json_schema={"required": ["missing"]},
                capture_private=True,
            )
    assert response.private_audit["response_body"]
    assert caught.value.private_audit["response_body"]
    assert caught.value.context["behavior"]["category"] == "schema"
    assert [p["max_tokens"] for p in payloads] == [512, 256]
    assert [p["thinking"]["type"] for p in payloads] == ["enabled", "disabled"]


@pytest.mark.parametrize("provider", ["mlx_vlm", "deepseek"])
async def test_no_private_capture_without_explicit_request(provider):
    b = deployment(provider)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(503, content=b"private server refusal")
        )
    ) as client:
        with pytest.raises(CortexMuxError) as caught:
            await create_provider(provider, client, b).execute(request(b))
    assert caught.value.private_audit is None
    assert "private server refusal" not in str(caught.value.to_dict())


def test_packaged_fixtures_match_synthetic_source():
    from pathlib import Path

    source = Path(__file__).resolve().parents[4] / "examples/data/behavior-profile-fixtures.json"
    data = json.loads(source.read_text())
    assert data["is_synthetic"] is True
    for name in ("magistral", "qwen"):
        fixtures = [fixture.model_dump(mode="json") for fixture in bundled_profile(name).fixtures]
        assert data["fixtures"][name] == fixtures
        assert all(fixture["is_synthetic"] is True for fixture in fixtures)
