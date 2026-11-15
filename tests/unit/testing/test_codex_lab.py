"""Offline replay, private storage, and content-free CLI regression tests."""

import json
import os
import socket
import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cortexmux import CortexMux
from cortexmux.cli.main import app
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ProviderResponseError,
    SecurityError,
    StructuredOutputValidationError,
)
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest, TextGenerationRequest
from cortexmux.schemas.responses import StructuredStreamCompleted, TextResponse
from cortexmux.testing import (
    CodexCaseStore,
    CodexReplayProvider,
    CodexTestCase,
    validate_codex_case,
)


@pytest.fixture
def case():
    return CodexTestCase(
        request=StructuredOutputRequest(
            provider="codex",
            model="simulated-model",
            prompt="PRIVATE input",
            json_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        ),
        content='{"count":42}',
    )


@pytest.fixture
def store(tmp_path):
    return CodexCaseStore(tmp_path / "private-lab")


def test_private_storage_roundtrip_and_report(store, case):
    path = store.save(case)
    assert store.load(case.case_id) == case
    assert store.list_cases() == [case.case_id]
    assert store.validate(case.case_id).expectation_met
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert (store.root / ".gitignore").read_text() == "*\n"
    report = next(store.root.glob("*.codex-validation.json")).read_text()
    assert "PRIVATE" not in report and "count" not in report
    assert case.content not in repr(case) and "PRIVATE" not in repr(case)
    with pytest.raises(InvalidRequestError):
        store.save(case)


@pytest.mark.parametrize("git_marker", ["directory", "worktree-file"])
def test_repository_storage_refused(tmp_path, git_marker):
    repo = tmp_path / "repo"
    repo.mkdir()
    if git_marker == "directory":
        (repo / ".git").mkdir()
    else:
        (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(SecurityError, match="outside Git"):
        CodexCaseStore(repo / "nested" / "lab")
    assert not (repo / "nested").exists()


def test_git_created_after_initialization_is_refused(store, case):
    store.save(case)
    (store.root / ".git").mkdir()
    with pytest.raises(SecurityError):
        store.load(case.case_id)
    with pytest.raises(SecurityError):
        store.validate(case.case_id)


def test_symlinks_traversal_and_existing_directory(store, case, tmp_path):
    store.save(case)
    with pytest.raises(InvalidRequestError):
        store.load("../../secret")
    private = tmp_path / "secret"
    private.write_text("SECRET")
    target = store.root / f"{'a' * 32}.codex-case.json"
    target.symlink_to(private)
    with pytest.raises(InvalidRequestError) as error:
        store.load("a" * 32)
    assert "SECRET" not in str(error.value)
    with pytest.raises(SecurityError):
        CodexCaseStore(tmp_path)


@pytest.mark.parametrize(
    "content,status,expected",
    [
        (" ", "completed", "empty_response"),
        ('{"count":"bad"}', "completed", "structured_output_invalid"),
        ("not-json PRIVATE", "completed", "structured_output_invalid"),
        ('{"count":42}', "interrupted", "interrupted"),
        ('{"count":42}', "failed", "failed"),
    ],
)
def test_validation_invalid_fixtures(case, content, status, expected):
    case = case.model_copy(update={"content": content, "status": status, "expect_valid": False})
    report = validate_codex_case(case)
    assert not report.response_valid
    assert report.expectation_met
    assert report.reason == expected
    assert "PRIVATE" not in report.model_dump_json()


def test_expected_output_distinct_from_schema(case):
    report = validate_codex_case(case.model_copy(update={"expected_content": '{"count":43}'}))
    assert report.response_valid
    assert not report.expectation_met
    assert report.expected_content_matches is False


async def test_offline_provider_facade_stream_and_input_match(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not start processes or access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    async with CortexMux(register_builtin_providers=False) as mux:
        mux.register_provider(CodexReplayProvider(case))
        req = case.request.model_copy(
            update={"provider": "codex-replay", "request_id": "replay-id"}
        )
        response = await mux.arun(req)
        assert response.parsed == {"count": 42}
        assert response.raw_metadata["simulation"] is True
        assert response.request_id == "replay-id"
        events = [event async for event in mux.astream(req)]
        assert isinstance(events[-1], StructuredStreamCompleted)
        assert events[-1].parsed == {"count": 42}
        with pytest.raises(InvalidRequestError):
            await mux.arun(req.model_copy(update={"prompt": "different PRIVATE"}))


@pytest.mark.parametrize(
    "status,error",
    [("interrupted", ProviderResponseError), ("completed", StructuredOutputValidationError)],
)
async def test_replay_invalid_never_completes(case, status, error):
    provider = CodexReplayProvider(
        case.model_copy(update={"status": status, "content": "bad PRIVATE"})
    )
    with pytest.raises(error) as caught:
        await anext(provider.stream(case.request))
    assert "PRIVATE" not in str(caught.value)


def test_record_strips_metadata_and_never_logs(store):
    request = TextGenerationRequest(
        provider="codex", model="test", prompt="PRIVATE", metadata={"secret": "SECRET"}
    )
    response = TextResponse(
        provider="codex",
        model="test",
        request_id=request.request_id,
        content="local answer",
        raw_metadata={"accessToken": "SECRET"},
    )
    path = store.record(request, response)
    data = path.read_text()
    assert "SECRET" not in data and "accessToken" not in data
    assert "PRIVATE" in data
    case = store.load(path.name.removesuffix(".codex-case.json"))
    assert case.origin == "recorded"
    with pytest.raises(InvalidRequestError):
        store.record(request, response.model_copy(update={"provider": "openai"}))


def test_malformed_and_oversized_case_errors_are_sanitized(store, case):
    path = store.save(case)
    path.write_text('{"content":"SECRET", "request":42}')
    with pytest.raises(InvalidRequestError) as error:
        store.load(case.case_id)
    assert "SECRET" not in str(error.value)
    path.write_text("x" * 4_000_001)
    with pytest.raises(InvalidRequestError):
        store.load(case.case_id)


def test_cli_import_validate_and_negative_case(tmp_path, case):
    runner = CliRunner()
    root = tmp_path / "lab"
    request_file = tmp_path / "request.json"
    response_file = tmp_path / "response.txt"
    request_file.write_text(case.request.model_dump_json())
    response_file.write_text('{"count":"PRIVATE"}')
    args = [
        "codex-lab",
        "add",
        "--root",
        str(root),
        "--request",
        str(request_file),
        "--response",
        str(response_file),
        "--expect-invalid",
    ]
    added = runner.invoke(app, args)
    assert added.exit_code == 0, added.output
    case_id = added.stdout.strip()
    checked = runner.invoke(app, ["codex-lab", "validate", case_id, "--root", str(root)])
    assert checked.exit_code == 0, checked.output
    assert "PRIVATE" not in checked.output
    assert json.loads(checked.stdout)["expectation_met"]
    listed = runner.invoke(app, ["codex-lab", "list", "--root", str(root)])
    assert listed.stdout.strip() == case_id
    args.remove("--expect-invalid")
    added = runner.invoke(app, args)
    checked = runner.invoke(
        app, ["codex-lab", "validate", added.stdout.strip(), "--root", str(root)]
    )
    assert checked.exit_code == 1
    assert "PRIVATE" not in checked.output


def test_cli_errors_hide_input(tmp_path):
    request = tmp_path / "request.json"
    request.write_text('{"prompt":"SECRET", "task":"nonsense"}')
    response = tmp_path / "response.txt"
    response.write_text("SECRET")
    result = CliRunner().invoke(
        app,
        [
            "codex-lab",
            "add",
            "--root",
            str(tmp_path / "lab"),
            "--request",
            str(request),
            "--response",
            str(response),
        ],
    )
    assert result.exit_code == 2
    assert "SECRET" not in result.output


def test_repository_ignores_private_case_files():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="private.codex-case.json\nprivate.codex-validation.json\n.cortexmux-private/input.txt\n",
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(result.stdout.splitlines()) == 3


async def test_text_and_chat_replay():
    for request in [
        TextGenerationRequest(prompt="test", model="test"),
        ChatRequest(messages=[{"role": "user", "content": "test"}], model="test"),
    ]:
        case = CodexTestCase(request=request, content="answer")
        response = await CodexReplayProvider(case).execute(request)
        assert response.task == request.task and response.content == "answer"
