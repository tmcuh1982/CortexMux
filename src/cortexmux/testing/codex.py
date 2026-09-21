"""Private, offline Codex cases, deterministic validation, and explicit replay."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from platformdirs import user_data_dir
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ProviderResponseError,
    SecurityError,
    StructuredOutputValidationError,
)
from cortexmux.core.json_schema import validate_json_schema, validate_json_schema_definition
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.codex.schemas import CodexRequestOptions
from cortexmux.schemas.common import HealthStatus, ModelInfo
from cortexmux.schemas.requests import (
    ChatRequest,
    CortexRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
)
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    StreamChunk,
    StreamEvent,
    StructuredResponse,
    StructuredStreamChunk,
    StructuredStreamCompleted,
    TextResponse,
)

LabRequest = Annotated[
    TextGenerationRequest | ChatRequest | StructuredOutputRequest, Field(discriminator="task")
]
_REQUEST: TypeAdapter[TextGenerationRequest | ChatRequest | StructuredOutputRequest] = TypeAdapter(
    LabRequest
)
_MAX_BYTES = 4_000_000
_SUFFIX = ".codex-case.json"
_MARKER = ".cortexmux-codex-lab"


class CodexTestCase(BaseModel):
    """One local response fixture; imported data is never an instruction to execute."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    format_version: Literal[1] = 1
    case_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[a-f0-9]{32}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    origin: Literal["synthetic", "recorded"] = "synthetic"
    request: LabRequest = Field(repr=False)
    content: str = Field(repr=False, max_length=1_000_000)
    status: Literal["completed", "interrupted", "failed"] = "completed"
    expect_valid: bool = True
    expected_content: str | None = Field(default=None, repr=False, max_length=1_000_000)

    @model_validator(mode="after")
    def validate_request(self) -> CodexTestCase:
        """Accept bounded text requests and remove unrelated metadata from storage."""
        if not self.request.model or self.request.provider not in {None, "codex", "codex-replay"}:
            raise ValueError("A Codex model and text request are required.")
        CodexRequestOptions.model_validate(self.request.options)
        if isinstance(self.request, ChatRequest) and any(
            message.images or message.role.value == "tool" for message in self.request.messages
        ):
            raise ValueError("Replay accepts text messages without tools or images.")
        if isinstance(self.request, StructuredOutputRequest):
            if self.request.json_schema is None:
                raise ValueError("A JSON Schema is required for structured replay.")
            validate_json_schema_definition(self.request.json_schema)
        self.request = self.request.model_copy(update={"metadata": {}})
        return self


class CodexValidationReport(BaseModel):
    """Content-free result; shape validation does not establish factual correctness."""

    case_id: str
    response_valid: bool
    expectation_met: bool
    reason: str | None = None
    expected_content_matches: bool | None = None
    simulation: Literal[True] = True


def parse_lab_request(data: str) -> TextGenerationRequest | ChatRequest | StructuredOutputRequest:
    """Parse a normalized request without echoing sensitive validation inputs."""
    try:
        return _REQUEST.validate_json(data)
    except ValueError:
        raise InvalidRequestError("Invalid local Codex request file.") from None


def read_lab_input(path: Path) -> str:
    """Read a bounded regular UTF-8 file without following a file symlink."""
    try:
        if path.is_symlink():
            raise ValueError("Symlink")
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Not a regular file")
            data = handle.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ValueError("File limit")
        return data.decode("utf-8")
    except (OSError, ValueError):
        raise InvalidRequestError(
            "Local Codex file is unreadable, non-regular or too large."
        ) from None


def validate_codex_case(case: CodexTestCase) -> CodexValidationReport:
    """Validate completion, JSON/schema, and an optional exact expected output locally."""
    reason: str | None = None
    if case.status != "completed":
        reason = case.status
    elif not case.content.strip():
        reason = "empty_response"
    elif isinstance(case.request, StructuredOutputRequest):
        try:
            parsed = json.loads(case.content)
            assert case.request.json_schema is not None
            validate_json_schema(parsed, case.request.json_schema)
        except (ValueError, RecursionError):
            reason = "structured_output_invalid"
    valid = reason is None
    matches = case.content == case.expected_content if case.expected_content is not None else None
    return CodexValidationReport(
        case_id=case.case_id,
        response_valid=valid,
        expectation_met=valid == case.expect_valid and matches is not False,
        reason=reason,
        expected_content_matches=matches,
    )


class CodexCaseStore:
    """Private case files outside every Git working tree; no transport or upload code."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = (
            Path(root) if root is not None else Path(user_data_dir("cortexmux")) / "codex-lab"
        )
        self.root = self.root.expanduser().absolute()
        self._check_location()

    def _check_location(self) -> None:
        for root in (self.root, self.root.resolve()):
            if root == Path.home() or root == Path(root.anchor):
                raise SecurityError("Use a dedicated local Codex test directory.")
            if any((parent / ".git").exists() for parent in (root, *root.parents)):
                raise SecurityError("Codex test data must be stored outside Git repositories.")
        if self.root.exists():
            if not self.root.is_dir() or self.root.is_symlink():
                raise SecurityError("Codex test storage must be a regular directory.")
            if not (self.root / _MARKER).is_file() and any(self.root.iterdir()):
                raise SecurityError("The directory is not an initialized Codex test store.")

    def initialize(self) -> Path:
        """Create a private owned directory and a defensive ignore-all Git rule."""
        self._check_location()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        for name, content in (
            (_MARKER, "CortexMux private Codex cases, version 1\n"),
            (".gitignore", "*\n"),
        ):
            if not (self.root / name).exists():
                self._write_new(self.root / name, content.encode())
        return self.root

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
        except OSError:
            raise InvalidRequestError(
                "Cannot create local Codex file; existing files are never overwritten."
            ) from None

    def save(self, case: CodexTestCase) -> Path:
        """Persist only the typed case; do not capture auth, raw logs or response metadata."""
        try:
            checked = CodexTestCase.model_validate(case.model_dump())
            data = checked.model_dump_json(indent=2).encode()
        except (ValueError, RecursionError):
            raise InvalidRequestError("Invalid local Codex case.") from None
        if len(data) > _MAX_BYTES:
            raise InvalidRequestError("Local Codex case exceeds the storage limit.")
        self.initialize()
        path = self.root / f"{checked.case_id}{_SUFFIX}"
        self._write_new(path, data)
        return path

    def load(self, case_id: str) -> CodexTestCase:
        """Load an identified case with bounded I/O and no raw data in errors."""
        self._check_location()
        if len(case_id) != 32 or any(character not in "0123456789abcdef" for character in case_id):
            raise InvalidRequestError("Invalid local Codex case identifier.")
        try:
            case = CodexTestCase.model_validate_json(
                read_lab_input(self.root / f"{case_id}{_SUFFIX}")
            )
        except (ValueError, RecursionError):
            raise InvalidRequestError("Invalid local Codex case file.") from None
        if case.case_id != case_id:
            raise InvalidRequestError("Local Codex case identifier mismatch.")
        return case

    def list_cases(self) -> list[str]:
        """List bounded opaque identifiers only, never prompts or outputs."""
        self._check_location()
        identifiers: list[str] = []
        for path in self.root.glob(f"*{_SUFFIX}"):
            identifiers.append(path.name.removesuffix(_SUFFIX))
            if len(identifiers) > 1000:
                raise InvalidRequestError("Local Codex store exceeds the 1000-case listing limit.")
        return sorted(identifiers)

    def record(
        self,
        request: TextGenerationRequest | ChatRequest | StructuredOutputRequest,
        response: TextResponse | ChatResponse | StructuredResponse,
    ) -> Path:
        """Explicitly save an already obtained Codex response without making a live call."""
        if (
            response.provider != "codex"
            or request.request_id != response.request_id
            or request.model != response.model
            or request.task != response.task
        ):
            raise InvalidRequestError("The Codex response does not match the supplied request.")
        try:
            case = CodexTestCase(request=request, content=response.content, origin="recorded")
        except ValueError:
            raise InvalidRequestError("Invalid Codex request for local recording.") from None
        return self.save(case)

    def validate(self, case_id: str) -> CodexValidationReport:
        """Validate a saved case and persist a report containing no request/output text."""
        report = validate_codex_case(self.load(case_id))
        self.initialize()
        path = self.root / f"{uuid4().hex}.codex-validation.json"
        self._write_new(path, report.model_dump_json(indent=2).encode())
        return report


class CodexReplayProvider(BaseProvider):
    """Replay exactly one local case; never falls back to Codex or other providers."""

    name = "codex-replay"

    def __init__(self, case: CodexTestCase) -> None:
        self._case = CodexTestCase.model_validate(case.model_dump())

    async def healthcheck(self) -> HealthStatus:
        """Report availability of the loaded offline fixture, not live account health."""
        return HealthStatus(
            provider=self.name, available=True, message="Offline replay fixture loaded."
        )

    async def list_models(self) -> list[ModelInfo]:
        """Expose only the recorded model, clearly marked as simulated."""
        assert self._case.request.model is not None
        return [
            ModelInfo(
                name=self._case.request.model, provider=self.name, metadata={"simulation": True}
            )
        ]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe local deterministic replay; no account, tools or model inference."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=frozenset({self._case.request.task}),
                streaming=True,
                structured_output=isinstance(self._case.request, StructuredOutputRequest),
                metadata={"simulation": True, "strict_no_tools": True},
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Match only the fixture's task and model."""
        return task == self._case.request.task and model in {None, self._case.request.model}

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Require matching inputs, validate the saved output, then return a marked response."""
        excluded = {"provider", "request_id", "metadata", "timeout", "model_profile", "stream"}
        if request.model_dump(exclude=excluded) != self._case.request.model_dump(exclude=excluded):
            raise InvalidRequestError("The replay request does not match the saved Codex case.")
        report = validate_codex_case(self._case)
        if not report.response_valid:
            error = (
                StructuredOutputValidationError
                if report.reason == "structured_output_invalid"
                else ProviderResponseError
            )
            raise error(
                "The saved Codex response is invalid.", provider=self.name, reason=report.reason
            )
        fields: dict[str, Any] = {
            "provider": self.name,
            "model": request.model,
            "request_id": request.request_id,
            "content": self._case.content,
            "raw_metadata": {
                "simulation": True,
                "case_id": self._case.case_id,
                "origin": self._case.origin,
            },
        }
        if isinstance(request, StructuredOutputRequest):
            return StructuredResponse(**fields, parsed=json.loads(self._case.content))
        if isinstance(request, ChatRequest):
            return ChatResponse(**fields)
        return TextResponse(**fields)

    async def stream(self, request: CortexRequest) -> AsyncIterator[StreamEvent]:
        """Emit deterministic chunks; replay does not reproduce live timing or protocol events."""
        response = await self.execute(request)
        assert isinstance(response, (TextResponse, ChatResponse, StructuredResponse))
        for offset in range(0, len(response.content), 256):
            content = response.content[offset : offset + 256]
            if isinstance(response, StructuredResponse):
                yield StructuredStreamChunk(
                    request_id=request.request_id,
                    provider=self.name,
                    model=request.model,
                    content=content,
                )
            else:
                yield StreamChunk(
                    request_id=request.request_id,
                    provider=self.name,
                    model=request.model,
                    content=content,
                    raw_metadata=response.raw_metadata,
                )
        if isinstance(response, StructuredResponse):
            yield StructuredStreamCompleted(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                content=response.content,
                parsed=response.parsed,
                raw_metadata=response.raw_metadata,
            )
        else:
            yield StreamChunk(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                content=response.content,
                done=True,
                raw_metadata=response.raw_metadata,
            )
