"""Response YAML exports preserve normalized data without unsafe YAML tags."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from cortexmux.core.exceptions import OptionalDependencyError
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    DataAnalysisResponse,
    DecisionResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    StructuredResponse,
    TextResponse,
    VisionResponse,
)


class ParsedRecord(BaseModel):
    """Representative validated output with JSON-normalized field types."""

    amount: Decimal
    created_at: datetime
    path: Path
    secret: SecretStr
    labels: list[str]
    optional: str | None = None


def test_structured_response_yaml_round_trip() -> None:
    yaml = pytest.importorskip("yaml")
    response = StructuredResponse(
        provider="fake",
        request_id="r",
        content='{"validated": true}',
        parsed=ParsedRecord(
            amount=Decimal("123.450"),
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            path=Path("report.csv"),
            secret=SecretStr("PRIVATE_SECRET"),
            labels=["été", "true", "null", "2026-01-02", "[red]literal[/red]"],
        ),
    )
    before = response.model_dump(mode="json")
    output = response.to_yaml()
    assert yaml.safe_load(output) == before
    assert response.model_dump(mode="json") == before
    assert "été" in output
    assert "PRIVATE_SECRET" not in output
    assert "!!python" not in output
    assert output.startswith("task:")


@pytest.mark.parametrize(
    "response_type, fields",
    [
        (CortexResponse, {"task": "chat"}),
        (TextResponse, {"content": "Bonjour"}),
        (ChatResponse, {"content": "Bonjour"}),
        (VisionResponse, {"content": "Une image"}),
        (EmbeddingResponse, {"embeddings": [[0.1, 0.2]]}),
        (ImageGenerationResponse, {"images": [], "prompt_id": "p"}),
        (DataAnalysisResponse, {"engine": "pandas", "profile": {}, "report_markdown": ""}),
        (
            DecisionResponse,
            {"answers": {"choice": {"type": "noul", "noul": 0.75}}},
        ),
    ],
)
def test_response_types_export_yaml(
    response_type: type[CortexResponse], fields: dict[str, object]
) -> None:
    yaml = pytest.importorskip("yaml")
    response = response_type.model_validate({"provider": "fake", "request_id": "r", **fields})
    assert yaml.safe_load(response.to_yaml()) == response.model_dump(mode="json")


def test_yaml_dependency_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    response = TextResponse(provider="fake", request_id="r", content="Hello")
    assert response.model_dump_json()
    with pytest.raises(OptionalDependencyError, match=r"cortexmux\[yaml\]"):
        response.to_yaml()
