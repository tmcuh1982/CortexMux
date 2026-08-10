"""CLI behavior tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cortexmux import CortexMux
from cortexmux.cli.main import app
from cortexmux.core.types import TaskType
from cortexmux.schemas.common import ImageArtifact, ModelInfo
from cortexmux.schemas.responses import (
    ChatResponse,
    DataAnalysisResponse,
    ImageGenerationResponse,
    TextResponse,
)
from cortexmux.selection import MachineProfile, QualificationManifest, QualificationSuite

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.5.0" in result.stdout


def test_doctor_without_network_providers(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[providers.ollama]\nenabled=false\n[providers.comfyui]\nenabled=false\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["doctor", "--json"],
        env={"CORTEXMUX_CONFIG": str(config)},
    )
    assert result.exit_code == 0
    assert '"provider": "data"' in result.stdout


def test_config_show_json(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[providers.ollama]\nheaders={Authorization="secret"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["config", "show", "--json"],
        env={"CORTEXMUX_CONFIG": str(config)},
    )
    assert result.exit_code == 0
    assert '"Authorization": "secret"' not in result.stdout
    assert '"Authorization": "**********"' in result.stdout


def test_invalid_arguments_have_usage_exit() -> None:
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 2


class FakeRegistry:
    """CLI registry test double."""

    def list(self) -> list[Any]:
        return [type("Provider", (), {"name": "fake"})()]


class FakeMux:
    """Context-managed facade test double."""

    registry = FakeRegistry()

    def __enter__(self) -> FakeMux:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def list_models(self, provider: str) -> list[ModelInfo]:
        return [ModelInfo(name="model", provider=provider)]

    def qualify_models(self, suite: QualificationSuite) -> QualificationManifest:
        return QualificationManifest(
            machine=MachineProfile(system="test", release="1", architecture="arm64"),
            candidates=[],
            recommendations=[],
        )

    def run(self, task: str, **fields: Any) -> TextResponse:
        return TextResponse(
            task=TaskType.TEXT_GENERATION,
            provider=str(fields.get("provider", "fake")),
            model=str(fields.get("model", "model")),
            request_id="r",
            content=f"{task} result",
        )

    def chat(self, prompt: str, **fields: Any) -> ChatResponse:
        return ChatResponse(
            provider=str(fields.get("provider", "fake")),
            model=str(fields.get("model", "model")),
            request_id="r",
            content="chat result",
        )

    def generate_image(self, prompt: str, **fields: Any) -> ImageGenerationResponse:
        return ImageGenerationResponse(
            provider="comfyui",
            request_id="r",
            prompt_id="p",
            images=[
                ImageArtifact(
                    path="/tmp/image.png",
                    filename="image.png",
                    node_id="9",
                    sequence=1,
                )
            ],
        )

    def analyze_data(self, source: Path, **fields: Any) -> DataAnalysisResponse:
        return DataAnalysisResponse(
            provider="data",
            request_id="r",
            engine="pandas",
            profile={},
            report_markdown="# Report\n",
        )


@pytest.fixture
def fake_mux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace CLI facade construction with a deterministic test double."""
    monkeypatch.setattr(
        CortexMux,
        "from_env",
        classmethod(lambda cls, **kwargs: FakeMux()),
    )


def test_provider_models_and_generation_commands(fake_mux: None, tmp_path: Path) -> None:
    assert runner.invoke(app, ["providers", "--json"]).exit_code == 0
    models = runner.invoke(app, ["models", "list", "--provider", "fake", "--json"])
    assert models.exit_code == 0 and "model" in models.stdout
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "candidates": [{"id": "one", "provider": "fake", "model": "model"}],
                "cases": [
                    {
                        "id": "case",
                        "task": "chat",
                        "prompt": "x",
                        "validator": "exact_text",
                        "expected_text": "OK",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    qualified = runner.invoke(
        app,
        [
            "models",
            "qualify",
            "--suite",
            str(suite),
            "--output",
            str(manifest),
        ],
    )
    assert qualified.exit_code == 0 and manifest.is_file()
    for command in ("chat", "generate"):
        result = runner.invoke(
            app,
            [command, "--prompt", "hello", "--model", "model"],
        )
        assert result.exit_code == 0
        assert "result" in result.stdout
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    vision = runner.invoke(
        app,
        [
            "vision",
            "--image",
            str(image),
            "--prompt",
            "look",
            "--model",
            "model",
        ],
    )
    assert vision.exit_code == 0
    embedded = runner.invoke(app, ["embed", "--text", "one", "--model", "model", "--json"])
    assert embedded.exit_code == 0


def test_image_data_and_workflow_commands(fake_mux: None, tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        '{"6":{"class_type":"Text","inputs":{"text":"old"}}}',
        encoding="utf-8",
    )
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        '{"prompt":{"node_id":"6","input":"text"}}',
        encoding="utf-8",
    )
    image = runner.invoke(
        app,
        [
            "image",
            "generate",
            "--workflow",
            str(workflow),
            "--bindings",
            str(bindings),
            "--prompt",
            "hello",
            "--json",
        ],
    )
    assert image.exit_code == 0
    source = tmp_path / "data.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    data = runner.invoke(
        app,
        ["data", "analyze", str(source), "--require-verified-calculations"],
    )
    assert data.exit_code == 0
    valid = runner.invoke(
        app,
        ["workflows", "validate", str(workflow), "--bindings", str(bindings)],
    )
    assert valid.exit_code == 0 and "Valid" in valid.stdout


def test_workflow_catalog_list_command(tmp_path: Path) -> None:
    workflow = tmp_path / "graph.json"
    workflow.write_text(
        '{"6":{"class_type":"Text","inputs":{"text":"old"}}}',
        encoding="utf-8",
    )
    (tmp_path / "demo.cortexmux.json").write_text(
        '{"name":"demo","description":"CLI test","workflow":"graph.json",'
        '"bindings":{"prompt":{"node_id":"6","input":"text"}}}',
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text(
        f'[providers.ollama]\nenabled=false\n[providers.comfyui]\nworkflow_dir="{tmp_path}"\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["workflows", "list", "--json"],
        env={"CORTEXMUX_CONFIG": str(config)},
    )
    assert result.exit_code == 0
    assert '"name": "demo"' in result.stdout
    assert '"binding_fields": [' in result.stdout
