"""Tests for the self-contained local demonstration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from cortexmux.providers.comfyui.catalog import WorkflowCatalog
from cortexmux.providers.comfyui.workflow import WorkflowDefinition
from cortexmux.schemas import ModelInfo

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_PATH = PROJECT_ROOT / "examples" / "full_local_demo.py"


def load_demo() -> ModuleType:
    """Load the example as a module without making examples a package."""
    spec = importlib.util.spec_from_file_location("full_local_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_csv_creation_and_model_resolution(tmp_path: Path) -> None:
    demo = load_demo()
    csv_path = demo.create_demo_csv(tmp_path / "demo.csv")
    content = csv_path.read_text(encoding="utf-8")
    assert "revenue_eur" in content
    assert "tok_demo_nord" in content
    models = [
        ModelInfo(name="other:latest", provider="ollama"),
        ModelInfo(name="qwen2.5-coder:7b", provider="ollama"),
    ]
    assert demo.resolve_model(models, "qwen2,5-coder") == "qwen2.5-coder:7b"
    assert demo.resolve_model(models, "missing") is None


def test_bundled_workflow_and_bindings_are_compatible() -> None:
    demo = load_demo()
    definition = WorkflowDefinition(
        name="demo",
        workflow=demo.DEFAULT_WORKFLOW_PATH,
        input_bindings=demo.load_bindings(demo.DEFAULT_BINDINGS_PATH),
        expected_output_nodes=["9"],
    )
    graph, ignored = definition.bind(
        {
            "prompt": "demo",
            "negative_prompt": "bad",
            "checkpoint": "checkpoint.safetensors",
            "seed": 42,
            "width": 512,
            "height": 512,
            "steps": 10,
            "guidance": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
        }
    )
    assert ignored == []
    assert graph["4"]["inputs"]["ckpt_name"] == "checkpoint.safetensors"
    assert graph["6"]["inputs"]["text"] == "demo"
    catalog = WorkflowCatalog(demo.DEFAULT_WORKFLOW_PATH.parent)
    assert catalog.get("text-to-image").expected_output_nodes == ["9"]


def test_demo_analysis_plan_covers_requested_dimensions() -> None:
    demo = load_demo()
    operations = [step["operation"] for step in demo.DEMO_ANALYSIS_PLAN["steps"]]
    assert "time_series" in operations
    assert "iqr_outliers" in operations
    groupings = [
        step["group_by"]
        for step in demo.DEMO_ANALYSIS_PLAN["steps"]
        if step["operation"] == "group_by"
    ]
    assert ["region"] in groupings
    assert ["category"] in groupings
