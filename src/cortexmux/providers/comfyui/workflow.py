"""ComfyUI API-format workflow loading, validation, and safe binding."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cortexmux.core.exceptions import WorkflowValidationError


class InputBinding(BaseModel):
    """Map a normalized field to a ComfyUI node input."""

    node_id: str
    input: str


class WorkflowDefinition(BaseModel):
    """A named ComfyUI API graph and its normalized bindings."""

    name: str
    workflow: Path | dict[str, Any]
    input_bindings: dict[str, InputBinding] = Field(default_factory=dict)
    expected_output_nodes: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def load_graph(self) -> dict[str, Any]:
        """Load and validate the graph without mutating caller-owned data."""
        if isinstance(self.workflow, Path):
            try:
                raw = json.loads(self.workflow.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkflowValidationError(
                    "Workflow file is missing or contains invalid JSON.",
                    path=str(self.workflow),
                ) from exc
        else:
            raw = copy.deepcopy(self.workflow)
        if not isinstance(raw, dict) or not raw:
            raise WorkflowValidationError("Workflow must be a non-empty API-format object.")
        for node_id, node in raw.items():
            if not isinstance(node_id, str) or not isinstance(node, dict):
                raise WorkflowValidationError("Workflow nodes must be keyed JSON objects.")
            if not isinstance(node.get("class_type"), str) or not isinstance(
                node.get("inputs"), dict
            ):
                raise WorkflowValidationError(
                    "Workflow is not ComfyUI API format.", node_id=node_id
                )
        self._validate_references(raw)
        return raw

    def bind(
        self,
        values: dict[str, str | int | float | bool | None],
        *,
        strict: bool = True,
    ) -> tuple[dict[str, Any], list[str]]:
        """Deep-copy the graph and inject configured normalized values."""
        graph = self.load_graph()
        ignored: list[str] = []
        for field_name, value in values.items():
            if value is None:
                continue
            binding = self.input_bindings.get(field_name)
            if binding is None:
                if strict:
                    raise WorkflowValidationError(
                        "No workflow binding exists for supplied field.", field=field_name
                    )
                ignored.append(field_name)
                continue
            node = graph.get(binding.node_id)
            if not isinstance(node, dict):
                raise WorkflowValidationError(
                    "Binding references a missing node.",
                    field=field_name,
                    node_id=binding.node_id,
                )
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or binding.input not in inputs:
                raise WorkflowValidationError(
                    "Binding references a missing node input.",
                    field=field_name,
                    node_id=binding.node_id,
                    input=binding.input,
                )
            inputs[binding.input] = value
        for node_id in self.expected_output_nodes:
            if node_id not in graph:
                raise WorkflowValidationError("Expected output node is missing.", node_id=node_id)
        return graph, ignored

    def _validate_references(self, graph: dict[str, Any]) -> None:
        for field_name, binding in self.input_bindings.items():
            node = graph.get(binding.node_id)
            if not isinstance(node, dict):
                raise WorkflowValidationError(
                    "Binding references a missing node.",
                    field=field_name,
                    node_id=binding.node_id,
                )
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or binding.input not in inputs:
                raise WorkflowValidationError(
                    "Binding references a missing input.",
                    field=field_name,
                    node_id=binding.node_id,
                    input=binding.input,
                )
