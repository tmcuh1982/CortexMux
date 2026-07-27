"""Discover reusable ComfyUI workflows from local catalog manifests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cortexmux.core.exceptions import WorkflowValidationError
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition


class WorkflowManifest(BaseModel):
    """On-disk descriptor for one reusable ComfyUI workflow."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    description: str | None = None
    workflow: str | dict[str, Any]
    bindings: str | dict[str, InputBinding] = Field(default_factory=dict)
    expected_output_nodes: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class WorkflowCatalogItem(BaseModel):
    """Safe public summary of a workflow catalog entry."""

    name: str
    description: str | None = None
    workflow_path: Path | None = None
    manifest_path: Path
    binding_fields: list[str] = Field(default_factory=list)
    expected_output_nodes: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class WorkflowCatalog:
    """Load named workflow definitions from ``*.cortexmux.json`` manifests."""

    manifest_pattern = "*.cortexmux.json"

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._items: dict[str, WorkflowCatalogItem] = {}
        self._definitions: dict[str, WorkflowDefinition] = {}
        self.refresh()

    def refresh(self) -> list[WorkflowCatalogItem]:
        """Rescan the catalog and return entries sorted by name."""
        items: dict[str, WorkflowCatalogItem] = {}
        definitions: dict[str, WorkflowDefinition] = {}
        if not self.root.exists():
            self._items = items
            self._definitions = definitions
            return []
        if not self.root.is_dir():
            raise WorkflowValidationError(
                "Workflow catalog path is not a directory.", path=str(self.root)
            )
        for manifest_path in sorted(self.root.rglob(self.manifest_pattern)):
            resolved_manifest = self._safe_path(manifest_path)
            manifest = self._load_manifest(resolved_manifest)
            if manifest.name in items:
                raise WorkflowValidationError(
                    "Workflow catalog contains a duplicate name.",
                    name=manifest.name,
                    path=str(resolved_manifest),
                )
            definition, workflow_path = self._definition(manifest, resolved_manifest)
            graph = definition.load_graph()
            for node_id in definition.expected_output_nodes:
                if node_id not in graph:
                    raise WorkflowValidationError(
                        "Expected output node is missing.",
                        name=definition.name,
                        node_id=node_id,
                    )
            item = WorkflowCatalogItem(
                name=manifest.name,
                description=manifest.description,
                workflow_path=workflow_path,
                manifest_path=resolved_manifest,
                binding_fields=sorted(definition.input_bindings),
                expected_output_nodes=definition.expected_output_nodes,
                metadata=manifest.metadata,
            )
            items[item.name] = item
            definitions[item.name] = definition
        self._items = items
        self._definitions = definitions
        return self.list()

    def list(self) -> list[WorkflowCatalogItem]:
        """Return defensive copies of catalog summaries sorted by name."""
        return [self._items[name].model_copy(deep=True) for name in sorted(self._items)]

    def get(self, name: str) -> WorkflowDefinition:
        """Return a defensive copy of a named workflow definition."""
        definition = self._definitions.get(name)
        if definition is None:
            raise WorkflowValidationError(
                "Workflow is not present in the catalog.",
                name=name,
                catalog=str(self.root),
            )
        return definition.model_copy(deep=True)

    def contains(self, name: str) -> bool:
        """Return whether the catalog contains a workflow name."""
        return name in self._definitions

    def _load_manifest(self, path: Path) -> WorkflowManifest:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowManifest.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise WorkflowValidationError(
                "Workflow catalog manifest is invalid.", path=str(path)
            ) from exc

    def _definition(
        self,
        manifest: WorkflowManifest,
        manifest_path: Path,
    ) -> tuple[WorkflowDefinition, Path | None]:
        workflow: Path | dict[str, Any]
        workflow_path: Path | None
        if isinstance(manifest.workflow, str):
            workflow_path = self._safe_path(manifest_path.parent / manifest.workflow)
            workflow = workflow_path
        else:
            workflow_path = None
            workflow = copy.deepcopy(manifest.workflow)

        if isinstance(manifest.bindings, str):
            bindings_path = self._safe_path(manifest_path.parent / manifest.bindings)
            try:
                raw_bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
                if not isinstance(raw_bindings, dict):
                    raise ValueError
                bindings = {
                    key: InputBinding.model_validate(value) for key, value in raw_bindings.items()
                }
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise WorkflowValidationError(
                    "Workflow catalog bindings are invalid.", path=str(bindings_path)
                ) from exc
        else:
            bindings = {
                key: value.model_copy(deep=True) for key, value in manifest.bindings.items()
            }

        return (
            WorkflowDefinition(
                name=manifest.name,
                workflow=workflow,
                input_bindings=bindings,
                expected_output_nodes=list(manifest.expected_output_nodes),
                metadata=dict(manifest.metadata),
            ),
            workflow_path,
        )

    def _safe_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.is_relative_to(self.root):
            raise WorkflowValidationError(
                "Workflow catalog path escapes its configured root.",
                path=str(path),
                catalog=str(self.root),
            )
        return resolved
