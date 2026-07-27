"""ComfyUI workflow schema compatibility exports."""

from cortexmux.providers.comfyui.catalog import WorkflowCatalogItem, WorkflowManifest
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition

__all__ = [
    "InputBinding",
    "WorkflowCatalogItem",
    "WorkflowDefinition",
    "WorkflowManifest",
]
