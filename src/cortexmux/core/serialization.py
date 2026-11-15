"""Serialization of JSON-compatible values for document exports."""

from __future__ import annotations

import json
from typing import Any

from cortexmux.core.exceptions import OptionalDependencyError


def dump_yaml(value: Any) -> str:
    """Serialize a JSON-compatible value as Unicode YAML using the optional extra."""
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError("PyYAML", "yaml") from exc
    return str(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def dump_markdown(value: Any) -> str:
    """Render structured values as a fenced JSON block in a Markdown document."""
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    return f"```json\n{payload}\n```\n"
