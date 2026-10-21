"""Safe serialization of JSON-compatible values for optional YAML exports."""

from __future__ import annotations

from typing import Any

from cortexmux.core.exceptions import OptionalDependencyError


def dump_yaml(value: Any) -> str:
    """Serialize a JSON-compatible value as Unicode YAML using the optional extra."""
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError("PyYAML", "yaml") from exc
    return str(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
