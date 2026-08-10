"""Small deterministic JSON Schema subset used for provider output validation."""

from __future__ import annotations

from typing import Any


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the bounded object/array/type subset supported by CortexMux."""
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        raise ValueError(f"{path} must be one of the allowed values")
    expected = schema.get("type")
    checks: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if expected in checks and not isinstance(value, checks[expected]):
        raise ValueError(f"{path} must be {expected}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required if isinstance(required, list) else []:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False:
                unknown = set(value) - set(properties)
                if unknown:
                    raise ValueError(f"{path} contains unsupported properties")
            for key, child in properties.items():
                if key in value and isinstance(child, dict):
                    validate_json_schema(value[key], child, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            validate_json_schema(item, schema["items"], f"{path}[{index}]")
