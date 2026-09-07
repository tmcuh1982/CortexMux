"""Deterministic validation for CortexMux's documented JSON Schema subset."""

from __future__ import annotations

import re
from typing import Any, cast

_ANNOTATION_KEYWORDS = {
    "$id",
    "$schema",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_VALIDATION_KEYWORDS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "not",
    "oneOf",
    "pattern",
    "properties",
    "required",
    "type",
    "uniqueItems",
}
_SUPPORTED_KEYWORDS = _ANNOTATION_KEYWORDS | _VALIDATION_KEYWORDS
_NUMBER_TYPES = (int, float)


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate a value against the explicitly supported JSON Schema subset.

    Unknown validation keywords are rejected instead of being silently ignored.
    Annotation keywords are accepted but do not affect validation.
    """
    _validate_schema_shape(schema)
    _validate(value, schema, schema, path)


def validate_json_schema_definition(schema: dict[str, Any]) -> None:
    """Reject malformed or unsupported schema constructs before provider I/O."""
    _validate_schema_shape(schema)


def _validate_schema_shape(schema: dict[str, Any]) -> None:
    for keyword in schema:
        if keyword not in _SUPPORTED_KEYWORDS:
            raise ValueError(f"Unsupported JSON Schema keyword: {keyword}")
    for keyword in ("properties", "$defs"):
        children = schema.get(keyword)
        if children is not None and not isinstance(children, dict):
            raise ValueError(f"JSON Schema {keyword} must be an object")
        if isinstance(children, dict):
            for child in children.values():
                if not isinstance(child, dict):
                    raise ValueError(f"JSON Schema {keyword} values must be schemas")
                _validate_schema_shape(child)
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise ValueError("JSON Schema items must be a schema")
        _validate_schema_shape(items)
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, (bool, dict)):
        raise ValueError("JSON Schema additionalProperties must be boolean or a schema")
    if isinstance(additional, dict):
        _validate_schema_shape(additional)
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if branches is not None:
            if not isinstance(branches, list) or not branches:
                raise ValueError(f"JSON Schema {keyword} must be a non-empty array")
            for branch in branches:
                if not isinstance(branch, dict):
                    raise ValueError(f"JSON Schema {keyword} values must be schemas")
                _validate_schema_shape(branch)
    negated = schema.get("not")
    if negated is not None:
        if not isinstance(negated, dict):
            raise ValueError("JSON Schema not must be a schema")
        _validate_schema_shape(negated)


def _validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise ValueError("JSON Schema $ref must be a string")
        _validate(value, _resolve_reference(reference, root), root, path)

    for branch in schema.get("allOf", []):
        _validate(value, branch, root, path)
    if "anyOf" in schema and not _matches_any(value, schema["anyOf"], root, path):
        raise ValueError(f"{path} must match at least one anyOf schema")
    if "oneOf" in schema:
        matches = sum(_matches(value, branch, root, path) for branch in schema["oneOf"])
        if matches != 1:
            raise ValueError(f"{path} must match exactly one oneOf schema")
    if "not" in schema and _matches(value, schema["not"], root, path):
        raise ValueError(f"{path} must not match the excluded schema")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal the constant value")
    allowed = schema.get("enum")
    if allowed is not None:
        if not isinstance(allowed, list):
            raise ValueError("JSON Schema enum must be an array")
        if not any(_json_equal(value, candidate) for candidate in allowed):
            raise ValueError(f"{path} must be one of the allowed values")

    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        label = " or ".join(expected) if isinstance(expected, list) else expected
        raise ValueError(f"{path} must be {label}")

    if isinstance(value, dict):
        _validate_object(value, schema, root, path)
    elif isinstance(value, list):
        _validate_array(value, schema, root, path)
    elif isinstance(value, str):
        _validate_string(value, schema, path)
    elif _is_number(value):
        _validate_number(value, schema, path)


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        if not expected or not all(isinstance(item, str) for item in expected):
            raise ValueError("JSON Schema type array must contain strings")
        return any(_matches_type(value, item) for item in expected)
    if not isinstance(expected, str):
        raise ValueError("JSON Schema type must be a string or array of strings")
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_number(value)
    checks: dict[str, type] = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "null": type(None),
    }
    if expected not in checks:
        raise ValueError(f"Unsupported JSON Schema type: {expected}")
    return isinstance(value, checks[expected])


def _validate_object(
    value: dict[str, Any], schema: dict[str, Any], root: dict[str, Any], path: str
) -> None:
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ValueError("JSON Schema required must be an array of strings")
    for key in required:
        if key not in value:
            raise ValueError(f"{path}.{key} is required")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("JSON Schema properties must be an object")
    unknown = set(value) - set(properties)
    additional = schema.get("additionalProperties", True)
    if additional is False and unknown:
        raise ValueError(f"{path} contains unsupported properties")
    if isinstance(additional, dict):
        for key in unknown:
            _validate(value[key], additional, root, f"{path}.{key}")
    for key, child in properties.items():
        if key in value:
            _validate(value[key], child, root, f"{path}.{key}")
    _check_length(value, schema, path, "Properties")


def _validate_array(
    value: list[Any], schema: dict[str, Any], root: dict[str, Any], path: str
) -> None:
    items = schema.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            _validate(item, items, root, f"{path}[{index}]")
    _check_length(value, schema, path, "Items")
    if schema.get("uniqueItems") is True:
        for index, item in enumerate(value):
            if any(_json_equal(item, other) for other in value[:index]):
                raise ValueError(f"{path} must contain unique items")


def _validate_string(value: str, schema: dict[str, Any], path: str) -> None:
    _check_length(value, schema, path, "Length")
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError("JSON Schema pattern must be a string")
        if re.search(pattern, value) is None:
            raise ValueError(f"{path} must match the required pattern")


def _validate_number(value: int | float, schema: dict[str, Any], path: str) -> None:
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        limit = schema.get(keyword)
        if limit is not None:
            if not _is_number(limit):
                raise ValueError(f"JSON Schema {keyword} must be numeric")
            numeric_limit = cast(int | float, limit)
            valid = (
                (keyword == "minimum" and value >= numeric_limit)
                or (keyword == "maximum" and value <= numeric_limit)
                or (keyword == "exclusiveMinimum" and value > numeric_limit)
                or (keyword == "exclusiveMaximum" and value < numeric_limit)
            )
            if not valid:
                raise ValueError(f"{path} violates {keyword}")


def _check_length(value: Any, schema: dict[str, Any], path: str, suffix: str) -> None:
    minimum = schema.get(f"min{suffix}")
    maximum = schema.get(f"max{suffix}")
    if minimum is not None and (not isinstance(minimum, int) or isinstance(minimum, bool)):
        raise ValueError(f"JSON Schema min{suffix} must be an integer")
    if maximum is not None and (not isinstance(maximum, int) or isinstance(maximum, bool)):
        raise ValueError(f"JSON Schema max{suffix} must be an integer")
    if minimum is not None and len(value) < minimum:
        raise ValueError(f"{path} violates min{suffix}")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{path} violates max{suffix}")


def _resolve_reference(reference: str, root: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not reference.startswith(prefix):
        raise ValueError("Only local $defs JSON Schema references are supported")
    key = reference[len(prefix) :].replace("~1", "/").replace("~0", "~")
    definitions = root.get("$defs")
    candidate = definitions.get(key) if isinstance(definitions, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError(f"Unresolved JSON Schema reference: {reference}")
    return cast(dict[str, Any], candidate)


def _matches_any(
    value: Any, branches: list[dict[str, Any]], root: dict[str, Any], path: str
) -> bool:
    return any(_matches(value, branch, root, path) for branch in branches)


def _matches(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> bool:
    try:
        _validate(value, schema, root, path)
    except ValueError:
        return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, _NUMBER_TYPES) and not isinstance(value, bool)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_number(left) and _is_number(right):
        return bool(left == right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return bool(left == right)
