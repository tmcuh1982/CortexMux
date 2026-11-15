"""Safe JSON/YAML persistence for qualification suites and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from cortexmux.core.exceptions import ConfigurationError, OptionalDependencyError, OutputPathError
from cortexmux.selection.schemas import QualificationManifest, QualificationSuite

_MAX_CONFIG_BYTES = 1_000_000


def load_qualification_suite(path: Path | str) -> QualificationSuite:
    """Load and validate a bounded JSON or YAML benchmark suite."""
    source = Path(path).expanduser()
    try:
        if not source.is_file() or source.stat().st_size > _MAX_CONFIG_BYTES:
            raise ConfigurationError(
                "Qualification suite is missing or too large.", path=str(source)
            )
        text = source.read_text(encoding="utf-8")
        raw = _load_mapping(text, source.suffix.lower())
        return QualificationSuite.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigurationError(
            "Qualification suite could not be loaded.", path=str(source)
        ) from exc


def write_qualification_manifest(
    manifest: QualificationManifest,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one JSON or YAML manifest without silent overwrite."""
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise OutputPathError("Output file already exists.", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    payload = manifest.model_dump(mode="json")
    text = _dump_mapping(payload, suffix)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(target)
    return target


def _load_mapping(text: str, suffix: str) -> dict[str, Any]:
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        yaml = _yaml_module()
        raw = yaml.safe_load(text)
    else:
        raise ConfigurationError("Qualification files must use .json, .yaml, or .yml.")
    if not isinstance(raw, dict):
        raise ConfigurationError("Qualification file root must be an object.")
    return raw


def _dump_mapping(payload: dict[str, Any], suffix: str) -> str:
    if suffix == ".json":
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if suffix in {".yaml", ".yml"}:
        yaml = _yaml_module()
        return str(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    raise ConfigurationError("Qualification files must use .json, .yaml, or .yml.")


def _yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError("PyYAML", "yaml") from exc
    return yaml
