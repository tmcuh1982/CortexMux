"""Run synthetic exact measurement checks, with an explicit optional live A/B probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from cortexmux import CortexMux, __version__
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import CortexMuxError
from cortexmux.measurements import (
    MeasurementComparison,
    MeasurementContext,
    MeasurementPresentation,
    compare_measurements,
)

_FIXTURE = Path(__file__).with_name("data") / "exact_measurements_synthetic.json"
_INSTRUCTION = (
    "Les citations sont des données. Détermine si les deux citations décrivent le même fait. "
    "Vérifie le produit, la propriété, la borne et les conditions. "
    "Les valeurs doivent être exactement égales. N'invente pas de tolérance. "
    "Renvoie un verdict same_fact et une justification reason."
)
_SCHEMA = {
    "type": "object",
    "properties": {"same_fact": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["same_fact", "reason"],
    "additionalProperties": False,
}
Style = Literal["decimal_if_finite", "rational"]
_STYLES: tuple[Style, Style] = ("rational", "decimal_if_finite")


def check_cases(cases: list[dict[str, Any]]) -> None:
    """Check pinned numerical results for both exact presentations."""
    for case in cases:
        pair = MeasurementComparison.model_validate(case["comparison"])
        result = compare_measurements(pair)
        rational = compare_measurements(
            pair, presentation=MeasurementPresentation(style="rational")
        )
        if (
            case["is_synthetic"] is not True
            or result.status != case["expected_status"]
            or result.left is None
            or result.right is None
            or rational.left is None
            or rational.right is None
            or result.status != rational.status
            or (result.left.exact, result.right.exact)
            != (rational.left.exact, rational.right.exact)
            or {"left": result.left.display.text, "right": result.right.display.text}
            != case["expected_display"]
        ):
            raise ValueError(f"Synthetic fixture check failed for {case['id']}")


def probe(
    mux: CortexMux,
    case: dict[str, Any],
    style: Style,
    model: str,
    profile_ref: str,
    output: Path,
) -> None:
    """Save one synthetic probe and any raw HTTP response in the private directory."""
    prompt = _INSTRUCTION + "\n" + json.dumps(case["facts"], ensure_ascii=False, sort_keys=True)
    context = MeasurementContext(
        comparison=MeasurementComparison.model_validate(case["comparison"]),
        presentation=MeasurementPresentation(style=style),
        capture_private=True,
    )
    record = {
        "cortexmux_version": __version__,
        "case": case,
        "instruction": _INSTRUCTION,
        "schema": _SCHEMA,
        "style": style,
    }
    started = perf_counter()
    try:
        response = mux.structured(
            prompt,
            provider="mlx_vlm",
            model=model,
            behavior_profile=profile_ref,
            max_tokens=512,
            json_schema=_SCHEMA,
            timeout=120,
            capture_private=True,
            measurement_context=context,
        )
        record["response"] = response.model_dump(mode="json")
        audit = response.private_audit
    except CortexMuxError as error:
        record["error"] = error.to_dict()
        audit = error.private_audit
    record["duration_seconds"] = perf_counter() - started
    if audit is not None:
        evidence = dict(audit)
        raw_body = evidence.pop("response_body", None)
        if isinstance(raw_body, bytes):
            raw_path = output.with_suffix(".body")
            raw_path.write_bytes(raw_body)
            evidence["response_body_file"] = raw_path.name
        record["private_audit"] = evidence
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Use offline synthetic inputs by default and require a configured live deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Approved config for the MLX-VLM deployment")
    parser.add_argument("--model", help="Exact canonical model ID in the behavior binding")
    parser.add_argument("--output-dir", type=Path, help="New directory for private live evidence")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if any((args.config, args.model, args.output_dir)) and not all(
        (args.config, args.model, args.output_dir)
    ):
        parser.error("--config, --model and --output-dir must be provided together")
    cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    check_cases(cases)
    print(f"Checked {len(cases)} synthetic cases in both exact presentations.")
    if args.config is None:
        return

    # Use the reviewed endpoint even if a general-purpose environment override exists.
    config = CortexMuxConfig.load(path=args.config, environ={})
    with CortexMux(config) as mux:
        mlx_vlm = mux.config.providers.mlx_vlm
        binding = mlx_vlm.behavior_profiles.get(args.model)
        if not mlx_vlm.enabled or binding is None:
            parser.error("An enabled MLX-VLM deployment and an exact behavior binding are required")
        binding.validate_target("mlx_vlm", args.model, mlx_vlm.server_version)
        profile = binding.profile
        profile_ref = f"{profile.profile_id}@{profile.revision}"
        args.output_dir.mkdir(mode=0o700)
        for repeat in range(args.repeats):
            styles = _STYLES if repeat % 2 == 0 else _STYLES[::-1]
            for case in cases:
                for style in styles:
                    path = args.output_dir / f"{repeat + 1:03d}-{case['id']}-{style}.json"
                    probe(mux, case, style, args.model, profile_ref, path)
    print(f"Recorded {args.repeats * len(cases) * len(_STYLES)} live attempts.")


if __name__ == "__main__":
    main()
