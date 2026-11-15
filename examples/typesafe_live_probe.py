"""Record reproducible TypeSafe prompts and responses for synthetic web pages.

Run with --dry-run to inspect the prompts without an API key or network access.
Live runs require an API key from the environment or an interactive hidden prompt.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from uuid import uuid4

if TYPE_CHECKING:
    from cortexmux.core.config import CortexMuxConfig


class ProbeCase(TypedDict):
    """Synthetic page and expected human labels for one probe."""

    id: str
    expected: dict[str, bool]
    state: dict[str, str]


QUESTIONS: dict[str, dict[str, str]] = {
    "relevant": {
        "type": "noul",
        "instructions": (
            "Le titre et le texte décrivent-ils principalement un robot spécifique "
            "ou un modèle de robot ? Ignore les caractéristiques techniques et ne "
            "te fie pas à l'URL seule."
        ),
    },
    "extractable": {
        "type": "noul",
        "instructions": (
            "Cette page contient-elle au moins une caractéristique technique explicite, "
            "attribuable au robot présenté et exploitable pour une fiche structurée "
            "(par exemple charge utile, portée, nombre d'axes ou autonomie) ?"
        ),
    },
}

CASES: tuple[ProbeCase, ...] = (
    {
        "id": "robot_with_specs",
        "expected": {"relevant": True, "extractable": True},
        "state": {
            "url": "https://example.org/robots/r7",
            "title": "Bras robotique industriel R7",
            "text": (
                "Le R7 est un robot articulé industriel à six axes. "
                "Sa charge utile est de 10 kg et sa portée est de 1,2 m."
            ),
        },
    },
    {
        "id": "robot_without_specs",
        "expected": {"relevant": True, "extractable": False},
        "state": {
            "url": "https://example.org/robots/nova",
            "title": "Présentation du robot Nova",
            "text": (
                "Nova est un nouveau robot domestique. Cette page annonce son arrivée "
                "prochaine sans fournir de caractéristique technique."
            ),
        },
    },
    {
        "id": "unrelated_page",
        "expected": {"relevant": False, "extractable": False},
        "state": {
            "url": "https://example.org/cuisine/cafe",
            "title": "Préparer un café filtre",
            "text": "Guide de préparation du café filtre avec de l'eau chaude et du café moulu.",
        },
    },
    {
        "id": "misleading_robot_url",
        "expected": {"relevant": False, "extractable": False},
        "state": {
            "url": "https://example.org/robots/coffee-guide",
            "title": "Préparer un café filtre",
            "text": "Guide de préparation du café filtre avec de l'eau chaude et du café moulu.",
        },
    },
)


def _write_json(path: Path, value: object) -> None:
    """Create a private JSON artifact without overwriting an earlier run."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _config() -> CortexMuxConfig:
    """Enable only TypeSafe and approve its exact provider host."""
    from cortexmux.core.config import CortexMuxConfig

    return CortexMuxConfig.model_validate(
        {
            "core": {"approved_hosts": ["api.typesafe.ai"]},
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "typesafe": {"enabled": True},
            },
        }
    )


def main() -> int:
    """Write exact prompts and normalized results for later review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without API calls")
    parser.add_argument(
        "--model", default="jev-1.13.0", help="TypeSafe alias or versioned model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/typesafe-probe"),
        help="Ignored local directory for run artifacts",
    )
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    prompts = {
        "run_id": run_id,
        "note": "Synthetic cases; expected labels are for review and are not sent to TypeSafe.",
        "cases": [
            {
                "id": case["id"],
                "expected": case["expected"],
                "request": {
                    "state": case["state"],
                    "model": args.model,
                    "questions": QUESTIONS,
                },
            }
            for case in CASES
        ],
    }
    _write_json(run_dir / "prompt.json", prompts)
    results: list[dict[str, object]] = []
    log: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "dry_run" if args.dry_run else "running",
        "threshold_used_by_this_example": 0.75,
        "results": results,
    }

    if not args.dry_run:
        if not os.environ.get("TYPESAFE_API_KEY"):
            if not sys.stdin.isatty():
                log["status"] = "missing_api_key"
                log["error"] = "Set TYPESAFE_API_KEY or run from an interactive terminal."
            else:
                os.environ["TYPESAFE_API_KEY"] = getpass.getpass("Clé TypeSafe : ")
        if log["status"] == "running":
            try:
                from cortexmux import CortexMux
                from cortexmux.core.exceptions import CortexMuxError, ProviderResponseError
                from cortexmux.schemas import NoulQuestion
                from cortexmux.schemas.decisions import NoulAnswer

                questions = {
                    key: NoulQuestion.model_validate(question)
                    for key, question in QUESTIONS.items()
                }
                with CortexMux(_config()) as mux:
                    for case in CASES:
                        try:
                            response = mux.decide(case["state"], questions, model=args.model)
                            relevant = response.answers["relevant"]
                            extractable = response.answers["extractable"]
                            if not isinstance(relevant, NoulAnswer) or not isinstance(
                                extractable, NoulAnswer
                            ):
                                raise ProviderResponseError(
                                    "Expected Noul answers for both questions.", provider="typesafe"
                                )
                            expected_second_validation = all(case["expected"].values())
                            route_passes = relevant.noul > 0.75 and extractable.noul > 0.75
                            results.append(
                                {
                                    "case_id": case["id"],
                                    "expected": case["expected"],
                                    "observed": {
                                        "relevant": relevant.noul,
                                        "extractable": extractable.noul,
                                    },
                                    "expected_second_validation": expected_second_validation,
                                    "would_request_second_validation": route_passes,
                                    "matches_expected_route": route_passes
                                    == expected_second_validation,
                                    "response": response.model_dump(mode="json"),
                                }
                            )
                            print(
                                f"{case['id']}: relevant={relevant.noul:.3f}, "
                                f"extractable={extractable.noul:.3f}, "
                                f"second_validation={route_passes}, "
                                f"expected={expected_second_validation}"
                            )
                        except CortexMuxError as exc:
                            results.append({"case_id": case["id"], "error": exc.to_dict()})
                            log["status"] = "error"
                            print(f"{case['id']}: {type(exc).__name__}")
                            break
            except ImportError as exc:
                log["status"] = "error"
                log["error"] = {"error": "ImportError", "message": str(exc)}
            except CortexMuxError as exc:
                log["status"] = "error"
                log["error"] = exc.to_dict()
            if log["status"] == "running":
                log["status"] = "ok"

    log["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(run_dir / "log.json", log)
    print(f"Prompts : {run_dir / 'prompt.json'}")
    print(f"Journal : {run_dir / 'log.json'}")
    print(f"État : {log['status']}")
    return 0 if log["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
