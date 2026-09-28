"""Check TypeSafe input/output contracts without judging decision quality."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "object_noul",
        "state": {
            "url": "https://example.org/robots/r7",
            "title": "Bras robotique R7",
            "text": "Robot à six axes, charge utile de 10 kg.",
        },
        "questions": {
            "relevant": {"type": "noul", "instructions": "Cette page décrit-elle un robot ?"},
            "extractable": {
                "type": "noul",
                "instructions": "Cette page contient-elle une caractéristique technique ?",
            },
        },
    },
    {
        "id": "string_choice",
        "state": "Le R7 est un bras robotique industriel à six axes.",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Quel type de produit est décrit ?",
                "criteria": {"robot": "Produit robotique", "other": "Autre produit"},
            }
        },
    },
    {
        "id": "array_score",
        "state": ["R7", "Robot industriel", "Six axes", "Charge utile de 10 kg"],
        "questions": {
            "detail": {
                "type": "score",
                "instructions": "Combien de détails techniques explicites sont présents ?",
                "criteria": ["Aucun", "Un", "Plusieurs"],
            }
        },
    },
)


def _write_json(path: Path, value: object) -> None:
    """Create a private JSON artifact without overwriting a previous result."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _probability(value: object) -> bool:
    """Check the JSON number contract for one probability."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _distribution(value: object, keys: set[str]) -> bool:
    """Check option IDs, probability bounds, and normalized mass."""
    return (
        isinstance(value, dict)
        and set(value) == keys
        and all(_probability(item) for item in value.values())
        and math.isclose(sum(value.values()), 1.0, abs_tol=0.01)
    )


def check_contract(request: dict[str, Any], response: dict[str, Any]) -> list[str]:
    """Return integration contract failures for one serialized response."""
    failures: list[str] = []
    if response.get("task") != "decision" or response.get("provider") != "typesafe":
        failures.append("task/provider")
    model = response.get("model")
    if not isinstance(model, str) or not model:
        failures.append("response model")
    elif re.fullmatch(r"jev-\d+\.\d+\.\d+", request["model"]) and model != request["model"]:
        failures.append("pinned model identity")
    request_id = response.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        failures.append("request_id")
    routing = response.get("routing")
    if not isinstance(routing, dict) or (
        routing.get("request_id") != request_id
        or routing.get("selected_provider") != "typesafe"
        or routing.get("selected_model") != request["model"]
        or routing.get("task") != "decision"
    ):
        failures.append("routing")
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(
        not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool) or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        failures.append("usage")
    answers = response.get("answers")
    questions = request["questions"]
    if not isinstance(answers, dict) or set(answers) != set(questions):
        failures.append("answer IDs")
        return failures
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            failures.append(f"{key}: answer type")
            continue
        if question["type"] == "noul":
            if not _probability(answer.get("noul")):
                failures.append(f"{key}: noul probability")
        elif question["type"] == "choice":
            probabilities = answer.get("probabilities")
            if not _distribution(probabilities, set(question["criteria"])):
                failures.append(f"{key}: choice distribution")
            elif answer.get("choice") not in probabilities:
                failures.append(f"{key}: selected choice")
            if not _probability(answer.get("confidence")):
                failures.append(f"{key}: confidence")
        elif question["type"] == "score":
            legend = {str(index): label for index, label in enumerate(question["criteria"])}
            if answer.get("legend") != legend:
                failures.append(f"{key}: score legend")
            if not _distribution(answer.get("probabilities"), set(legend)):
                failures.append(f"{key}: score distribution")
            score = answer.get("score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0 <= score <= len(legend) - 1
            ):
                failures.append(f"{key}: score")
            if not _probability(answer.get("confidence")):
                failures.append(f"{key}: confidence")
    return failures


def verify_run(run_dir: Path) -> int:
    """Check saved prompts and responses without an API key or network access."""
    prompt = json.loads((run_dir / "prompt.json").read_text(encoding="utf-8"))
    log = json.loads((run_dir / "log.json").read_text(encoding="utf-8"))
    responses = {item["case_id"]: item for item in log["results"]}
    reports: list[dict[str, object]] = []
    for case in prompt["cases"]:
        item = responses.get(case["id"])
        failures = (
            check_contract(case["request"], item["response"])
            if item is not None and isinstance(item.get("response"), dict)
            else ["missing response"]
        )
        reports.append({"case_id": case["id"], "contract_ok": not failures, "failures": failures})
    if set(responses) != {case["id"] for case in prompt["cases"]}:
        reports.append({"case_id": "_run", "contract_ok": False, "failures": ["case IDs"]})
    passed = all(item["contract_ok"] for item in reports) and log.get("status") == "ok"
    report = {
        "run_id": prompt.get("run_id"),
        "contract_ok": passed,
        "cases": reports,
    }
    report_path = run_dir / f"contract-report-{uuid4().hex[:8]}.json"
    _write_json(report_path, report)
    print(f"Contrat : {'OK' if passed else 'ÉCHEC'}")
    print(f"Rapport : {report_path}")
    return 0 if passed else 1


def _config() -> Any:
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
    """Run or replay technical input/output checks for TypeSafe."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write requests without API calls")
    mode.add_argument("--verify-run", type=Path, help="Check an existing prompt/log directory")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/typesafe-contract"))
    args = parser.parse_args()
    if args.verify_run is not None:
        return verify_run(args.verify_run)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    prompt = {
        "run_id": run_id,
        "cases": [
            {
                "id": case["id"],
                "request": {
                    "state": case["state"],
                    "model": args.model,
                    "questions": case["questions"],
                },
            }
            for case in CASES
        ],
    }
    _write_json(run_dir / "prompt.json", prompt)
    log: dict[str, Any] = {"run_id": run_id, "status": "dry_run", "results": []}
    if not args.dry_run:
        if not os.environ.get("TYPESAFE_API_KEY"):
            if sys.stdin.isatty():
                os.environ["TYPESAFE_API_KEY"] = getpass.getpass("Clé TypeSafe : ")
            else:
                log["status"] = "missing_api_key"
        if log["status"] != "missing_api_key":
            try:
                from pydantic import TypeAdapter

                from cortexmux import CortexMux
                from cortexmux.core.exceptions import CortexMuxError
                from cortexmux.schemas.decisions import DecisionQuestion

                adapter = TypeAdapter(DecisionQuestion)
                with CortexMux(_config()) as mux:
                    for case in prompt["cases"]:
                        request = case["request"]
                        questions = {
                            key: adapter.validate_python(value)
                            for key, value in request["questions"].items()
                        }
                        try:
                            response = mux.decide(request["state"], questions, model=args.model)
                            serialized = response.model_dump(mode="json")
                            failures = check_contract(request, serialized)
                            log["results"].append(
                                {
                                    "case_id": case["id"],
                                    "response": serialized,
                                    "contract_ok": not failures,
                                    "failures": failures,
                                }
                            )
                            print(f"{case['id']}: {failures if failures else 'OK'}")
                            if failures:
                                log["status"] = "contract_failed"
                                break
                        except CortexMuxError as exc:
                            log["results"].append({"case_id": case["id"], "error": exc.to_dict()})
                            log["status"] = "api_error"
                            break
                if log["status"] == "dry_run":
                    log["status"] = "ok"
            except ImportError as exc:
                log["status"] = "import_error"
                log["error"] = str(exc)
            except CortexMuxError as exc:
                log["status"] = "api_error"
                log["error"] = exc.to_dict()
    _write_json(run_dir / "log.json", log)
    print(f"Prompts : {run_dir / 'prompt.json'}")
    print(f"Journal : {run_dir / 'log.json'}")
    print(f"État : {log['status']}")
    return 0 if log["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
