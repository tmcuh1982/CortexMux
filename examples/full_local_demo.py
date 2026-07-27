"""Run an end-to-end CortexMux demo against local Ollama and ComfyUI services.

The script always creates or reuses a deterministic CSV dataset and analyzes it.
When local services are available, it also:

* selects an installed ``qwen2.5-coder`` Ollama tag;
* runs a chat request and asks Ollama to interpret the CSV analysis;
* selects a ComfyUI checkpoint and executes the bundled API-format workflow.

Run ``python examples/full_local_demo.py --help`` for service and workflow options.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortexmux import CortexMux
from cortexmux.core.exceptions import CortexMuxError
from cortexmux.schemas import HealthStatus, ModelInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "examples" / "data" / "demo_sales.csv"
DEFAULT_WORKFLOW_PATH = PROJECT_ROOT / "examples" / "workflows" / "text_to_image.example.json"
DEFAULT_BINDINGS_PATH = (
    PROJECT_ROOT / "examples" / "workflows" / "text_to_image.bindings.example.json"
)

DEMO_ANALYSIS_REQUEST = (
    "Analyse le chiffre d'affaires mensuel, compare les régions et catégories, "
    "identifie les valeurs inhabituelles et propose trois actions concrètes. "
    "Réponds en français en distinguant les résultats calculés des interprétations."
)
DEMO_ANALYSIS_PLAN = {
    "steps": [
        {"operation": "shape"},
        {"operation": "missing_values"},
        {"operation": "describe"},
        {"operation": "date_parse", "columns": ["date"]},
        {
            "operation": "time_series",
            "columns": ["date", "revenue_eur"],
            "frequency": "ME",
        },
        {
            "operation": "group_by",
            "group_by": ["region"],
            "aggregations": {"revenue_eur": "sum", "units": "sum"},
        },
        {
            "operation": "group_by",
            "group_by": ["category"],
            "aggregations": {"revenue_eur": "sum", "units": "sum"},
        },
        {
            "operation": "top_n",
            "columns": ["revenue_eur"],
            "n": 5,
            "ascending": False,
        },
        {"operation": "iqr_outliers", "columns": ["revenue_eur"]},
    ]
}
DEMO_CHAT_REQUEST = (
    "Tu testes CortexMux avec le modèle local qwen2.5-coder. "
    "Réponds en français avec une phrase confirmant le nom du modèle et explique "
    "brièvement pourquoi une analyse déterministe est préférable à du code généré."
)
DEMO_IMAGE_PROMPT = (
    "A premium industrial agricultural sensor in a wheat field at sunrise, "
    "photorealistic product photography, clean composition, high detail"
)

DEMO_ROWS = [
    ("2026-01-05", "Nord", "Capteur", 12, 420.0, 4.7, 0, "tok_demo_nord"),
    ("2026-01-18", "Sud", "Passerelle", 7, 690.0, 4.4, 1, "tok_demo_sud"),
    ("2026-02-04", "Est", "Capteur", 15, 420.0, 4.6, 0, "tok_demo_est"),
    ("2026-02-21", "Ouest", "Abonnement", 28, 49.0, 4.2, 2, "tok_demo_ouest"),
    ("2026-03-02", "Nord", "Passerelle", 9, 690.0, 4.8, 0, "tok_demo_nord"),
    ("2026-03-23", "Sud", "Capteur", 19, 420.0, 4.1, 3, "tok_demo_sud"),
    ("2026-04-07", "Est", "Abonnement", 36, 49.0, 4.5, 1, "tok_demo_est"),
    ("2026-04-26", "Ouest", "Capteur", 17, 420.0, 4.3, 0, "tok_demo_ouest"),
    ("2026-05-09", "Nord", "Capteur", 21, 420.0, 4.9, 0, "tok_demo_nord"),
    ("2026-05-19", "Sud", "Passerelle", 11, 690.0, 4.0, 4, "tok_demo_sud"),
    ("2026-06-03", "Est", "Capteur", 18, 420.0, 4.6, 1, "tok_demo_est"),
    ("2026-06-27", "Ouest", "Abonnement", 42, 49.0, 4.4, 0, "tok_demo_ouest"),
    ("2026-07-06", "Nord", "Passerelle", 13, 690.0, 4.8, 0, "tok_demo_nord"),
    ("2026-07-22", "Sud", "Capteur", 24, 420.0, 3.8, 6, "tok_demo_sud"),
    ("2026-08-08", "Est", "Abonnement", 48, 49.0, 4.7, 0, "tok_demo_est"),
    ("2026-08-24", "Ouest", "Capteur", 20, 420.0, 4.5, 1, "tok_demo_ouest"),
]

console = Console()


def create_demo_csv(path: Path, *, overwrite: bool = False) -> Path:
    """Create the deterministic demonstration dataset and return its path."""
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "date",
                "region",
                "category",
                "units",
                "unit_price_eur",
                "satisfaction",
                "support_incidents",
                "api_token",
                "revenue_eur",
            ]
        )
        for row in DEMO_ROWS:
            writer.writerow([*row, round(row[3] * row[4], 2)])
    return path


def resolve_model(models: list[ModelInfo], requested: str) -> str | None:
    """Resolve an Ollama model, accepting tagged and untagged Qwen names."""
    normalized = requested.replace(",", ".").lower()
    exact = next((model.name for model in models if model.name.lower() == normalized), None)
    if exact:
        return exact
    prefix = f"{normalized}:"
    return next(
        (model.name for model in models if model.name.lower().startswith(prefix)),
        None,
    )


def load_bindings(path: Path) -> dict[str, dict[str, str]]:
    """Load a JSON binding object used by the bundled ComfyUI workflow."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Le fichier de bindings doit contenir un objet JSON.")
    return {
        str(key): {"node_id": str(value["node_id"]), "input": str(value["input"])}
        for key, value in data.items()
        if isinstance(value, dict) and "node_id" in value and "input" in value
    }


def list_managed_models(
    mux: CortexMux,
) -> tuple[dict[str, HealthStatus], dict[str, list[ModelInfo]]]:
    """Print health and models for every provider registered in CortexMux."""
    statuses = {status.provider: status for status in mux.health()}
    models_by_provider: dict[str, list[ModelInfo]] = {}
    table = Table(title="Providers et modèles gérés par CortexMux")
    table.add_column("Provider")
    table.add_column("État")
    table.add_column("Modèles / moteurs")

    for provider in mux.registry.list():
        status = statuses[provider.name]
        models: list[ModelInfo] = []
        if status.available:
            try:
                models = mux.list_models(provider.name)
            except CortexMuxError as exc:
                status = status.model_copy(update={"available": False, "message": str(exc)})
                statuses[provider.name] = status
        models_by_provider[provider.name] = models
        names = ", ".join(model.name for model in models) or "aucun / indisponible"
        table.add_row(
            provider.name,
            "[green]disponible[/green]" if status.available else "[yellow]indisponible[/yellow]",
            names,
        )
    console.print(table)
    return statuses, models_by_provider


def run_analysis(
    mux: CortexMux,
    *,
    csv_path: Path,
    ollama_model: str | None,
) -> None:
    """Analyze the CSV and print structured results without charts or a report file."""
    console.print(Panel(DEMO_ANALYSIS_REQUEST, title="Requête d'analyse de démonstration"))
    response = mux.analyze_data(
        csv_path,
        instruction=DEMO_ANALYSIS_REQUEST,
        engine="auto",
        plan=DEMO_ANALYSIS_PLAN,
        interpretation_provider="ollama" if ollama_model else None,
        interpretation_model=ollama_model,
        verify_calculations=True,
    )
    console.print_json(json.dumps(response.results, ensure_ascii=False, default=str))
    if response.interpretation:
        console.print(Panel(response.interpretation, title="Interprétation Ollama"))
    if response.calculation_verifications:
        table = Table(title="Vérification mathématique")
        table.add_column("Calcul")
        table.add_column("Statut")
        table.add_column("Annoncé")
        table.add_column("Recalculé")
        for verification in response.calculation_verifications:
            table.add_row(
                verification.label,
                verification.status.value,
                str(verification.claimed_result),
                str(verification.expected_result or ""),
            )
        console.print(table)
    if response.warnings:
        console.print("[yellow]Avertissements d'analyse :[/yellow]")
        for warning in response.warnings:
            console.print(f"  - {warning}")


def run_ollama_chat(mux: CortexMux, model: str) -> None:
    """Run a small local Qwen chat request."""
    console.print(Panel(DEMO_CHAT_REQUEST, title=f"Requête Ollama — {model}"))
    response = mux.chat(
        prompt=DEMO_CHAT_REQUEST,
        provider="ollama",
        model=model,
        temperature=0.1,
    )
    console.print(Panel(response.content, title="Réponse Ollama"))


def run_image_generation(
    mux: CortexMux,
    *,
    workflow_path: Path,
    bindings_path: Path,
    checkpoint: str,
    output_dir: Path,
) -> list[str]:
    """Generate an image through the bundled generic ComfyUI workflow."""
    console.print(Panel(DEMO_IMAGE_PROMPT, title="Prompt image de démonstration"))
    response = mux.generate_image(
        prompt=DEMO_IMAGE_PROMPT,
        workflow=workflow_path,
        bindings=load_bindings(bindings_path),
        expected_output_nodes=["9"],
        checkpoint=checkpoint,
        negative_prompt="text, watermark, blurry, low quality, distorted",
        seed=42,
        width=768,
        height=768,
        steps=24,
        guidance=7.0,
        sampler="euler",
        scheduler="normal",
        output_dir=output_dir / "images",
    )
    paths = [image.path for image in response.images]
    console.print("[green]Images générées :[/green]")
    for path in paths:
        console.print(f"  - {path}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the demonstration CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--reset-data", action="store_true")
    parser.add_argument("--ollama-model", default="qwen2.5-coder")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW_PATH)
    parser.add_argument("--bindings", type=Path, default=DEFAULT_BINDINGS_PATH)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "demo")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if Ollama or ComfyUI cannot be tested.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the complete local demonstration and return a process status."""
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    csv_path = create_demo_csv(args.csv.resolve(), overwrite=args.reset_data)
    console.print(f"[green]CSV de démonstration :[/green] {csv_path}")
    failures: list[str] = []

    with CortexMux.from_env() as mux:
        statuses, models_by_provider = list_managed_models(mux)
        ollama_model: str | None = None
        if not args.skip_ollama:
            if statuses["ollama"].available:
                ollama_model = resolve_model(models_by_provider["ollama"], args.ollama_model)
                if ollama_model is None:
                    failures.append(
                        f"Le modèle {args.ollama_model!r} n'est pas installé dans Ollama."
                    )
                else:
                    try:
                        run_ollama_chat(mux, ollama_model)
                    except CortexMuxError as exc:
                        failures.append(f"Requête Ollama échouée : {exc}")
                        ollama_model = None
            else:
                failures.append(f"Ollama indisponible : {statuses['ollama'].message}")

        try:
            run_analysis(
                mux,
                csv_path=csv_path,
                ollama_model=ollama_model,
            )
        except CortexMuxError as exc:
            failures.append(f"Analyse CSV échouée : {exc}")

        if not args.skip_image:
            if not statuses["comfyui"].available:
                failures.append(f"ComfyUI indisponible : {statuses['comfyui'].message}")
            elif not args.workflow.is_file() or not args.bindings.is_file():
                failures.append("Le workflow ou son fichier de bindings est introuvable.")
            else:
                comfy_models = models_by_provider["comfyui"]
                checkpoint = args.checkpoint or next(
                    (model.name for model in comfy_models if model.family == "checkpoints"),
                    None,
                )
                if checkpoint is None:
                    failures.append("Aucun checkpoint ComfyUI n'est disponible.")
                else:
                    try:
                        run_image_generation(
                            mux,
                            workflow_path=args.workflow.resolve(),
                            bindings_path=args.bindings.resolve(),
                            checkpoint=checkpoint,
                            output_dir=output_dir,
                        )
                    except (CortexMuxError, OSError, ValueError) as exc:
                        failures.append(f"Génération d'image échouée : {exc}")

    if failures:
        console.print("\n[yellow]Intégrations non validées :[/yellow]")
        for failure in failures:
            console.print(f"  - {failure}")
    console.print(
        "\n[green]Démonstration terminée.[/green] "
        "Utilisez --strict pour rendre les services locaux obligatoires."
    )
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
