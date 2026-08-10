"""Typer command-line interface for CortexMux."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    ConfigurationError,
    CortexMuxError,
    ProviderUnavailableError,
    WorkflowValidationError,
)
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition
from cortexmux.schemas.progress import ProgressEvent
from cortexmux.selection import load_qualification_suite, write_qualification_manifest
from cortexmux.version import __version__

app = typer.Typer(help="One interface. Multiple models. Full control.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect provider models.")
image_app = typer.Typer(help="Generate images.")
data_app = typer.Typer(help="Analyze local data.")
config_app = typer.Typer(help="Inspect configuration.")
workflows_app = typer.Typer(help="Inspect and validate ComfyUI workflows.")
app.add_typer(models_app, name="models")
app.add_typer(image_app, name="image")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
app.add_typer(workflows_app, name="workflows")
console = Console()
error_console = Console(stderr=True)


@app.command()
def version() -> None:
    """Print the installed CortexMux version."""
    console.print(__version__)


@app.command()
def providers(as_json: bool = typer.Option(False, "--json")) -> None:
    """List registered providers."""
    with CortexMux.from_env() as mux:
        names = [provider.name for provider in mux.registry.list()]
    _display(names, as_json)


@app.command()
def doctor(as_json: bool = typer.Option(False, "--json")) -> None:
    """Diagnose configuration, optional dependencies, and local providers."""
    try:
        with CortexMux.from_env() as mux:
            health = mux.health()
            payload = {
                "config_path": str(mux.config.config_path) if mux.config.config_path else None,
                "providers": [status.model_dump(mode="json") for status in health],
                "output_dir": str(mux.config.core.output_dir),
                "output_dir_parent_exists": mux.config.core.output_dir.parent.exists(),
                "optional_dependencies": _optional_status(),
            }
        if as_json:
            _display(payload, True)
            return
        table = Table(title="CortexMux doctor")
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Message")
        for status in health:
            table.add_row(
                status.provider,
                "[green]available[/green]" if status.available else "[yellow]unavailable[/yellow]",
                status.message,
            )
        console.print(table)
        console.print(f"Configuration: {payload['config_path'] or 'built-in defaults'}")
        console.print(f"Output directory: {payload['output_dir']}")
    except CortexMuxError as exc:
        _fail(exc)


@models_app.command("list")
def models_list(
    provider: str = typer.Option(..., "--provider"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List normalized provider models."""
    try:
        with CortexMux.from_env() as mux:
            models = mux.list_models(provider)
        _display([model.model_dump(mode="json") for model in models], as_json)
    except CortexMuxError as exc:
        _fail(exc)


@models_app.command("qualify")
def models_qualify(
    suite: Path = typer.Option(..., "--suite", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Benchmark model configurations and write a portable recommendation manifest."""
    try:
        qualification_suite = load_qualification_suite(suite)
        with CortexMux.from_env(config_path=config) as mux:
            manifest = mux.qualify_models(qualification_suite)
        written = write_qualification_manifest(manifest, output, overwrite=overwrite)
        _display(
            {
                "output": str(written),
                "recommendations": [
                    item.model_dump(mode="json") for item in manifest.recommendations
                ],
            },
            True,
        )
    except (CortexMuxError, OSError, ValueError) as exc:
        _fail(exc)


@app.command()
def chat(
    prompt: str = typer.Option(..., "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Chat with a configured model."""
    _run_response("chat", as_json, prompt=prompt, provider=provider, model=model)


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Generate text with a configured model."""
    _run_response("text_generation", as_json, prompt=prompt, provider=provider, model=model)


@app.command()
def vision(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    prompt: str = typer.Option(..., "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Analyze an image with a vision-capable model."""
    _run_response(
        "vision",
        as_json,
        images=[image],
        prompt=prompt,
        provider=provider,
        model=model,
    )


@app.command()
def embed(
    text: list[str] = typer.Option(..., "--text"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Generate embeddings."""
    _run_response("embedding", as_json, inputs=text, provider=provider, model=model)


@image_app.command("generate")
def image_generate(
    workflow: str = typer.Option(..., "--workflow"),
    bindings: Path | None = typer.Option(None, "--bindings", exists=True, dir_okay=False),
    prompt: str = typer.Option(..., "--prompt"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Execute a ComfyUI workflow path or catalog name."""
    try:
        binding_data = json.loads(bindings.read_text(encoding="utf-8")) if bindings else {}
        with CortexMux.from_env() as mux:
            response = mux.generate_image(
                prompt,
                workflow=workflow,
                bindings=binding_data,
                checkpoint=checkpoint,
                on_progress=_display_progress if progress and not as_json else None,
            )
        _response_output(response, as_json)
    except (CortexMuxError, OSError, json.JSONDecodeError) as exc:
        _fail(exc)


@data_app.command("analyze")
def data_analyze(
    source: Path = typer.Argument(..., exists=True, dir_okay=False),
    instruction: str | None = typer.Option(None, "--instruction"),
    engine: str = typer.Option("auto", "--engine"),
    interpretation_provider: str | None = typer.Option(None, "--provider"),
    interpretation_model: str | None = typer.Option(None, "--model"),
    charts: bool = typer.Option(False, "--charts"),
    require_verified_calculations: bool = typer.Option(False, "--require-verified-calculations"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Analyze a local tabular file deterministically."""
    try:
        with CortexMux.from_env() as mux:
            response = mux.analyze_data(
                source,
                instruction=instruction,
                engine=engine,
                interpretation_provider=interpretation_provider,
                interpretation_model=interpretation_model,
                create_charts=charts,
                require_verified_calculations=require_verified_calculations,
            )
        if as_json:
            _response_output(response, True)
        else:
            console.print(response.interpretation or response.results)
            _display_calculation_verifications(response.calculation_verifications)
    except CortexMuxError as exc:
        _fail(exc)


@config_app.command("show")
def config_show(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show effective redacted configuration."""
    try:
        config = CortexMuxConfig.load()
        _display(config.model_dump(mode="json"), as_json)
    except CortexMuxError as exc:
        _fail(exc)


@workflows_app.command("list")
def workflows_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List reusable workflows in the configured ComfyUI catalog."""
    try:
        with CortexMux.from_env() as mux:
            workflows = mux.list_workflows()
        payload = [workflow.model_dump(mode="json") for workflow in workflows]
        if as_json:
            _display(payload, True)
            return
        table = Table(title="ComfyUI workflow catalog")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Bindings")
        table.add_column("Output nodes")
        for workflow in workflows:
            table.add_row(
                workflow.name,
                workflow.description or "",
                ", ".join(workflow.binding_fields),
                ", ".join(workflow.expected_output_nodes),
            )
        console.print(table)
    except CortexMuxError as exc:
        _fail(exc)


@workflows_app.command("show")
def workflows_show(
    name: str = typer.Argument(...),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show one reusable workflow catalog entry."""
    try:
        with CortexMux.from_env() as mux:
            workflows = mux.list_workflows()
        workflow = next((item for item in workflows if item.name == name), None)
        if workflow is None:
            raise WorkflowValidationError("Workflow is not present in the catalog.", name=name)
        _display(workflow.model_dump(mode="json"), as_json)
    except CortexMuxError as exc:
        _fail(exc)


@workflows_app.command("validate")
def workflows_validate(
    workflow: Path = typer.Argument(..., exists=True, dir_okay=False),
    bindings: Path | None = typer.Option(None, "--bindings", exists=True, dir_okay=False),
) -> None:
    """Validate ComfyUI API-format JSON and optional bindings."""
    try:
        binding_data: dict[str, Any] = {}
        if bindings:
            raw = json.loads(bindings.read_text(encoding="utf-8"))
            binding_data = {key: InputBinding.model_validate(value) for key, value in raw.items()}
        definition = WorkflowDefinition(
            name=workflow.stem, workflow=workflow, input_bindings=binding_data
        )
        graph = definition.load_graph()
        console.print(f"[green]Valid[/green]: {len(graph)} nodes")
    except (CortexMuxError, OSError, ValueError, json.JSONDecodeError) as exc:
        _fail(exc)


def _run_response(task: str, as_json: bool, **fields: Any) -> None:
    try:
        with CortexMux.from_env() as mux:
            response = mux.chat(**fields) if task == "chat" else mux.run(task, **fields)
        _response_output(response, as_json)
    except CortexMuxError as exc:
        _fail(exc)


def _response_output(response: Any, as_json: bool, text_field: str = "content") -> None:
    if as_json:
        _display(response.model_dump(mode="json"), True)
    else:
        console.print(getattr(response, text_field, response))


def _display(value: Any, as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps(value, default=str))
    else:
        console.print(value)


def _display_progress(event: ProgressEvent) -> None:
    percent = f" {event.progress:.0%}" if event.progress is not None else ""
    node = f" node={event.node_id}" if event.node_id else ""
    console.print(f"[cyan]{event.stage.value}[/cyan]{percent}{node}: {event.message or ''}")


def _display_calculation_verifications(verifications: list[Any]) -> None:
    if not verifications:
        return
    table = Table(title="Mathematical verification")
    table.add_column("Claim")
    table.add_column("Status")
    table.add_column("Claimed")
    table.add_column("Expected")
    for verification in verifications:
        table.add_row(
            verification.label,
            verification.status.value,
            str(verification.claimed_result),
            str(verification.expected_result) if verification.expected_result is not None else "",
        )
    console.print(table)


def _optional_status() -> dict[str, bool]:
    import importlib.util

    return {
        package: importlib.util.find_spec(package) is not None
        for package in ("pandas", "polars", "duckdb", "openpyxl", "matplotlib", "websockets")
    }


def _fail(exc: BaseException) -> None:
    error_console.print(f"[red]{exc}[/red]")
    if isinstance(exc, ConfigurationError):
        raise typer.Exit(code=2)
    if isinstance(exc, ProviderUnavailableError):
        raise typer.Exit(code=3)
    raise typer.Exit(code=4)


if __name__ == "__main__":
    app()
