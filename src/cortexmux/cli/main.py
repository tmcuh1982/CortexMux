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
)
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition
from cortexmux.version import __version__

app = typer.Typer(help="One interface. Multiple models. Full control.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect provider models.")
image_app = typer.Typer(help="Generate images.")
data_app = typer.Typer(help="Analyze local data.")
config_app = typer.Typer(help="Inspect configuration.")
workflows_app = typer.Typer(help="Validate ComfyUI workflows.")
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
    workflow: Path = typer.Option(..., "--workflow", exists=True, dir_okay=False),
    bindings: Path = typer.Option(..., "--bindings", exists=True, dir_okay=False),
    prompt: str = typer.Option(..., "--prompt"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Execute a ComfyUI API-format workflow."""
    try:
        binding_data = json.loads(bindings.read_text(encoding="utf-8"))
        with CortexMux.from_env() as mux:
            response = mux.generate_image(
                prompt,
                workflow=workflow,
                bindings=binding_data,
                checkpoint=checkpoint,
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
            )
        _response_output(response, as_json, text_field="report_markdown")
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
            response = mux.run(task, **fields)
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
