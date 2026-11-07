"""Typer command-line interface for CortexMux."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from cortexmux import CortexMux
from cortexmux.cli.codex_lab import app as codex_lab_app
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    ConfigurationError,
    CortexMuxError,
    ProviderUnavailableError,
    WorkflowValidationError,
)
from cortexmux.core.serialization import dump_markdown, dump_yaml
from cortexmux.providers.comfyui.workflow import InputBinding, WorkflowDefinition
from cortexmux.schemas.progress import ProgressEvent
from cortexmux.schemas.responses import CortexResponse
from cortexmux.selection import load_qualification_suite, write_qualification_manifest
from cortexmux.version import __version__

app = typer.Typer(help="One interface. Multiple models. Full control.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect provider models.")
image_app = typer.Typer(help="Generate images.")
data_app = typer.Typer(help="Analyze local data.")
config_app = typer.Typer(help="Inspect configuration.")
workflows_app = typer.Typer(help="Inspect and validate ComfyUI workflows.")
app.add_typer(codex_lab_app, name="codex-lab")
app.add_typer(models_app, name="models")
app.add_typer(image_app, name="image")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
app.add_typer(workflows_app, name="workflows")
console = Console()
error_console = Console(stderr=True)
_MAX_INPUT_FILE_BYTES = 1024 * 1024


class OutputFormat(StrEnum):
    """Supported CLI display formats."""

    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    MD = "md"


def _resolve_format(as_json: bool, output_format: OutputFormat | None) -> OutputFormat:
    if as_json and output_format not in (None, OutputFormat.JSON):
        raise typer.BadParameter("--json can only be combined with --format json.")
    if output_format == OutputFormat.MD:
        return OutputFormat.MARKDOWN
    return OutputFormat.JSON if as_json else (output_format or OutputFormat.TEXT)


def _read_text_file(path: Path, option: str) -> str:
    try:
        if not path.is_file():
            raise typer.BadParameter("Expected a regular local file.", param_hint=option)
        with path.open("rb") as source:
            data = source.read(_MAX_INPUT_FILE_BYTES + 1)
    except OSError as exc:
        raise typer.BadParameter("Could not read the local file.", param_hint=option) from exc
    if len(data) > _MAX_INPUT_FILE_BYTES:
        raise typer.BadParameter("Input file exceeds 1 MiB.", param_hint=option)
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter("Input file must be UTF-8.", param_hint=option) from exc
    if not content.strip():
        raise typer.BadParameter("Input file must contain text.", param_hint=option)
    return content


def _resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if (prompt is None) == (prompt_file is None):
        raise typer.BadParameter("Provide exactly one of --prompt or --prompt-file.")
    if prompt_file is not None:
        return _read_text_file(prompt_file, "--prompt-file")
    assert prompt is not None
    return prompt


@app.command()
def version() -> None:
    """Print the installed CortexMux version."""
    console.print(__version__)


@app.command()
def providers(
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """List registered providers."""
    selected_format = _resolve_format(as_json, output_format)
    with CortexMux.from_env() as mux:
        names = [provider.name for provider in mux.registry.list()]
    try:
        _display(names, selected_format)
    except CortexMuxError as exc:
        _fail(exc)


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """Diagnose configuration, optional dependencies, and local providers."""
    selected_format = _resolve_format(as_json, output_format)
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
        if selected_format != OutputFormat.TEXT:
            _display(payload, selected_format)
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
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """List normalized provider models."""
    selected_format = _resolve_format(as_json, output_format)
    try:
        with CortexMux.from_env() as mux:
            models = mux.list_models(provider)
        _display([model.model_dump(mode="json") for model in models], selected_format)
    except CortexMuxError as exc:
        _fail(exc)


@models_app.command("qualify")
def models_qualify(
    suite: Path = typer.Option(..., "--suite", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    config: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """Benchmark model configurations and write a portable recommendation manifest."""
    selected_format = _resolve_format(as_json, output_format or OutputFormat.JSON)
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
            selected_format,
        )
    except (CortexMuxError, OSError, ValueError) as exc:
        _fail(exc)


@app.command()
def chat(
    prompt: str | None = typer.Option(None, "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read a UTF-8 prompt file."
    ),
) -> None:
    """Chat with a configured model."""
    selected_format = _resolve_format(as_json, output_format)
    _run_response(
        "chat",
        selected_format,
        prompt=_resolve_prompt(prompt, prompt_file),
        provider=provider,
        model=model,
    )


@app.command()
def generate(
    prompt: str | None = typer.Option(None, "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read a UTF-8 prompt file."
    ),
) -> None:
    """Generate text with a configured model."""
    selected_format = _resolve_format(as_json, output_format)
    _run_response(
        "text_generation",
        selected_format,
        prompt=_resolve_prompt(prompt, prompt_file),
        provider=provider,
        model=model,
    )


@app.command()
def vision(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    prompt: str | None = typer.Option(None, "--prompt"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read a UTF-8 prompt file."
    ),
) -> None:
    """Analyze an image with a vision-capable model."""
    selected_format = _resolve_format(as_json, output_format)
    _run_response(
        "vision",
        selected_format,
        images=[image],
        prompt=_resolve_prompt(prompt, prompt_file),
        provider=provider,
        model=model,
    )


@app.command()
def embed(
    text: list[str] | None = typer.Option(None, "--text"),
    provider: str = typer.Option("ollama", "--provider"),
    model: str = typer.Option(..., "--model"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    text_file: list[Path] | None = typer.Option(None, "--text-file", help="Read UTF-8 text files."),
) -> None:
    """Generate embeddings."""
    selected_format = _resolve_format(as_json, output_format)
    if bool(text) == bool(text_file):
        raise typer.BadParameter("Provide either --text or --text-file.")
    inputs = [_read_text_file(path, "--text-file") for path in text_file] if text_file else text
    _run_response("embedding", selected_format, inputs=inputs, provider=provider, model=model)


@image_app.command("generate")
def image_generate(
    workflow: str = typer.Option(..., "--workflow"),
    bindings: Path | None = typer.Option(None, "--bindings", exists=True, dir_okay=False),
    prompt: str | None = typer.Option(None, "--prompt"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read a UTF-8 prompt file."
    ),
) -> None:
    """Execute a ComfyUI workflow path or catalog name."""
    selected_format = _resolve_format(as_json, output_format)
    prompt = _resolve_prompt(prompt, prompt_file)
    try:
        binding_data = json.loads(bindings.read_text(encoding="utf-8")) if bindings else {}
        with CortexMux.from_env() as mux:
            response = mux.generate_image(
                prompt,
                workflow=workflow,
                bindings=binding_data,
                checkpoint=checkpoint,
                on_progress=_display_progress
                if progress and selected_format == OutputFormat.TEXT
                else None,
            )
        _response_output(response, selected_format)
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
    output_format: OutputFormat | None = typer.Option(None, "--format"),
    instruction_file: Path | None = typer.Option(
        None, "--instruction-file", help="Read a UTF-8 analysis instruction file."
    ),
) -> None:
    """Analyze a local tabular file deterministically."""
    selected_format = _resolve_format(as_json, output_format)
    if instruction_file is not None:
        if instruction is not None:
            raise typer.BadParameter("--instruction and --instruction-file are mutually exclusive.")
        instruction = _read_text_file(instruction_file, "--instruction-file")
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
        if selected_format != OutputFormat.TEXT:
            _response_output(response, selected_format)
        else:
            console.print(response.interpretation or response.results)
            _display_calculation_verifications(response.calculation_verifications)
    except CortexMuxError as exc:
        _fail(exc)


@config_app.command("show")
def config_show(
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """Show effective redacted configuration."""
    selected_format = _resolve_format(as_json, output_format)
    try:
        config = CortexMuxConfig.load()
        _display(config.model_dump(mode="json"), selected_format)
    except CortexMuxError as exc:
        _fail(exc)


@workflows_app.command("list")
def workflows_list(
    as_json: bool = typer.Option(False, "--json"),
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """List reusable workflows in the configured ComfyUI catalog."""
    selected_format = _resolve_format(as_json, output_format)
    try:
        with CortexMux.from_env() as mux:
            workflows = mux.list_workflows()
        payload = [workflow.model_dump(mode="json") for workflow in workflows]
        if selected_format != OutputFormat.TEXT:
            _display(payload, selected_format)
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
    output_format: OutputFormat | None = typer.Option(None, "--format"),
) -> None:
    """Show one reusable workflow catalog entry."""
    selected_format = _resolve_format(as_json, output_format)
    try:
        with CortexMux.from_env() as mux:
            workflows = mux.list_workflows()
        workflow = next((item for item in workflows if item.name == name), None)
        if workflow is None:
            raise WorkflowValidationError("Workflow is not present in the catalog.", name=name)
        _display(workflow.model_dump(mode="json"), selected_format)
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


def _run_response(task: str, selected_format: OutputFormat, **fields: Any) -> None:
    try:
        with CortexMux.from_env() as mux:
            response = mux.chat(**fields) if task == "chat" else mux.run(task, **fields)
        _response_output(response, selected_format)
    except CortexMuxError as exc:
        _fail(exc)


def _response_output(response: CortexResponse, selected_format: OutputFormat) -> None:
    if selected_format == OutputFormat.MARKDOWN:
        typer.echo(response.to_markdown(), nl=False)
    elif selected_format != OutputFormat.TEXT:
        _display(response.model_dump(mode="json"), selected_format)
    else:
        console.print(getattr(response, "content", response))


def _display(value: Any, selected_format: OutputFormat) -> None:
    if selected_format == OutputFormat.YAML:
        typer.echo(dump_yaml(value), nl=False)
    elif selected_format == OutputFormat.MARKDOWN:
        typer.echo(dump_markdown(value), nl=False)
    elif selected_format == OutputFormat.JSON:
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
    error_console.print(str(exc), style="red", markup=False)
    if isinstance(exc, ConfigurationError):
        raise typer.Exit(code=2)
    if isinstance(exc, ProviderUnavailableError):
        raise typer.Exit(code=3)
    raise typer.Exit(code=4)


if __name__ == "__main__":
    app()
