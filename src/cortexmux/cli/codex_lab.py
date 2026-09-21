"""Content-safe CLI for importing and validating private offline Codex fixtures."""

from pathlib import Path
from typing import Literal

import typer

from cortexmux.core.exceptions import CortexMuxError
from cortexmux.testing.codex import CodexCaseStore, CodexTestCase, parse_lab_request, read_lab_input

app = typer.Typer(help="Offline Codex fixtures stored outside Git. Never connects to Codex.")


@app.command("init")
def initialize(root: Path | None = typer.Option(None, "--root")) -> None:
    """Create a private store outside any Git repository."""
    try:
        typer.echo(str(CodexCaseStore(root).initialize()))
    except (CortexMuxError, OSError):
        typer.echo(
            "Cannot initialize private Codex storage; use a dedicated directory outside Git.",
            err=True,
        )
        raise typer.Exit(2) from None


@app.command("add")
def add(
    request: Path = typer.Option(..., "--request"),
    response: Path = typer.Option(..., "--response"),
    root: Path | None = typer.Option(None, "--root"),
    origin: Literal["synthetic", "recorded"] = typer.Option("synthetic", "--origin"),
    status: Literal["completed", "interrupted", "failed"] = typer.Option("completed", "--status"),
    expect_valid: bool = typer.Option(True, "--expect-valid/--expect-invalid"),
    expected: Path | None = typer.Option(None, "--expected"),
) -> None:
    """Import a typed request and raw response text; print only the new case identifier."""
    try:
        case = CodexTestCase(
            request=parse_lab_request(read_lab_input(request)),
            content=read_lab_input(response),
            origin=origin,
            status=status,
            expect_valid=expect_valid,
            expected_content=read_lab_input(expected) if expected else None,
        )
        CodexCaseStore(root).save(case)
        typer.echo(case.case_id)
    except (CortexMuxError, ValueError, OSError):
        typer.echo(
            "Cannot import local Codex case. Check request, response and private storage.", err=True
        )
        raise typer.Exit(2) from None


@app.command("list")
def list_cases(root: Path | None = typer.Option(None, "--root")) -> None:
    """List opaque local case identifiers without displaying their contents."""
    try:
        for case_id in CodexCaseStore(root).list_cases():
            typer.echo(case_id)
    except (CortexMuxError, OSError):
        typer.echo("Cannot list private Codex cases.", err=True)
        raise typer.Exit(2) from None


@app.command("validate")
def validate(
    case_id: str | None = typer.Argument(None),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Validate one case or all cases; exit 1 when an expected outcome is not met."""
    try:
        store = CodexCaseStore(root)
        identifiers = [case_id] if case_id else store.list_cases()
        if not identifiers:
            typer.echo("No local Codex cases to validate.", err=True)
            raise typer.Exit(2)
        passed = True
        for identifier in identifiers:
            report = store.validate(identifier)
            typer.echo(report.model_dump_json())
            passed = passed and report.expectation_met
    except (CortexMuxError, OSError):
        typer.echo("Cannot validate private Codex cases.", err=True)
        raise typer.Exit(2) from None
    if not passed:
        raise typer.Exit(1)
