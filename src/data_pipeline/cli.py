from __future__ import annotations

from pathlib import Path

import typer

from .evidence_alignment import run_evidence_alignment
from .pipeline import run_all as run_pipeline
from .question_cleaning import run_question_cleaning
from .sft_export import export_sft_dataset
from .sft_split import export_sft_splits


app = typer.Typer()


@app.command("run-all")
def run_all() -> None:
    """Run local PDF extraction and regulatory question extraction."""
    result = run_pipeline(Path("."))
    documents = result["documents"]
    pages = result["pages"]
    questions = result["questions"]
    failures = result["failures"]
    typer.echo(f"documents={len(documents)}")
    typer.echo(f"pages={len(pages)}")
    typer.echo(f"questions={len(questions)}")
    typer.echo(f"failures_or_warnings={len(failures)}")


@app.command("clean-questions")
def clean_questions() -> None:
    """Clean and validate extracted regulatory questions."""
    result = run_question_cleaning(Path("."))
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("align-evidence")
def align_evidence() -> None:
    """Retrieve annual-report evidence candidates for clean training questions."""
    result = run_evidence_alignment(Path("."))
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("export-sft")
def export_sft() -> None:
    """Export chat-format SFT data from annual-report evidence candidates."""
    result = export_sft_dataset(Path("."))
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("export-sft-splits")
def export_sft_splits_command() -> None:
    """Export company-level train/validation/test SFT splits."""
    result = export_sft_splits(Path("."))
    for key, value in result.items():
        typer.echo(f"{key}={value}")


@app.command("version", hidden=True)
def version() -> None:
    typer.echo("0.1.0")


if __name__ == "__main__":
    app()
