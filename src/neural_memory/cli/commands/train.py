"""Doc-to-brain training command: train expert brains from documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async


def train(
    path: Annotated[str, typer.Argument(help="Directory or file to train from")] = ".",
    domain: Annotated[
        str,
        typer.Option("--domain", "-d", help="Domain tag (e.g., react, kubernetes)"),
    ] = "",
    brain: Annotated[
        str,
        typer.Option("--brain", "-b", help="Target brain name (default: current)"),
    ] = "",
    extensions: Annotated[
        list[str] | None,
        typer.Option("--ext", "-e", help="File extensions (default: .md)"),
    ] = None,
    no_consolidate: Annotated[
        bool,
        typer.Option("--no-consolidate", help="Skip ENRICH consolidation"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Train a brain from documentation files (markdown)."""
    run_async(_train_async(path, domain, brain, extensions, no_consolidate, json_output))


async def _train_async(
    path: str,
    domain: str,
    brain: str,
    extensions: list[str] | None,
    no_consolidate: bool,
    json_output: bool,
) -> None:
    """Async implementation of the train command."""
    from neural_memory.engine.doc_trainer import DocTrainer, TrainingConfig

    config = get_config()
    storage = await get_storage(config, force_sqlite=True)

    target = Path(path).resolve()
    if not target.exists():
        typer.echo(f"Error: Path not found: {target}", err=True)
        raise typer.Exit(code=1)

    brain_data = await storage.get_brain(storage._current_brain_id or "")
    if not brain_data:
        typer.echo("Error: No brain configured", err=True)
        raise typer.Exit(code=1)

    tc = TrainingConfig(
        domain_tag=domain,
        brain_name=brain,
        extensions=tuple(extensions) if extensions else (".md",),
        consolidate=not no_consolidate,
    )

    trainer = DocTrainer(storage, brain_data.config)
    storage.disable_auto_save()

    try:
        if target.is_file():
            typer.echo(f"Training from {target.name}...")
            result = await trainer.train_file(target, tc)
        else:
            typer.echo(f"Training from {target} ({', '.join(tc.extensions)})...")
            result = await trainer.train_directory(target, tc)

        await storage.batch_save()
    finally:
        storage.enable_auto_save()

    if json_output:
        output_result(
            {
                "files_processed": result.files_processed,
                "chunks_encoded": result.chunks_encoded,
                "chunks_skipped": result.chunks_skipped,
                "neurons_created": result.neurons_created,
                "synapses_created": result.synapses_created,
                "hierarchy_synapses": result.hierarchy_synapses,
                "enrichment_synapses": result.enrichment_synapses,
                "brain_name": result.brain_name,
            }
        )
    else:
        typer.echo(f"Files processed:      {result.files_processed}")
        typer.echo(f"Chunks encoded:       {result.chunks_encoded}")
        if result.chunks_skipped:
            typer.echo(f"Chunks skipped:       {result.chunks_skipped}")
        typer.echo(f"Neurons created:      {result.neurons_created}")
        typer.echo(f"Synapses created:     {result.synapses_created}")
        typer.echo(f"Hierarchy synapses:   {result.hierarchy_synapses}")
        typer.echo(f"Enrichment synapses:  {result.enrichment_synapses}")
        typer.echo(f"Brain:                {result.brain_name}")


def register(app: typer.Typer) -> None:
    """Register the train command with the CLI app."""
    app.command(name="train")(train)
