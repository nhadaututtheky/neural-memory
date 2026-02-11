"""Learned workflow habit commands."""

from __future__ import annotations

from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async

habits_app = typer.Typer(help="Learned workflow habit commands")


@habits_app.command("list")
def habits_list(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List learned workflow habits.

    Shows all habits discovered through action sequence mining,
    including step sequences, frequencies, and confidence scores.

    Examples:
        nmem habits list
        nmem habits list --json
    """

    async def _list() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]

            if json_output:
                output_result(
                    {
                        "habits": [
                            {
                                "name": h.summary or "unnamed",
                                "steps": h.metadata.get("_workflow_actions", []),
                                "frequency": h.metadata.get("_habit_frequency", 0),
                                "confidence": h.metadata.get("_habit_confidence", 0.0),
                                "fiber_id": h.id,
                            }
                            for h in habits
                        ],
                        "count": len(habits),
                    },
                    True,
                )
            else:
                if not habits:
                    typer.echo(
                        "No learned habits yet. Use NeuralMemory tools to build action history."
                    )
                    return

                typer.echo(f"Learned habits ({len(habits)}):")
                for h in habits:
                    steps = h.metadata.get("_workflow_actions", [])
                    freq = h.metadata.get("_habit_frequency", 0)
                    conf = h.metadata.get("_habit_confidence", 0.0)
                    typer.echo(f"  {h.summary or 'unnamed'}")
                    typer.echo(f"    Steps: {' → '.join(steps)}")
                    typer.echo(f"    Frequency: {freq}, Confidence: {conf:.2f}")
        finally:
            await storage.close()

    run_async(_list())


@habits_app.command("show")
def habits_show(
    name: Annotated[str, typer.Argument(help="Habit name to show details for")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show details of a specific learned habit.

    Examples:
        nmem habits show recall-edit-test
        nmem habits show recall-edit-test --json
    """

    async def _show() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern") and f.summary == name]

            if not habits:
                typer.echo(f"No habit found with name: {name}")
                raise typer.Exit(code=1)

            habit = habits[0]
            steps = habit.metadata.get("_workflow_actions", [])
            freq = habit.metadata.get("_habit_frequency", 0)
            conf = habit.metadata.get("_habit_confidence", 0.0)

            if json_output:
                output_result(
                    {
                        "name": habit.summary or "unnamed",
                        "steps": steps,
                        "frequency": freq,
                        "confidence": conf,
                        "fiber_id": habit.id,
                        "neuron_count": len(habit.neuron_ids),
                        "synapse_count": len(habit.synapse_ids),
                        "created_at": habit.created_at.isoformat(),
                    },
                    True,
                )
            else:
                typer.echo(f"Habit: {habit.summary or 'unnamed'}")
                typer.echo(f"  Steps: {' → '.join(steps)}")
                typer.echo(f"  Frequency: {freq}")
                typer.echo(f"  Confidence: {conf:.2f}")
                typer.echo(f"  Neurons: {len(habit.neuron_ids)}")
                typer.echo(f"  Synapses: {len(habit.synapse_ids)}")
                typer.echo(f"  Created: {habit.created_at.isoformat()}")
        finally:
            await storage.close()

    run_async(_show())


@habits_app.command("clear")
def habits_clear(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Clear all learned habits.

    Removes all WORKFLOW fibers with _habit_pattern metadata.
    This does not affect action event history.

    Examples:
        nmem habits clear
        nmem habits clear --force
    """

    async def _clear() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]

            if not habits:
                typer.echo("No habits to clear.")
                return

            if not force:
                confirm = typer.confirm(f"Clear {len(habits)} learned habits?")
                if not confirm:
                    typer.echo("Cancelled.")
                    return

            cleared = 0
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1

            typer.echo(f"Cleared {cleared} learned habits.")
        finally:
            await storage.close()

    run_async(_clear())
