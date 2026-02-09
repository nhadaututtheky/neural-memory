"""Neural Memory CLI main entry point."""

from __future__ import annotations

import typer

# Main app
app = typer.Typer(
    name="nmem",
    help="Neural Memory - Reflex-based memory for AI agents",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _app_callback(ctx: typer.Context) -> None:
    """Global callback: runs before every command."""
    if ctx.invoked_subcommand is None:
        return

    from neural_memory.cli.update_check import run_update_check_background

    run_update_check_background()


# Register sub-apps (brain, project, shared)
from neural_memory.cli.commands.brain import brain_app  # noqa: E402
from neural_memory.cli.commands.habits import habits_app  # noqa: E402
from neural_memory.cli.commands.project import project_app  # noqa: E402
from neural_memory.cli.commands.shared import shared_app  # noqa: E402
from neural_memory.cli.commands.version import version_app  # noqa: E402

app.add_typer(brain_app, name="brain")
app.add_typer(project_app, name="project")
app.add_typer(shared_app, name="shared")
app.add_typer(habits_app, name="habits")
app.add_typer(version_app, name="version")

# Register top-level commands
from neural_memory.cli.commands import (  # noqa: E402
    codebase,
    info,
    listing,
    memory,
    shortcuts,
    tools,
    train,
    update,
)

memory.register(app)
listing.register(app)
info.register(app)
tools.register(app)
shortcuts.register(app)
codebase.register(app)
train.register(app)
update.register(app)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
