"""Issue #132: regression test for ``nmem recall --limit``.

Bé Mi reported that ``nmem recall "X" --limit 3`` used to work but fails now.
v4.51.4 restores ``--limit`` as an approximate cap (maps to max_tokens).
"""

from __future__ import annotations

import inspect

from neural_memory.cli.commands.memory import recall


class TestRecallLimitOption:
    def test_recall_has_limit_parameter(self) -> None:
        sig = inspect.signature(recall)
        assert "limit" in sig.parameters, "recall() must accept --limit (issue #132)"

    def test_limit_default_is_none(self) -> None:
        """Default must be None so behaviour is unchanged when --limit is omitted."""
        sig = inspect.signature(recall)
        assert sig.parameters["limit"].default is None

    def test_limit_accepts_optional_int(self) -> None:
        """Type must allow int | None so typer parses ``-l 3`` correctly."""
        sig = inspect.signature(recall)
        annotation = sig.parameters["limit"].annotation
        # Annotated[int | None, ...] — the runtime-visible origin is int | None
        assert "int" in str(annotation)
        assert "None" in str(annotation)
