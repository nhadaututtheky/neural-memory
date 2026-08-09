"""Base dependency contract (Phase 7 REQ-29)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def _parse_base_dependencies() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    # Find [project] dependencies = [ ... ]
    m = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert m is not None, "dependencies block missing in pyproject.toml"
    block = m.group(1)
    deps = re.findall(r'"([^"]+)"', block)
    # Strip version pins to package names
    names = []
    for d in deps:
        name = re.split(r"[<>=!\[]", d, maxsplit=1)[0].strip().lower()
        if name:
            names.append(name)
    return names


class TestBaseDependencyBudget:
    def test_at_most_four_base_deps(self) -> None:
        names = _parse_base_dependencies()
        assert len(names) <= 4, f"base deps exceed budget: {names}"

    def test_required_base_packages_present(self) -> None:
        names = set(_parse_base_dependencies())
        for required in ("aiosqlite", "typer", "rich", "networkx"):
            assert required in names, f"missing base dep {required}"

    def test_optional_moved_out_of_base(self) -> None:
        names = set(_parse_base_dependencies())
        for forbidden in ("aiohttp", "pydantic", "typing_extensions", "fastapi"):
            assert forbidden not in names, f"{forbidden} must not be a base dependency"

    def test_sync_and_server_extras_declared(self) -> None:
        text = PYPROJECT.read_text(encoding="utf-8")
        assert "sync = [" in text or "sync = [" in text
        assert "aiohttp" in text
        assert "server = [" in text
        assert "pydantic" in text
        assert "fastapi" in text


class TestNetworkxStaysBase:
    def test_memory_store_imports_networkx(self) -> None:
        # In-memory path needs networkx; keep as base #4 until proven otherwise
        from neural_memory.storage.memory_store import InMemoryStorage

        assert InMemoryStorage is not None
