"""Conftest for Pro tests — skip entire directory when Pro deps missing."""

from __future__ import annotations

import pytest

from neural_memory.pro import is_pro_deps_installed

if not is_pro_deps_installed():
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip all Pro tests if deps are missing."""
    if is_pro_deps_installed():
        return
    skip_marker = pytest.mark.skip(reason="Pro dependencies not installed")
    for item in items:
        if "unit/pro" in str(item.fspath) or "unit\\pro" in str(item.fspath):
            item.add_marker(skip_marker)
