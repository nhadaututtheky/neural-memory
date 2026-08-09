"""Optional dependency gates — lazy imports with exact install hints.

Base install must stay usable without optional extras. Call
``require_capability`` only when a feature actually starts.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class MissingCapabilityError(ImportError):
    """Raised when an optional feature is used without its install extra."""

    def __init__(self, feature: str, extra: str, cause: BaseException | None = None) -> None:
        self.feature = feature
        self.extra = extra
        hint = f"pip install 'neural-memory[{extra}]'"
        msg = f"{feature} requires optional dependency group [{extra}]. Install with: {hint}"
        if cause is not None:
            msg = f"{msg} (import error: {type(cause).__name__}: {cause})"
        super().__init__(msg)
        if cause is not None:
            self.__cause__ = cause


# Map feature keys → (module to import, extra name)
_CAPABILITY_MAP: dict[str, tuple[str, str]] = {
    "aiohttp": ("aiohttp", "sync"),
    "sync": ("aiohttp", "sync"),
    "shared_storage": ("aiohttp", "sync"),
    "pydantic": ("pydantic", "server"),
    "server": ("fastapi", "server"),
    "fastapi": ("fastapi", "server"),
    "uvicorn": ("uvicorn", "server"),
    "openclaw": ("pydantic", "server"),
    "networkx": ("networkx", "graph"),  # normally base; used if split later
}


def require_capability(module: str, extra: str, feature: str) -> ModuleType:
    """Import ``module`` or raise ``MissingCapabilityError`` with install hint.

    Args:
        module: Python module name (e.g. ``aiohttp``).
        extra: Optional-dependency extra (e.g. ``sync``).
        feature: Human feature name for the error message.

    Returns:
        Imported module.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingCapabilityError(feature, extra, cause=exc) from exc


def import_optional(module: str, extra: str, feature: str) -> ModuleType:
    """Alias for ``require_capability`` (explicit import style)."""
    return require_capability(module, extra, feature)


def require_named_capability(name: str, *, feature: str | None = None) -> ModuleType:
    """Resolve a named capability from the internal map."""
    if name not in _CAPABILITY_MAP:
        raise MissingCapabilityError(
            feature or name,
            "all",
            cause=KeyError(f"unknown capability {name!r}"),
        )
    module, extra = _CAPABILITY_MAP[name]
    return require_capability(module, extra, feature or name)


def has_capability(module: str) -> bool:
    """Return True if module is importable."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def capability_status() -> dict[str, Any]:
    """Snapshot of optional capability availability (for footprint scripts)."""
    keys = sorted({mod for mod, _extra in _CAPABILITY_MAP.values()})
    return {mod: has_capability(mod) for mod in keys}
