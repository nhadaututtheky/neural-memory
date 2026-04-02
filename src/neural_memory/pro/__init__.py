"""Neural Memory Pro — Advanced features bundled in the main package.

Pro features are included with neural-memory. Activate with a license key.
All Pro dependencies (numpy, hnswlib, msgpack) are bundled in the main install.
"""

from __future__ import annotations

PRO_VERSION = "0.3.0"


def is_pro_deps_installed() -> bool:
    """Return True if all Pro dependencies are installed.

    Since Pro deps are now bundled in main dependencies, this always returns True
    unless the user manually uninstalled numpy/hnswlib/msgpack.
    """
    try:
        import hnswlib as _hnswlib  # noqa: F401
        import msgpack as _msgpack  # noqa: F401
        import numpy as _np  # noqa: F401
    except ImportError:
        return False
    return True


def get_missing_deps() -> list[str]:
    """Return list of missing Pro dependencies (should be empty after install)."""
    missing: list[str] = []
    try:
        import numpy as _np  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import hnswlib as _hnswlib  # noqa: F401
    except ImportError:
        missing.append("hnswlib")
    try:
        import msgpack as _msgpack  # noqa: F401
    except ImportError:
        missing.append("msgpack")
    return missing


PRO_INSTALL_HINT = 'pip install neural-memory  # Pro deps are bundled'

__all__ = [
    "PRO_VERSION",
    "is_pro_deps_installed",
    "get_missing_deps",
    "PRO_INSTALL_HINT",
]
