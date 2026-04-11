"""Neural Memory Pro — Advanced features unlocked with a license key.

Pro features are included with neural-memory but require optional dependencies.
Install with: pip install neural-memory[pro]
"""

from __future__ import annotations

PRO_VERSION = "0.3.0"


def is_pro_deps_installed() -> bool:
    """Return True if all Pro dependencies are installed.

    Pro deps (numpy, hnswlib, msgpack) are optional — install with:
    pip install neural-memory[pro]
    """
    try:
        import hnswlib as _hnswlib
        import msgpack as _msgpack
        import numpy as _np
    except ImportError:
        return False
    return True


def get_missing_deps() -> list[str]:
    """Return list of missing Pro dependencies (should be empty after install)."""
    missing: list[str] = []
    try:
        import numpy as _np
    except ImportError:
        missing.append("numpy")
    try:
        import hnswlib as _hnswlib
    except ImportError:
        missing.append("hnswlib")
    try:
        import msgpack as _msgpack
    except ImportError:
        missing.append("msgpack")
    return missing


PRO_INSTALL_HINT = "pip install neural-memory[pro]"

__all__ = [
    "PRO_VERSION",
    "is_pro_deps_installed",
    "get_missing_deps",
    "PRO_INSTALL_HINT",
]
