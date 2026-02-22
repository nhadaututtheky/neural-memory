"""Device identity for multi-device sync.

Provides stable device identification by persisting a generated device ID
to disk on first access. The device name is derived from the machine hostname.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable device identity record."""

    device_id: str
    device_name: str
    registered_at: datetime


def get_device_id(config_dir: Path) -> str:
    """Return the persistent device ID for this machine.

    Reads the ID from ``{config_dir}/device_id``.  If the file does not
    exist, a new 16-character hex ID is generated, written to that file,
    and returned.

    Args:
        config_dir: Directory where the ``device_id`` file is stored (usually
            the NeuralMemory data directory).

    Returns:
        16-character hex device identifier string.
    """
    id_path = config_dir / "device_id"

    if id_path.exists():
        try:
            existing = id_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError:
            pass  # Fall through to generate a new ID

    # Generate a new device ID
    new_id = uuid4().hex[:16]

    config_dir.mkdir(parents=True, exist_ok=True)
    id_path.write_text(new_id, encoding="utf-8")

    return new_id


def get_device_name() -> str:
    """Return the machine hostname as the device name.

    Returns:
        The network node name of this machine (e.g. ``"my-macbook"``).
        Falls back to ``"unknown"`` if the hostname cannot be determined.
    """
    name = platform.node()
    return name if name else "unknown"


def get_device_info(config_dir: Path) -> DeviceInfo:
    """Return a fully-populated :class:`DeviceInfo` for this machine.

    Args:
        config_dir: Directory where the ``device_id`` file is stored.

    Returns:
        :class:`DeviceInfo` with the stable device ID, hostname, and the
        current UTC time as ``registered_at``.
    """
    return DeviceInfo(
        device_id=get_device_id(config_dir),
        device_name=get_device_name(),
        registered_at=utcnow(),
    )
