"""CLI configuration management.

This module provides backward-compatible configuration for the CLI.
New code should use unified_config.py for cross-tool compatibility.

Storage locations:
- Legacy: ~/.neural-memory/brains/<name>.json (JSON files)
- New:    ~/.neuralmemory/brains/<name>.db (SQLite database)

The CLI automatically migrates to the new unified config when:
- NEURALMEMORY_DIR environment variable is set, OR
- ~/.neuralmemory/config.toml exists
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def get_default_data_dir() -> Path:
    """Get default data directory for neural-memory.

    Priority:
    1. NEURALMEMORY_DIR environment variable (new unified location)
    2. ~/.neuralmemory/ (if config.toml exists there)
    3. ~/.neural-memory/ (legacy location)
    """
    # Check for env var first (new unified approach)
    env_dir = os.environ.get("NEURALMEMORY_DIR")
    if env_dir:
        return Path(env_dir)

    # Check if new unified config exists
    unified_dir = Path.home() / ".neuralmemory"
    if (unified_dir / "config.toml").exists():
        return unified_dir

    # Fall back to legacy location
    return Path.home() / ".neural-memory"


def use_unified_config() -> bool:
    """Check if we should use the unified config system."""
    env_dir = os.environ.get("NEURALMEMORY_DIR")
    if env_dir:
        return True

    unified_dir = Path.home() / ".neuralmemory"
    return (unified_dir / "config.toml").exists()


@dataclass
class SharedModeConfig:
    """Configuration for shared/remote storage mode."""

    enabled: bool = False
    server_url: str = "http://localhost:8000"
    api_key: str | None = None
    timeout: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "server_url": self.server_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedModeConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            server_url=data.get("server_url", "http://localhost:8000"),
            api_key=data.get("api_key"),
            timeout=data.get("timeout", 30.0),
        )


_BRAIN_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


def _sync_brain_to_toml(data_dir: Path, brain_name: str) -> None:
    """Sync current_brain value into config.toml so MCP server picks it up.

    Uses a regex replacement on the existing TOML file to avoid
    needing a full TOML parser for writing. Only touches the
    ``current_brain`` line.
    """
    toml_path = data_dir / "config.toml"
    if not toml_path.exists():
        return
    if not _BRAIN_NAME_RE.match(brain_name):
        return

    try:
        content = toml_path.read_text(encoding="utf-8")
        updated = re.sub(
            r'^current_brain\s*=\s*"[^"]*"',
            f'current_brain = "{brain_name}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if updated != content:
            toml_path.write_text(updated, encoding="utf-8")
            logger.debug("Synced current_brain=%s to config.toml", brain_name)
    except Exception:
        logger.warning("Failed to sync current_brain to config.toml", exc_info=True)


@dataclass
class CLIConfig:
    """CLI configuration."""

    data_dir: Path = field(default_factory=get_default_data_dir)
    current_brain: str = "default"
    default_depth: int | None = None  # Auto-detect
    default_max_tokens: int = 500
    json_output: bool = False
    shared: SharedModeConfig = field(default_factory=SharedModeConfig)

    @classmethod
    def load(cls, data_dir: Path | None = None) -> CLIConfig:
        """Load configuration from file."""
        if data_dir is None:
            data_dir = get_default_data_dir()

        config_file = data_dir / "config.json"

        if not config_file.exists():
            # Create default config
            config = cls(data_dir=data_dir)
            config.save()
            return config

        with open(config_file, encoding="utf-8") as f:
            data = json.load(f)

        # Parse shared config
        shared_data = data.get("shared", {})
        shared_config = SharedModeConfig.from_dict(shared_data)

        return cls(
            data_dir=data_dir,
            current_brain=data.get("current_brain", "default"),
            default_depth=data.get("default_depth"),
            default_max_tokens=data.get("default_max_tokens", 500),
            json_output=data.get("json_output", False),
            shared=shared_config,
        )

    def save(self) -> None:
        """Save configuration to file.

        Writes to config.json (CLI) and also syncs current_brain
        to config.toml (unified config) so the MCP server stays
        in sync.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.data_dir / "config.json"

        data = {
            "current_brain": self.current_brain,
            "default_depth": self.default_depth,
            "default_max_tokens": self.default_max_tokens,
            "json_output": self.json_output,
            "shared": self.shared.to_dict(),
            "updated_at": utcnow().isoformat(),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Sync current_brain to config.toml so MCP server picks it up
        _sync_brain_to_toml(self.data_dir, self.current_brain)

    @property
    def brains_dir(self) -> Path:
        """Get brains directory."""
        return self.data_dir / "brains"

    @property
    def is_shared_mode(self) -> bool:
        """Check if shared mode is enabled."""
        return self.shared.enabled

    def get_brain_path(self, brain_name: str | None = None) -> Path:
        """Get path to brain data file."""
        name = brain_name or self.current_brain
        return self.brains_dir / f"{name}.json"

    def list_brains(self) -> list[str]:
        """List available brains."""
        if not self.brains_dir.exists():
            return []
        # Check both JSON (legacy) and DB (new) files
        json_brains = [p.stem for p in self.brains_dir.glob("*.json")]
        db_brains = [p.stem for p in self.brains_dir.glob("*.db")]
        return list(set(json_brains + db_brains))

    @property
    def use_sqlite(self) -> bool:
        """Check if SQLite storage should be used (unified mode)."""
        return use_unified_config()

    def get_brain_db_path(self, brain_name: str | None = None) -> Path:
        """Get path to brain SQLite database (unified mode)."""
        name = brain_name or self.current_brain
        return self.brains_dir / f"{name}.db"
