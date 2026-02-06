"""Unified configuration for NeuralMemory across all tools.

This module provides a single configuration system that works across:
- CLI (nmem command)
- MCP server (Claude Code, Cursor, AntiGravity)
- REST API server
- Any future integrations

Configuration is stored in ~/.neuralmemory/config.toml
Brain data is stored in ~/.neuralmemory/brains/<name>.db (SQLite)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Valid brain name: alphanumeric, hyphens, underscores, dots (no path separators)
_BRAIN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# Try to import tomllib (Python 3.11+) or fallback
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


def get_neuralmemory_dir() -> Path:
    """Get NeuralMemory data directory.

    Priority:
    1. NEURALMEMORY_DIR environment variable
    2. ~/.neuralmemory/
    """
    env_dir = os.environ.get("NEURALMEMORY_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".neuralmemory"


def get_default_brain() -> str:
    """Get default brain name.

    Priority:
    1. NEURALMEMORY_BRAIN environment variable
    2. "default"
    """
    return os.environ.get("NEURALMEMORY_BRAIN", "default")


@dataclass
class AutoConfig:
    """Auto-capture configuration for MCP server."""

    enabled: bool = True
    capture_decisions: bool = True
    capture_errors: bool = True
    capture_todos: bool = True
    capture_facts: bool = True
    capture_insights: bool = True
    min_confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capture_decisions": self.capture_decisions,
            "capture_errors": self.capture_errors,
            "capture_todos": self.capture_todos,
            "capture_facts": self.capture_facts,
            "capture_insights": self.capture_insights,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoConfig:
        return cls(
            enabled=data.get("enabled", True),
            capture_decisions=data.get("capture_decisions", True),
            capture_errors=data.get("capture_errors", True),
            capture_todos=data.get("capture_todos", True),
            capture_facts=data.get("capture_facts", True),
            capture_insights=data.get("capture_insights", True),
            min_confidence=data.get("min_confidence", 0.7),
        )


@dataclass
class BrainSettings:
    """Settings for brain behavior."""

    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "reinforcement_delta": self.reinforcement_delta,
            "activation_threshold": self.activation_threshold,
            "max_spread_hops": self.max_spread_hops,
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainSettings:
        return cls(
            decay_rate=data.get("decay_rate", 0.1),
            reinforcement_delta=data.get("reinforcement_delta", 0.05),
            activation_threshold=data.get("activation_threshold", 0.2),
            max_spread_hops=data.get("max_spread_hops", 4),
            max_context_tokens=data.get("max_context_tokens", 1500),
        )


@dataclass
class UnifiedConfig:
    """Unified configuration for NeuralMemory.

    This configuration is shared across all tools:
    - CLI: nmem commands
    - MCP: Claude Code, Cursor, AntiGravity
    - API: REST server

    Storage location: ~/.neuralmemory/config.toml
    Brain location: ~/.neuralmemory/brains/<name>.db
    """

    # Base directory for all NeuralMemory data
    data_dir: Path = field(default_factory=get_neuralmemory_dir)

    # Current active brain
    current_brain: str = field(default_factory=get_default_brain)

    # Brain settings
    brain: BrainSettings = field(default_factory=BrainSettings)

    # Auto-capture settings for MCP
    auto: AutoConfig = field(default_factory=AutoConfig)

    # CLI preferences
    json_output: bool = False
    default_depth: int | None = None
    default_max_tokens: int = 500

    # Metadata
    version: str = "1.0"
    updated_at: datetime | None = None

    @classmethod
    def load(cls, config_path: Path | None = None) -> UnifiedConfig:
        """Load configuration from file, or create default if doesn't exist."""
        if config_path is None:
            data_dir = get_neuralmemory_dir()
            config_path = data_dir / "config.toml"
        else:
            data_dir = config_path.parent

        if not config_path.exists():
            # Create default config
            config = cls(data_dir=data_dir)
            config.save()
            return config

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        return cls(
            data_dir=data_dir,
            current_brain=data.get("current_brain", get_default_brain()),
            brain=BrainSettings.from_dict(data.get("brain", {})),
            auto=AutoConfig.from_dict(data.get("auto", {})),
            json_output=data.get("cli", {}).get("json_output", False),
            default_depth=data.get("cli", {}).get("default_depth"),
            default_max_tokens=data.get("cli", {}).get("default_max_tokens", 500),
            version=data.get("version", "1.0"),
        )

    def save(self) -> None:
        """Save configuration to TOML file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.data_dir / "config.toml"

        # Build TOML content manually (no toml write dependency)
        lines = [
            "# NeuralMemory Configuration",
            "# This config is shared by CLI, MCP server, and all integrations",
            "",
            f'version = "{self.version}"',
            f'current_brain = "{self.current_brain}"',
            "",
            "# Brain behavior settings",
            "[brain]",
            f"decay_rate = {self.brain.decay_rate}",
            f"reinforcement_delta = {self.brain.reinforcement_delta}",
            f"activation_threshold = {self.brain.activation_threshold}",
            f"max_spread_hops = {self.brain.max_spread_hops}",
            f"max_context_tokens = {self.brain.max_context_tokens}",
            "",
            "# Auto-capture settings for MCP server",
            "[auto]",
            f"enabled = {'true' if self.auto.enabled else 'false'}",
            f"capture_decisions = {'true' if self.auto.capture_decisions else 'false'}",
            f"capture_errors = {'true' if self.auto.capture_errors else 'false'}",
            f"capture_todos = {'true' if self.auto.capture_todos else 'false'}",
            f"capture_facts = {'true' if self.auto.capture_facts else 'false'}",
            f"capture_insights = {'true' if self.auto.capture_insights else 'false'}",
            f"min_confidence = {self.auto.min_confidence}",
            "",
            "# CLI preferences",
            "[cli]",
            f"json_output = {'true' if self.json_output else 'false'}",
            f"default_max_tokens = {self.default_max_tokens}",
        ]

        if self.default_depth is not None:
            lines.append(f"default_depth = {self.default_depth}")

        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @property
    def brains_dir(self) -> Path:
        """Get directory where brain databases are stored."""
        return self.data_dir / "brains"

    @property
    def config_path(self) -> Path:
        """Get path to config file."""
        return self.data_dir / "config.toml"

    def get_brain_db_path(self, brain_name: str | None = None) -> Path:
        """Get path to brain SQLite database.

        Args:
            brain_name: Brain name, or use current_brain if None

        Returns:
            Path to SQLite database file

        Raises:
            ValueError: If brain name contains invalid characters
        """
        name = brain_name or self.current_brain
        if not _BRAIN_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid brain name '{name}': must contain only "
                "alphanumeric characters, hyphens, underscores, or dots"
            )
        db_path = (self.brains_dir / f"{name}.db").resolve()
        if not db_path.is_relative_to(self.brains_dir.resolve()):
            raise ValueError(f"Invalid brain name '{name}': path traversal detected")
        return db_path

    def list_brains(self) -> list[str]:
        """List available brains (by database files)."""
        if not self.brains_dir.exists():
            return []
        return [p.stem for p in self.brains_dir.glob("*.db")]

    def switch_brain(self, brain_name: str) -> None:
        """Switch to a different brain and save config."""
        self.current_brain = brain_name
        self.save()


# Singleton instance for easy access
_config: UnifiedConfig | None = None


def get_config(reload: bool = False) -> UnifiedConfig:
    """Get the unified configuration (singleton).

    Args:
        reload: Force reload from disk

    Returns:
        UnifiedConfig instance
    """
    global _config
    if _config is None or reload:
        _config = UnifiedConfig.load()
    return _config


async def get_shared_storage(brain_name: str | None = None):
    """Get SQLite storage for shared brain access.

    This is the main entry point for getting storage that works
    across CLI, MCP, and other tools.

    Args:
        brain_name: Brain name, or use config's current_brain if None

    Returns:
        SQLiteStorage instance, initialized and ready to use
    """
    from neural_memory.core.brain import Brain
    from neural_memory.storage.sqlite_store import SQLiteStorage

    config = get_config()
    db_path = config.get_brain_db_path(brain_name)

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create and initialize storage
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    # Create brain if it doesn't exist
    name = brain_name or config.current_brain
    brain = await storage.get_brain(name)

    if brain is None:
        from neural_memory.core.brain import BrainConfig

        brain_config = BrainConfig(
            decay_rate=config.brain.decay_rate,
            reinforcement_delta=config.brain.reinforcement_delta,
            activation_threshold=config.brain.activation_threshold,
            max_spread_hops=config.brain.max_spread_hops,
            max_context_tokens=config.brain.max_context_tokens,
        )
        brain = Brain.create(name=name, config=brain_config, brain_id=name)
        await storage.save_brain(brain)

    storage.set_brain(brain.id)
    return storage
