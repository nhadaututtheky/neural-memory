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

import json
import logging
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage

# Valid brain name: alphanumeric, hyphens, underscores, dots (no path separators)
_BRAIN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# Valid sync identifier: alphanumeric, hyphens, underscores, dots, @ (for emails)
_SYNC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.@]*$")
_SYNC_ID_MAX_LEN = 128


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
    1. NEURALMEMORY_BRAIN environment variable (validated)
    2. "default"
    """
    name = os.environ.get("NEURALMEMORY_BRAIN", "default")
    if not _BRAIN_NAME_PATTERN.match(name):
        return "default"
    return name


@dataclass
class AutoConfig:
    """Auto-capture configuration for MCP server."""

    enabled: bool = True
    capture_decisions: bool = True
    capture_errors: bool = True
    capture_todos: bool = True
    capture_facts: bool = True
    capture_insights: bool = True
    capture_preferences: bool = True
    min_confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capture_decisions": self.capture_decisions,
            "capture_errors": self.capture_errors,
            "capture_todos": self.capture_todos,
            "capture_facts": self.capture_facts,
            "capture_insights": self.capture_insights,
            "capture_preferences": self.capture_preferences,
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
            capture_preferences=data.get("capture_preferences", True),
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
    freshness_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "reinforcement_delta": self.reinforcement_delta,
            "activation_threshold": self.activation_threshold,
            "max_spread_hops": self.max_spread_hops,
            "max_context_tokens": self.max_context_tokens,
            "freshness_weight": self.freshness_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainSettings:
        return cls(
            decay_rate=data.get("decay_rate", 0.1),
            reinforcement_delta=data.get("reinforcement_delta", 0.05),
            activation_threshold=data.get("activation_threshold", 0.2),
            max_spread_hops=data.get("max_spread_hops", 4),
            max_context_tokens=data.get("max_context_tokens", 1500),
            freshness_weight=data.get("freshness_weight", 0.0),
        )


@dataclass
class EternalConfig:
    """Eternal context auto-save configuration."""

    enabled: bool = True
    notifications: bool = True
    snapshot_retention_days: int = 7
    auto_save_interval: int = 15
    context_warning_threshold: float = 0.8
    max_context_tokens: int = 128_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notifications": self.notifications,
            "snapshot_retention_days": self.snapshot_retention_days,
            "auto_save_interval": self.auto_save_interval,
            "context_warning_threshold": self.context_warning_threshold,
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EternalConfig:
        return cls(
            enabled=data.get("enabled", True),
            notifications=data.get("notifications", True),
            snapshot_retention_days=data.get("snapshot_retention_days", 7),
            auto_save_interval=data.get("auto_save_interval", 15),
            context_warning_threshold=data.get("context_warning_threshold", 0.8),
            max_context_tokens=data.get("max_context_tokens", 128_000),
        )


@dataclass(frozen=True)
class MaintenanceConfig:
    """Proactive brain maintenance configuration.

    Controls the health pulse system that piggybacks on remember/recall
    operations to detect brain degradation and surface maintenance hints.
    """

    enabled: bool = True
    check_interval: int = 25
    fiber_warn_threshold: int = 500
    neuron_warn_threshold: int = 2000
    synapse_warn_threshold: int = 5000
    orphan_ratio_threshold: float = 0.25
    expired_memory_warn_threshold: int = 10
    stale_fiber_ratio_threshold: float = 0.3
    stale_fiber_days: int = 90
    auto_consolidate: bool = True
    auto_consolidate_strategies: tuple[str, ...] = ("prune", "merge")
    consolidate_cooldown_minutes: int = 60
    dream_cooldown_hours: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "check_interval": self.check_interval,
            "fiber_warn_threshold": self.fiber_warn_threshold,
            "neuron_warn_threshold": self.neuron_warn_threshold,
            "synapse_warn_threshold": self.synapse_warn_threshold,
            "orphan_ratio_threshold": self.orphan_ratio_threshold,
            "expired_memory_warn_threshold": self.expired_memory_warn_threshold,
            "stale_fiber_ratio_threshold": self.stale_fiber_ratio_threshold,
            "stale_fiber_days": self.stale_fiber_days,
            "auto_consolidate": self.auto_consolidate,
            "auto_consolidate_strategies": list(self.auto_consolidate_strategies),
            "consolidate_cooldown_minutes": self.consolidate_cooldown_minutes,
            "dream_cooldown_hours": self.dream_cooldown_hours,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaintenanceConfig:
        strategies = data.get("auto_consolidate_strategies", ("prune", "merge"))
        if isinstance(strategies, list):
            strategies = tuple(strategies)
        return cls(
            enabled=data.get("enabled", True),
            check_interval=data.get("check_interval", 25),
            fiber_warn_threshold=data.get("fiber_warn_threshold", 500),
            neuron_warn_threshold=data.get("neuron_warn_threshold", 2000),
            synapse_warn_threshold=data.get("synapse_warn_threshold", 5000),
            orphan_ratio_threshold=data.get("orphan_ratio_threshold", 0.25),
            expired_memory_warn_threshold=data.get("expired_memory_warn_threshold", 10),
            stale_fiber_ratio_threshold=data.get("stale_fiber_ratio_threshold", 0.3),
            stale_fiber_days=data.get("stale_fiber_days", 90),
            auto_consolidate=data.get("auto_consolidate", True),
            auto_consolidate_strategies=strategies,
            consolidate_cooldown_minutes=data.get("consolidate_cooldown_minutes", 60),
            dream_cooldown_hours=data.get("dream_cooldown_hours", 24),
        )


@dataclass(frozen=True)
class ConflictConfig:
    """Auto-conflict resolution configuration.

    Controls whether trivial conflicts are automatically resolved
    instead of requiring manual intervention.
    """

    auto_resolve_trivial: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_resolve_trivial": self.auto_resolve_trivial,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConflictConfig:
        return cls(
            auto_resolve_trivial=data.get("auto_resolve_trivial", True),
        )


@dataclass(frozen=True)
class SafetyConfig:
    """Safety and auto-redaction configuration.

    Controls automatic redaction of high-severity sensitive content
    instead of blocking the entire operation.
    """

    auto_redact_min_severity: int = 3  # Auto-redact severity 3+ by default

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_redact_min_severity": self.auto_redact_min_severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        severity = data.get("auto_redact_min_severity", 3)
        try:
            severity = max(1, min(int(severity), 3))
        except (ValueError, TypeError):
            severity = 3
        return cls(auto_redact_min_severity=severity)


@dataclass(frozen=True)
class DedupSettings:
    """LLM-powered deduplication settings.

    Controls the 3-tier dedup pipeline: SimHash -> Embedding -> LLM.
    All off by default to preserve zero-LLM core.
    """

    enabled: bool = False
    simhash_threshold: int = 10
    embedding_threshold: float = 0.85
    embedding_ambiguous_low: float = 0.75
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_max_pairs_per_encode: int = 3
    merge_strategy: str = "keep_newer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "simhash_threshold": self.simhash_threshold,
            "embedding_threshold": self.embedding_threshold,
            "embedding_ambiguous_low": self.embedding_ambiguous_low,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_pairs_per_encode": self.llm_max_pairs_per_encode,
            "merge_strategy": self.merge_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DedupSettings:
        return cls(
            enabled=bool(data.get("enabled", False)),
            simhash_threshold=int(data.get("simhash_threshold", 10)),
            embedding_threshold=float(data.get("embedding_threshold", 0.85)),
            embedding_ambiguous_low=float(data.get("embedding_ambiguous_low", 0.75)),
            llm_enabled=bool(data.get("llm_enabled", False)),
            llm_provider=str(data.get("llm_provider", "none")),
            llm_model=str(data.get("llm_model", "")),
            llm_max_pairs_per_encode=int(data.get("llm_max_pairs_per_encode", 3)),
            merge_strategy=str(data.get("merge_strategy", "keep_newer")),
        )


@dataclass(frozen=True)
class Mem0SyncConfig:
    """Auto-sync configuration for Mem0 integration.

    When enabled, the MCP server auto-detects Mem0 (via MEM0_API_KEY env var
    or self_hosted flag) and syncs memories in background on startup.
    """

    enabled: bool = True
    self_hosted: bool = False
    user_id: str = ""
    agent_id: str = ""
    cooldown_minutes: int = 60
    sync_on_startup: bool = True
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "self_hosted": self.self_hosted,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "cooldown_minutes": self.cooldown_minutes,
            "sync_on_startup": self.sync_on_startup,
        }
        if self.limit is not None:
            result["limit"] = self.limit
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mem0SyncConfig:
        user_id = _sanitize_sync_id(data.get("user_id", ""))
        agent_id = _sanitize_sync_id(data.get("agent_id", ""))
        try:
            cooldown = max(1, min(int(data.get("cooldown_minutes", 60)), 1440))
        except (ValueError, TypeError):
            cooldown = 60
        raw_limit = data.get("limit")
        try:
            limit = max(1, min(int(raw_limit), 100_000)) if raw_limit is not None else None
        except (ValueError, TypeError):
            limit = None
        return cls(
            enabled=bool(data.get("enabled", True)),
            self_hosted=bool(data.get("self_hosted", False)),
            user_id=user_id,
            agent_id=agent_id,
            cooldown_minutes=cooldown,
            sync_on_startup=bool(data.get("sync_on_startup", True)),
            limit=limit,
        )


def _sanitize_sync_id(value: str) -> str:
    """Sanitize a sync identifier (user_id, agent_id).

    Strips whitespace, truncates to max length, and validates against
    allowed characters to prevent TOML injection and log injection.
    Returns empty string if invalid.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:_SYNC_ID_MAX_LEN]
    if not _SYNC_ID_PATTERN.match(cleaned):
        return ""
    return cleaned


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

    # Eternal context settings
    eternal: EternalConfig = field(default_factory=EternalConfig)

    # Proactive maintenance settings
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)

    # Conflict resolution settings
    conflict: ConflictConfig = field(default_factory=ConflictConfig)

    # Safety settings
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Dedup settings
    dedup: DedupSettings = field(default_factory=DedupSettings)

    # Mem0 auto-sync settings
    mem0_sync: Mem0SyncConfig = field(default_factory=Mem0SyncConfig)

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
            # Migrate current_brain from legacy config.json if available
            legacy_brain = _read_legacy_brain(data_dir)
            config = cls(
                data_dir=data_dir,
                current_brain=legacy_brain or get_default_brain(),
            )
            config.save()
            if legacy_brain:
                _logger = logging.getLogger(__name__)
                _logger.info(
                    "Migrated current_brain=%s from legacy config.json to config.toml",
                    legacy_brain,
                )
            return config

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        return cls(
            data_dir=data_dir,
            current_brain=data.get("current_brain", get_default_brain()),
            brain=BrainSettings.from_dict(data.get("brain", {})),
            auto=AutoConfig.from_dict(data.get("auto", {})),
            eternal=EternalConfig.from_dict(data.get("eternal", {})),
            maintenance=MaintenanceConfig.from_dict(data.get("maintenance", {})),
            conflict=ConflictConfig.from_dict(data.get("conflict", {})),
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            dedup=DedupSettings.from_dict(data.get("dedup", {})),
            mem0_sync=Mem0SyncConfig.from_dict(data.get("mem0_sync", {})),
            json_output=data.get("cli", {}).get("json_output", False),
            default_depth=data.get("cli", {}).get("default_depth"),
            default_max_tokens=data.get("cli", {}).get("default_max_tokens", 500),
            version=data.get("version", "1.0"),
        )

    def save(self) -> None:
        """Save configuration to TOML file (atomic write via temp+rename)."""
        import tempfile

        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.data_dir / "config.toml"

        # Validate brain name before writing to prevent TOML injection
        if not _BRAIN_NAME_PATTERN.match(self.current_brain):
            raise ValueError("Invalid brain name for config save")

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
            f"freshness_weight = {self.brain.freshness_weight}",
            "",
            "# Auto-capture settings for MCP server",
            "[auto]",
            f"enabled = {'true' if self.auto.enabled else 'false'}",
            f"capture_decisions = {'true' if self.auto.capture_decisions else 'false'}",
            f"capture_errors = {'true' if self.auto.capture_errors else 'false'}",
            f"capture_todos = {'true' if self.auto.capture_todos else 'false'}",
            f"capture_facts = {'true' if self.auto.capture_facts else 'false'}",
            f"capture_insights = {'true' if self.auto.capture_insights else 'false'}",
            f"capture_preferences = {'true' if self.auto.capture_preferences else 'false'}",
            f"min_confidence = {self.auto.min_confidence}",
            "",
            "# Eternal context settings",
            "[eternal]",
            f"enabled = {'true' if self.eternal.enabled else 'false'}",
            f"notifications = {'true' if self.eternal.notifications else 'false'}",
            f"snapshot_retention_days = {self.eternal.snapshot_retention_days}",
            f"auto_save_interval = {self.eternal.auto_save_interval}",
            f"context_warning_threshold = {self.eternal.context_warning_threshold}",
            f"max_context_tokens = {self.eternal.max_context_tokens}",
            "",
            "# Proactive maintenance settings",
            "[maintenance]",
            f"enabled = {'true' if self.maintenance.enabled else 'false'}",
            f"check_interval = {self.maintenance.check_interval}",
            f"fiber_warn_threshold = {self.maintenance.fiber_warn_threshold}",
            f"neuron_warn_threshold = {self.maintenance.neuron_warn_threshold}",
            f"synapse_warn_threshold = {self.maintenance.synapse_warn_threshold}",
            f"orphan_ratio_threshold = {self.maintenance.orphan_ratio_threshold}",
            f"expired_memory_warn_threshold = {self.maintenance.expired_memory_warn_threshold}",
            f"stale_fiber_ratio_threshold = {self.maintenance.stale_fiber_ratio_threshold}",
            f"stale_fiber_days = {self.maintenance.stale_fiber_days}",
            f"auto_consolidate = {'true' if self.maintenance.auto_consolidate else 'false'}",
            f"auto_consolidate_strategies = {json.dumps(list(self.maintenance.auto_consolidate_strategies))}",
            f"consolidate_cooldown_minutes = {self.maintenance.consolidate_cooldown_minutes}",
            f"dream_cooldown_hours = {self.maintenance.dream_cooldown_hours}",
            "",
            "# Conflict resolution settings",
            "[conflict]",
            f"auto_resolve_trivial = {'true' if self.conflict.auto_resolve_trivial else 'false'}",
            "",
            "# Safety settings",
            "[safety]",
            f"auto_redact_min_severity = {self.safety.auto_redact_min_severity}",
            "",
            "# Dedup settings",
            "[dedup]",
            f"enabled = {'true' if self.dedup.enabled else 'false'}",
            f"simhash_threshold = {self.dedup.simhash_threshold}",
            f"embedding_threshold = {self.dedup.embedding_threshold}",
            f"embedding_ambiguous_low = {self.dedup.embedding_ambiguous_low}",
            f"llm_enabled = {'true' if self.dedup.llm_enabled else 'false'}",
            f'llm_provider = "{self.dedup.llm_provider}"',
            f'llm_model = "{self.dedup.llm_model}"',
            f"llm_max_pairs_per_encode = {self.dedup.llm_max_pairs_per_encode}",
            f'merge_strategy = "{self.dedup.merge_strategy}"',
            "",
            "# Mem0 auto-sync settings",
            "[mem0_sync]",
            f"enabled = {'true' if self.mem0_sync.enabled else 'false'}",
            f"self_hosted = {'true' if self.mem0_sync.self_hosted else 'false'}",
            f'user_id = "{_sanitize_sync_id(self.mem0_sync.user_id)}"',
            f'agent_id = "{_sanitize_sync_id(self.mem0_sync.agent_id)}"',
            f"cooldown_minutes = {self.mem0_sync.cooldown_minutes}",
            f"sync_on_startup = {'true' if self.mem0_sync.sync_on_startup else 'false'}",
        ]

        if self.mem0_sync.limit is not None:
            lines.append(f"limit = {self.mem0_sync.limit}")

        lines += [
            "",
            "# CLI preferences",
            "[cli]",
            f"json_output = {'true' if self.json_output else 'false'}",
            f"default_max_tokens = {self.default_max_tokens}",
        ]

        if self.default_depth is not None:
            lines.append(f"default_depth = {self.default_depth}")

        # Atomic write: write to temp file, then rename
        content = "\n".join(lines) + "\n"
        fd, tmp_path = tempfile.mkstemp(dir=str(self.data_dir), suffix=".toml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(config_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

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
                "Invalid brain name: must contain only "
                "alphanumeric characters, hyphens, underscores, or dots"
            )
        db_path = (self.brains_dir / f"{name}.db").resolve()
        if not db_path.is_relative_to(self.brains_dir.resolve()):
            raise ValueError("Invalid brain name: path traversal detected")
        return db_path

    def list_brains(self) -> list[str]:
        """List available brains (by database files)."""
        if not self.brains_dir.exists():
            return []
        return sorted(p.stem for p in self.brains_dir.glob("*.db"))

    def switch_brain(self, brain_name: str) -> None:
        """Switch to a different brain and save config."""
        if not _BRAIN_NAME_PATTERN.match(brain_name):
            raise ValueError(
                "Invalid brain name: must contain only "
                "alphanumeric characters, hyphens, underscores, or dots"
            )
        self.current_brain = brain_name
        self.save()


# Singleton instance for easy access
_config: UnifiedConfig | None = None

# Cached storage instances keyed by db_path string
_storage_cache: dict[str, SQLiteStorage] = {}


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


def _read_legacy_brain(data_dir: Path) -> str | None:
    """Read current_brain from legacy config.json during first-time migration.

    Checks both the given data_dir and the legacy ~/.neural-memory/ location
    for an existing config.json with a non-default brain selection.

    Returns:
        The brain name string, or ``None`` if no legacy config found
        or it uses the default brain.
    """
    # Check locations in priority order
    candidates = [data_dir / "config.json"]
    legacy_dir = Path.home() / ".neural-memory"
    if legacy_dir != data_dir:
        candidates.append(legacy_dir / "config.json")

    for config_file in candidates:
        if not config_file.is_file():
            continue
        try:
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("current_brain")
            if isinstance(name, str) and name != "default" and _BRAIN_NAME_PATTERN.match(name):
                return name
        except Exception:
            continue
    return None


def _read_current_brain_from_toml() -> str | None:
    """Read just the current_brain value from config.toml on disk.

    This is a lightweight read used by ``get_shared_storage`` to detect
    brain switches made by the CLI (which writes to config.toml via
    ``_sync_brain_to_toml``).  It avoids a full config reload.

    Returns:
        The current_brain string, or ``None`` if the file is missing
        or cannot be parsed.
    """
    toml_path = get_neuralmemory_dir() / "config.toml"
    if not toml_path.exists():
        return None
    try:
        import tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        name = data.get("current_brain")
        if isinstance(name, str) and _BRAIN_NAME_PATTERN.match(name):
            return name
    except Exception:
        pass
    return None


_MIN_LEGACY_DB_BYTES = 8192  # skip empty-schema-only files


def _migrate_legacy_db(config: UnifiedConfig, brain_name: str | None) -> None:
    """Auto-migrate flat-layout default.db → brains/default.db.

    Before the brains/ directory was introduced, NeuralMemory stored its
    database at ``~/.neuralmemory/default.db``.  The new layout puts each
    brain in ``~/.neuralmemory/brains/<name>.db``.

    This function copies the old file into the new location **once**, so
    users upgrading from older versions keep their data.  The old file is
    preserved as a backup.

    Only the ``"default"`` brain is eligible — it was the only brain that
    existed in the flat layout.
    """
    name = brain_name or config.current_brain
    if name != "default":
        return

    old_path = config.data_dir / "default.db"
    new_path = config.brains_dir / "default.db"

    if new_path.exists():
        return
    if not old_path.is_file():
        return
    if old_path.stat().st_size < _MIN_LEGACY_DB_BYTES:
        return

    logger = logging.getLogger(__name__)
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)

        # Also copy WAL/SHM if present so SQLite sees a consistent state.
        for suffix in ("-wal", "-shm"):
            wal = old_path.with_name(old_path.name + suffix)
            if wal.is_file():
                shutil.copy2(wal, new_path.with_name(new_path.name + suffix))

        logger.info(
            "Migrated legacy database: %s → %s (%d bytes)",
            old_path,
            new_path,
            new_path.stat().st_size,
        )
    except Exception:
        logger.warning(
            "Failed to migrate legacy database %s — data is still safe in the original location",
            old_path,
            exc_info=True,
        )


async def get_shared_storage(brain_name: str | None = None) -> SQLiteStorage:
    """Get SQLite storage for shared brain access.

    This is the main entry point for getting storage that works
    across CLI, MCP, and other tools. Storage instances are cached
    per database path to avoid connection leaks.

    Args:
        brain_name: Brain name, or use config's current_brain if None

    Returns:
        SQLiteStorage instance, initialized and ready to use
    """
    from neural_memory.core.brain import Brain
    from neural_memory.storage.sqlite_store import SQLiteStorage

    config = get_config()

    # When no explicit brain is requested, re-read current_brain from
    # config.toml on disk.  This picks up brain switches made by the CLI
    # (which writes to both config.json and config.toml) without
    # requiring a full MCP server restart.
    if brain_name is None:
        disk_brain = _read_current_brain_from_toml()
        if disk_brain is not None and disk_brain != config.current_brain:
            logger = logging.getLogger(__name__)
            logger.info("Brain changed on disk: %s → %s", config.current_brain, disk_brain)
            config.current_brain = disk_brain

    # Auto-migrate flat-layout DB → brains/ layout (one-time, non-blocking)
    _migrate_legacy_db(config, brain_name)

    db_path = config.get_brain_db_path(brain_name)
    cache_key = str(db_path)

    # Return cached storage if available and still open
    if cache_key in _storage_cache:
        cached = _storage_cache[cache_key]
        if cached._conn is not None:
            name = brain_name or config.current_brain
            cached.set_brain(name)
            return cached

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create and initialize storage
    storage = SQLiteStorage(db_path)
    try:
        await storage.initialize()
    except Exception:
        await storage.close()
        raise

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
            freshness_weight=config.brain.freshness_weight,
        )
        brain = Brain.create(name=name, config=brain_config, brain_id=name)
        await storage.save_brain(brain)

    storage.set_brain(brain.id)
    _storage_cache[cache_key] = storage
    return storage
