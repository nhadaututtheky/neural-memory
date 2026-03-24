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

import asyncio
import json
import logging
import os
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Valid brain name: alphanumeric, hyphens, underscores, dots (no path separators)
_BRAIN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# Valid sync identifier: alphanumeric, hyphens, underscores, dots, @ (for emails)
_SYNC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.@]*$")
_SYNC_ID_MAX_LEN = 128

# Valid TOML string value: alphanumeric, hyphens, underscores, dots, slashes, spaces
_TOML_SAFE_STRING = re.compile(r"^[a-zA-Z0-9_\-\./ ]*$")
_TOML_STR_MAX_LEN = 128


def get_neuralmemory_dir() -> Path:
    """Get NeuralMemory data directory.

    Priority:
    1. NEURALMEMORY_DIR environment variable
    2. ~/.neuralmemory/
    """
    env_dir = os.environ.get("NEURALMEMORY_DIR")
    if env_dir:
        return Path(env_dir).resolve()
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


@dataclass(frozen=True)
class EmbeddingSettings:
    """Settings for embedding-based cross-language recall."""

    enabled: bool = False
    provider: str = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7

    _VALID_PROVIDERS: ClassVar[tuple[str, ...]] = (
        "sentence_transformer",
        "openai",
        "openrouter",
        "gemini",
        "ollama",
        "auto",
        "",
    )

    def __post_init__(self) -> None:
        if self.provider not in self._VALID_PROVIDERS:
            import logging

            logging.getLogger(__name__).warning(
                "Invalid embedding provider %r, falling back to disabled. Valid: %s",
                self.provider,
                self._VALID_PROVIDERS,
            )
            object.__setattr__(self, "provider", "")
            object.__setattr__(self, "enabled", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSettings:
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=str(data.get("provider", "sentence_transformer")),
            model=str(data.get("model", "all-MiniLM-L6-v2")),
            similarity_threshold=float(data.get("similarity_threshold", 0.7)),
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
class WriteGateConfig:
    """Write-gate configuration for memory quality enforcement.

    When enabled, memories that fail quality checks are rejected before storage.
    This prevents low-quality content from degrading brain purity.

    Addresses GitHub Issue #95: write-gate to improve brain purity.
    """

    enabled: bool = False  # opt-in, backward compat
    min_length: int = 30  # reject content shorter than this
    min_quality_score: int = 3  # reject score below this (0-10 scale)
    auto_capture_min_score: int = 5  # stricter threshold for passive captures
    max_content_length: int = 2000  # reject wall-of-text above this
    reject_generic_filler: bool = True  # reject "done", "ok", "completed" etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_length": self.min_length,
            "min_quality_score": self.min_quality_score,
            "auto_capture_min_score": self.auto_capture_min_score,
            "max_content_length": self.max_content_length,
            "reject_generic_filler": self.reject_generic_filler,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WriteGateConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_length=int(data.get("min_length", 30)),
            min_quality_score=int(data.get("min_quality_score", 3)),
            auto_capture_min_score=int(data.get("auto_capture_min_score", 5)),
            max_content_length=int(data.get("max_content_length", 2000)),
            reject_generic_filler=bool(data.get("reject_generic_filler", True)),
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
    consolidation_ratio_threshold: float = 0.1
    auto_consolidate: bool = True
    auto_consolidate_strategies: tuple[str, ...] = ("prune", "merge", "mature", "infer")
    consolidate_cooldown_minutes: int = 30
    dream_cooldown_hours: int = 24
    expiry_cleanup_enabled: bool = True
    expiry_cleanup_interval_hours: int = 12
    expiry_cleanup_max_per_run: int = 100
    scheduled_consolidation_enabled: bool = True
    scheduled_consolidation_interval_hours: int = 24
    scheduled_consolidation_strategies: tuple[str, ...] = ("prune", "merge", "enrich")
    version_check_enabled: bool = True
    version_check_interval_hours: int = 24
    # Lifecycle engine configuration
    lifecycle_enabled: bool = True
    lifecycle_heat_threshold: float = 0.5
    lifecycle_recency_active_days: float = 3.0
    # Auto-decay in serve daemon
    decay_enabled: bool = True
    decay_interval_hours: int = 12
    # Scheduled re-index
    reindex_enabled: bool = False
    reindex_paths: tuple[str, ...] = ()
    reindex_interval_hours: int = 168  # weekly
    reindex_extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".rst",
        ".html",
        ".css",
    )
    # Notifications (webhook + health alerts)
    notifications_enabled: bool = False
    notifications_webhook_url: str = ""
    notifications_health_threshold: str = "D"  # alert at D or F
    notifications_daily_summary: bool = False
    notifications_zero_activity_alert: bool = True

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
            "consolidation_ratio_threshold": self.consolidation_ratio_threshold,
            "auto_consolidate": self.auto_consolidate,
            "auto_consolidate_strategies": list(self.auto_consolidate_strategies),
            "consolidate_cooldown_minutes": self.consolidate_cooldown_minutes,
            "dream_cooldown_hours": self.dream_cooldown_hours,
            "expiry_cleanup_enabled": self.expiry_cleanup_enabled,
            "expiry_cleanup_interval_hours": self.expiry_cleanup_interval_hours,
            "expiry_cleanup_max_per_run": self.expiry_cleanup_max_per_run,
            "scheduled_consolidation_enabled": self.scheduled_consolidation_enabled,
            "scheduled_consolidation_interval_hours": self.scheduled_consolidation_interval_hours,
            "scheduled_consolidation_strategies": list(self.scheduled_consolidation_strategies),
            "version_check_enabled": self.version_check_enabled,
            "version_check_interval_hours": self.version_check_interval_hours,
            "lifecycle_enabled": self.lifecycle_enabled,
            "lifecycle_heat_threshold": self.lifecycle_heat_threshold,
            "lifecycle_recency_active_days": self.lifecycle_recency_active_days,
            "decay_enabled": self.decay_enabled,
            "decay_interval_hours": self.decay_interval_hours,
            "reindex_enabled": self.reindex_enabled,
            "reindex_paths": list(self.reindex_paths),
            "reindex_interval_hours": self.reindex_interval_hours,
            "reindex_extensions": list(self.reindex_extensions),
            "notifications_enabled": self.notifications_enabled,
            "notifications_webhook_url": self.notifications_webhook_url,
            "notifications_health_threshold": self.notifications_health_threshold,
            "notifications_daily_summary": self.notifications_daily_summary,
            "notifications_zero_activity_alert": self.notifications_zero_activity_alert,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaintenanceConfig:
        strategies = data.get("auto_consolidate_strategies", ("prune", "merge", "mature", "infer"))
        if isinstance(strategies, list):
            strategies = tuple(strategies)
        sched_strategies = data.get(
            "scheduled_consolidation_strategies", ("prune", "merge", "enrich")
        )
        if isinstance(sched_strategies, list):
            sched_strategies = tuple(sched_strategies)
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
            consolidation_ratio_threshold=data.get("consolidation_ratio_threshold", 0.1),
            auto_consolidate=data.get("auto_consolidate", True),
            auto_consolidate_strategies=strategies,
            consolidate_cooldown_minutes=data.get("consolidate_cooldown_minutes", 30),
            dream_cooldown_hours=data.get("dream_cooldown_hours", 24),
            expiry_cleanup_enabled=data.get("expiry_cleanup_enabled", True),
            expiry_cleanup_interval_hours=data.get("expiry_cleanup_interval_hours", 12),
            expiry_cleanup_max_per_run=data.get("expiry_cleanup_max_per_run", 100),
            scheduled_consolidation_enabled=data.get("scheduled_consolidation_enabled", True),
            scheduled_consolidation_interval_hours=data.get(
                "scheduled_consolidation_interval_hours", 24
            ),
            scheduled_consolidation_strategies=sched_strategies,
            version_check_enabled=data.get("version_check_enabled", True),
            version_check_interval_hours=data.get("version_check_interval_hours", 24),
            lifecycle_enabled=data.get("lifecycle_enabled", True),
            lifecycle_heat_threshold=data.get("lifecycle_heat_threshold", 0.5),
            lifecycle_recency_active_days=data.get("lifecycle_recency_active_days", 3.0),
            decay_enabled=data.get("decay_enabled", True),
            decay_interval_hours=data.get("decay_interval_hours", 12),
            reindex_enabled=data.get("reindex_enabled", False),
            reindex_paths=tuple(data.get("reindex_paths", ())),
            reindex_interval_hours=data.get("reindex_interval_hours", 168),
            reindex_extensions=tuple(
                data.get(
                    "reindex_extensions",
                    (
                        ".md",
                        ".txt",
                        ".py",
                        ".js",
                        ".ts",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".toml",
                        ".rst",
                        ".html",
                        ".css",
                    ),
                )
            ),
            notifications_enabled=data.get("notifications_enabled", False),
            notifications_webhook_url=data.get("notifications_webhook_url", ""),
            notifications_health_threshold=data.get("notifications_health_threshold", "D"),
            notifications_daily_summary=data.get("notifications_daily_summary", False),
            notifications_zero_activity_alert=data.get("notifications_zero_activity_alert", True),
        )


_VALID_TOOL_TIERS = frozenset({"minimal", "standard", "full"})


@dataclass(frozen=True)
class ToolTierConfig:
    """MCP tool tier configuration.

    Controls which tools are exposed via tools/list to reduce token overhead.
    Hidden tools remain callable via dispatch — only schema exposure changes.
    """

    tier: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return {"tier": self.tier}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolTierConfig:
        raw = str(data.get("tier", "full")).lower().strip()
        if raw not in _VALID_TOOL_TIERS:
            raw = "full"
        return cls(tier=raw)


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
class EncryptionConfig:
    """Encryption configuration for sensitive memory content.

    When enabled, neuron content detected as sensitive (or explicitly flagged)
    is encrypted using Fernet symmetric encryption with per-brain keys.
    """

    enabled: bool = True
    auto_encrypt_sensitive: bool = True
    keys_dir: str = ""  # empty = use {data_dir}/keys/

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_encrypt_sensitive": self.auto_encrypt_sensitive,
            "keys_dir": self.keys_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptionConfig:
        keys_dir = str(data.get("keys_dir", ""))[:256]
        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_encrypt_sensitive=bool(data.get("auto_encrypt_sensitive", True)),
            keys_dir=keys_dir,
        )


@dataclass(frozen=True)
class SyncConfig:
    """Multi-device sync configuration."""

    enabled: bool = False
    hub_url: str = ""
    api_key: str = ""
    auto_sync: bool = False
    sync_interval_seconds: int = 300
    conflict_strategy: str = "prefer_recent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hub_url": self.hub_url,
            "api_key": self.api_key,
            "auto_sync": self.auto_sync,
            "sync_interval_seconds": self.sync_interval_seconds,
            "conflict_strategy": self.conflict_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncConfig:
        strategy = str(data.get("conflict_strategy", "prefer_recent"))
        valid_strategies = {"prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"}
        if strategy not in valid_strategies:
            strategy = "prefer_recent"
        try:
            interval = max(10, min(int(data.get("sync_interval_seconds", 300)), 86400))
        except (ValueError, TypeError):
            interval = 300
        hub_url = str(data.get("hub_url", ""))
        # Sanitize hub_url - only allow http/https URLs
        if hub_url and not hub_url.startswith(("http://", "https://")):
            hub_url = ""
        # Truncate URL to reasonable length
        hub_url = hub_url[:256]
        api_key = str(data.get("api_key", ""))
        # Validate api_key format: must start with nmk_ or be empty
        if api_key and not api_key.startswith("nmk_"):
            api_key = ""
        return cls(
            enabled=bool(data.get("enabled", False)),
            hub_url=hub_url,
            api_key=api_key,
            auto_sync=bool(data.get("auto_sync", False)),
            sync_interval_seconds=interval,
            conflict_strategy=strategy,
        )


@dataclass(frozen=True)
class DedupSettings:
    """LLM-powered deduplication settings.

    Controls the 3-tier dedup pipeline: SimHash -> Embedding -> LLM.
    All off by default to preserve zero-LLM core.
    """

    enabled: bool = False
    simhash_threshold: int = 7  # tighter: ~89% similarity (was 10 / ~85%)
    embedding_threshold: float = 0.85
    embedding_ambiguous_low: float = 0.75
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_max_pairs_per_encode: int = 3
    merge_strategy: str = "keep_newer"
    max_candidates: int = 30  # wider search (was 10)

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
            "max_candidates": self.max_candidates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DedupSettings:
        return cls(
            enabled=bool(data.get("enabled", False)),
            simhash_threshold=int(data.get("simhash_threshold", 7)),
            embedding_threshold=float(data.get("embedding_threshold", 0.85)),
            embedding_ambiguous_low=float(data.get("embedding_ambiguous_low", 0.75)),
            llm_enabled=bool(data.get("llm_enabled", False)),
            llm_provider=str(data.get("llm_provider", "none")),
            llm_model=str(data.get("llm_model", "")),
            llm_max_pairs_per_encode=int(data.get("llm_max_pairs_per_encode", 3)),
            merge_strategy=str(data.get("merge_strategy", "keep_newer")),
            max_candidates=int(data.get("max_candidates", 30)),
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


def _sanitize_toml_str(value: str) -> str:
    """Sanitize a string value for safe TOML serialization.

    Prevents TOML injection by stripping control chars and quotes.
    Returns empty string if value contains unsafe characters.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:_TOML_STR_MAX_LEN]
    if not _TOML_SAFE_STRING.match(cleaned):
        return ""
    return cleaned


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


_VALID_STORAGE_BACKENDS = {"sqlite", "falkordb", "postgres"}


def _validate_storage_backend(value: str) -> str:
    """Validate and return storage backend, defaulting to sqlite."""
    if value in _VALID_STORAGE_BACKENDS:
        return value
    logger.warning("Unknown storage_backend '%s', falling back to 'sqlite'", value)
    return "sqlite"


@dataclass(frozen=True)
class FalkorDBConfig:
    """FalkorDB graph storage backend configuration."""

    host: str = "localhost"
    port: int = 6379
    username: str = ""
    password: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***" if self.password else "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FalkorDBConfig:
        host = str(os.environ.get("NEURAL_MEMORY_FALKORDB_HOST") or data.get("host", "localhost"))[
            :256
        ]
        try:
            port_raw = os.environ.get("NEURAL_MEMORY_FALKORDB_PORT") or data.get("port", 6379)
            port = max(1, min(int(port_raw), 65535))
        except (ValueError, TypeError):
            port = 6379
        username = str(
            os.environ.get("NEURAL_MEMORY_FALKORDB_USERNAME") or data.get("username", "")
        )[:128]
        password_env = os.environ.get("NEURAL_MEMORY_FALKORDB_PASSWORD")
        password_file = data.get("password", "")
        if not password_env and password_file:
            logger.warning(
                "FalkorDB password read from config.toml — prefer NEURAL_MEMORY_FALKORDB_PASSWORD env var"
            )
        password = str(password_env or password_file)[:256]
        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
        )


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL + pgvector storage backend configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "neuralmemory"
    user: str = "postgres"
    password: str = ""
    embedding_dim: int = 384

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": "***" if self.password else "",
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostgresConfig:
        host = str(os.environ.get("NEURAL_MEMORY_POSTGRES_HOST") or data.get("host", "localhost"))[
            :256
        ]
        try:
            port_raw = os.environ.get("NEURAL_MEMORY_POSTGRES_PORT") or data.get("port", 5432)
            port = max(1, min(int(port_raw), 65535))
        except (ValueError, TypeError):
            port = 5432
        database = str(
            os.environ.get("NEURAL_MEMORY_POSTGRES_DATABASE")
            or data.get("database", "neuralmemory")
        )[:128]
        user = str(os.environ.get("NEURAL_MEMORY_POSTGRES_USER") or data.get("user", "postgres"))[
            :128
        ]
        password_env = os.environ.get("NEURAL_MEMORY_POSTGRES_PASSWORD")
        password_file = data.get("password", "")
        if not password_env and password_file:
            logger.warning(
                "PostgreSQL password read from config.toml — prefer NEURAL_MEMORY_POSTGRES_PASSWORD env var"
            )
        password = str(password_env or password_file)[:256]
        try:
            embedding_dim = max(1, int(data.get("embedding_dim", 384)))
        except (ValueError, TypeError):
            embedding_dim = 384
        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            embedding_dim=embedding_dim,
        )


@dataclass(frozen=True)
class ToolMemoryConfig:
    """Tool memory auto-capture configuration.

    When enabled, a PostToolUse Claude Code hook captures lightweight
    metadata about every MCP tool call into a JSONL buffer. A deferred
    processing step (during consolidation) promotes patterns to neurons
    and synapses (EFFECTIVE_FOR, USED_WITH).
    """

    enabled: bool = True
    min_duration_ms: int = 0  # Ignore tool calls faster than this
    blacklist: tuple[str, ...] = ()  # Tool name prefixes to skip
    cooccurrence_window_s: int = 60  # Seconds for USED_WITH detection
    min_frequency: int = 3  # Min calls before creating a tool neuron
    max_buffer_lines: int = 10000  # Truncate JSONL buffer beyond this
    process_batch_size: int = 200  # Max events per processing cycle

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_duration_ms": self.min_duration_ms,
            "blacklist": list(self.blacklist),
            "cooccurrence_window_s": self.cooccurrence_window_s,
            "min_frequency": self.min_frequency,
            "max_buffer_lines": self.max_buffer_lines,
            "process_batch_size": self.process_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMemoryConfig:
        blacklist_raw = data.get("blacklist", [])
        if isinstance(blacklist_raw, (list, tuple)):
            blacklist = tuple(str(b)[:128] for b in blacklist_raw[:50])
        else:
            blacklist = ()
        try:
            min_dur = max(0, min(int(data.get("min_duration_ms", 0)), 60_000))
        except (ValueError, TypeError):
            min_dur = 0
        try:
            window = max(1, min(int(data.get("cooccurrence_window_s", 60)), 3600))
        except (ValueError, TypeError):
            window = 60
        try:
            min_freq = max(1, min(int(data.get("min_frequency", 3)), 100))
        except (ValueError, TypeError):
            min_freq = 3
        try:
            max_buf = max(100, min(int(data.get("max_buffer_lines", 10000)), 1_000_000))
        except (ValueError, TypeError):
            max_buf = 10000
        try:
            batch = max(10, min(int(data.get("process_batch_size", 200)), 10000))
        except (ValueError, TypeError):
            batch = 200
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_duration_ms=min_dur,
            blacklist=blacklist,
            cooccurrence_window_s=window,
            min_frequency=min_freq,
            max_buffer_lines=max_buf,
            process_batch_size=batch,
        )


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram backup integration configuration.

    Bot token is read from NMEM_TELEGRAM_BOT_TOKEN env var (never in config file).
    Chat IDs are stored in config.toml [telegram] section.
    """

    enabled: bool = False
    chat_ids: tuple[str, ...] = ()
    max_file_size_mb: int = 50
    backup_on_consolidation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "chat_ids": list(self.chat_ids),
            "max_file_size_mb": self.max_file_size_mb,
            "backup_on_consolidation": self.backup_on_consolidation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramConfig:
        raw_ids = data.get("chat_ids", [])
        if isinstance(raw_ids, (list, tuple)):
            chat_ids = tuple(str(cid).strip() for cid in raw_ids if str(cid).strip())
        else:
            chat_ids = ()
        try:
            max_size = max(1, min(int(data.get("max_file_size_mb", 50)), 2000))
        except (ValueError, TypeError):
            max_size = 50
        return cls(
            enabled=bool(data.get("enabled", False)),
            chat_ids=chat_ids,
            max_file_size_mb=max_size,
            backup_on_consolidation=bool(data.get("backup_on_consolidation", False)),
        )


@dataclass
class BudgetRetrievalConfig:
    """Token budget configuration for retrieval context allocation.

    Controls how budget-aware retrieval allocates the context window
    across candidate fibers using value-per-token ranking.
    """

    enabled: bool = True
    default_tokens: int = 4000
    system_overhead: int = 50
    per_fiber_overhead: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_tokens": self.default_tokens,
            "system_overhead": self.system_overhead,
            "per_fiber_overhead": self.per_fiber_overhead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetRetrievalConfig:
        try:
            default_tokens = max(50, min(int(data.get("default_tokens", 4000)), 100_000))
        except (ValueError, TypeError):
            default_tokens = 4000
        try:
            system_overhead = max(0, min(int(data.get("system_overhead", 50)), 500))
        except (ValueError, TypeError):
            system_overhead = 50
        try:
            per_fiber_overhead = max(0, min(int(data.get("per_fiber_overhead", 15)), 200))
        except (ValueError, TypeError):
            per_fiber_overhead = 15
        return cls(
            enabled=bool(data.get("enabled", True)),
            default_tokens=default_tokens,
            system_overhead=system_overhead,
            per_fiber_overhead=per_fiber_overhead,
        )


@dataclass
class ResponseConfig:
    """MCP response compaction settings.

    Controls how verbose MCP tool responses are. Compact mode strips
    metadata hints, truncates lists, and shortens content previews
    to reduce token waste in agent context windows.
    """

    # Enable compact mode globally (agents can also set per-call via compact=true)
    compact_mode: bool = False

    # Max items in list fields before truncation (compact mode only)
    max_list_items: int = 10

    # Strip DX hint fields (maintenance_hint, update_hint, onboarding, etc.)
    strip_hints: bool = True

    # Max chars for content preview in list responses
    content_preview_length: int = 120

    # Auto-compact threshold: if any list in response has more items than this,
    # compact mode is applied automatically (0 = disabled)
    auto_compact_threshold: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseConfig:
        """Create from TOML dict."""
        return cls(
            compact_mode=bool(data.get("compact_mode", False)),
            max_list_items=int(data.get("max_list_items", 10)),
            strip_hints=bool(data.get("strip_hints", True)),
            content_preview_length=int(data.get("content_preview_length", 120)),
            auto_compact_threshold=int(data.get("auto_compact_threshold", 20)),
        )


@dataclass(frozen=True)
class WatcherConfig:
    """Settings for file watcher auto-ingestion."""

    enabled: bool = False
    paths: tuple[str, ...] = ()
    extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".json",
        ".csv",
        ".xlsx",
        ".py",
        ".ts",
        ".js",
    )
    ignore_patterns: tuple[str, ...] = (
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        ".env",
    )
    debounce_seconds: float = 2.0
    max_file_size_mb: int = 10
    max_watched_dirs: int = 10
    memory_type: str = "fact"
    domain_tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "paths": list(self.paths),
            "extensions": list(self.extensions),
            "ignore_patterns": list(self.ignore_patterns),
            "debounce_seconds": self.debounce_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_watched_dirs": self.max_watched_dirs,
            "memory_type": self.memory_type,
            "domain_tag": self.domain_tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatcherConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            paths=tuple(data.get("paths", ())),
            extensions=tuple(data.get("extensions", cls.extensions)),
            ignore_patterns=tuple(data.get("ignore_patterns", cls.ignore_patterns)),
            debounce_seconds=float(data.get("debounce_seconds", 2.0)),
            max_file_size_mb=int(data.get("max_file_size_mb", 10)),
            max_watched_dirs=int(data.get("max_watched_dirs", 10)),
            memory_type=str(data.get("memory_type", "fact")),
            domain_tag=str(data.get("domain_tag", "")),
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

    # Embedding settings (cross-language recall)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)

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

    # Encryption settings
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)

    # Write gate (quality enforcement before storage)
    write_gate: WriteGateConfig = field(default_factory=WriteGateConfig)

    # Dedup settings
    dedup: DedupSettings = field(default_factory=DedupSettings)

    # MCP tool tier settings
    tool_tier: ToolTierConfig = field(default_factory=ToolTierConfig)

    # Mem0 auto-sync settings
    mem0_sync: Mem0SyncConfig = field(default_factory=Mem0SyncConfig)

    # Device identity (stable per-machine ID)
    device_id: str = ""

    # Multi-device sync settings
    sync: SyncConfig = field(default_factory=SyncConfig)

    # Storage backend: "sqlite" (default) or "falkordb"
    storage_backend: str = "sqlite"

    # Tool memory auto-capture
    tool_memory: ToolMemoryConfig = field(default_factory=ToolMemoryConfig)

    # Telegram backup integration
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # File watcher auto-ingestion
    watcher: WatcherConfig = field(default_factory=WatcherConfig)

    # FalkorDB config (used when storage_backend == "falkordb")
    falkordb: FalkorDBConfig = field(default_factory=FalkorDBConfig)

    # PostgreSQL config (used when storage_backend == "postgres")
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    # MCP response compaction
    response: ResponseConfig = field(default_factory=ResponseConfig)

    # Token budget retrieval
    budget: BudgetRetrievalConfig = field(default_factory=BudgetRetrievalConfig)

    # CLI preferences
    json_output: bool = False
    default_depth: int | None = None
    default_max_tokens: int = 500

    # Metadata
    version: str = "1.0"

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
            from neural_memory.sync.device import get_device_id as _get_device_id

            config = cls(
                data_dir=data_dir,
                current_brain=legacy_brain or get_default_brain(),
                device_id=_get_device_id(data_dir),
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

        from neural_memory.sync.device import get_device_id

        sync_data = data.get("sync", {})
        # device_id: prefer explicit value in [sync] section, else generate/read from file
        raw_device_id = str(data.get("device_id", "") or sync_data.get("device_id", "")).strip()
        if not raw_device_id:
            raw_device_id = get_device_id(data_dir)

        return cls(
            data_dir=data_dir,
            current_brain=(
                os.environ.get("NEURALMEMORY_BRAIN")
                or os.environ.get("NMEM_BRAIN")
                or data.get("current_brain", get_default_brain())
            ),
            brain=BrainSettings.from_dict(data.get("brain", {})),
            embedding=EmbeddingSettings.from_dict(data.get("embedding", {})),
            auto=AutoConfig.from_dict(data.get("auto", {})),
            eternal=EternalConfig.from_dict(data.get("eternal", {})),
            maintenance=MaintenanceConfig.from_dict(data.get("maintenance", {})),
            conflict=ConflictConfig.from_dict(data.get("conflict", {})),
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            encryption=EncryptionConfig.from_dict(data.get("encryption", {})),
            write_gate=WriteGateConfig.from_dict(data.get("write_gate", {})),
            dedup=DedupSettings.from_dict(data.get("dedup", {})),
            tool_memory=ToolMemoryConfig.from_dict(data.get("tool_memory", {})),
            telegram=TelegramConfig.from_dict(data.get("telegram", {})),
            tool_tier=ToolTierConfig.from_dict(data.get("tool_tier", {})),
            mem0_sync=Mem0SyncConfig.from_dict(data.get("mem0_sync", {})),
            watcher=WatcherConfig.from_dict(data.get("watcher", {})),
            device_id=raw_device_id,
            sync=SyncConfig.from_dict(sync_data),
            storage_backend=_validate_storage_backend(str(data.get("storage_backend", "sqlite"))),
            falkordb=FalkorDBConfig.from_dict(data.get("falkordb", {})),
            postgres=PostgresConfig.from_dict(data.get("postgres", {})),
            response=ResponseConfig.from_dict(data.get("response", {})),
            budget=BudgetRetrievalConfig.from_dict(data.get("budget", {})),
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
            "# Embedding settings (cross-language recall via Gemini/OpenAI/OpenRouter)",
            "[embedding]",
            f"enabled = {'true' if self.embedding.enabled else 'false'}",
            f'provider = "{self.embedding.provider}"',
            f'model = "{self.embedding.model}"',
            f"similarity_threshold = {self.embedding.similarity_threshold}",
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
            f"expiry_cleanup_enabled = {'true' if self.maintenance.expiry_cleanup_enabled else 'false'}",
            f"expiry_cleanup_interval_hours = {self.maintenance.expiry_cleanup_interval_hours}",
            f"expiry_cleanup_max_per_run = {self.maintenance.expiry_cleanup_max_per_run}",
            f"scheduled_consolidation_enabled = {'true' if self.maintenance.scheduled_consolidation_enabled else 'false'}",
            f"scheduled_consolidation_interval_hours = {self.maintenance.scheduled_consolidation_interval_hours}",
            f"scheduled_consolidation_strategies = {json.dumps(list(self.maintenance.scheduled_consolidation_strategies))}",
            "",
            "# Conflict resolution settings",
            "[conflict]",
            f"auto_resolve_trivial = {'true' if self.conflict.auto_resolve_trivial else 'false'}",
            "",
            "# Safety settings",
            "[safety]",
            f"auto_redact_min_severity = {self.safety.auto_redact_min_severity}",
            "",
            "# Encryption settings",
            "[encryption]",
            f"enabled = {'true' if self.encryption.enabled else 'false'}",
            f"auto_encrypt_sensitive = {'true' if self.encryption.auto_encrypt_sensitive else 'false'}",
            f'keys_dir = "{_sanitize_toml_str(self.encryption.keys_dir)}"',
            "",
            "# Write gate (quality enforcement before storage)",
            "[write_gate]",
            f"enabled = {'true' if self.write_gate.enabled else 'false'}",
            f"min_length = {self.write_gate.min_length}",
            f"min_quality_score = {self.write_gate.min_quality_score}",
            f"auto_capture_min_score = {self.write_gate.auto_capture_min_score}",
            f"max_content_length = {self.write_gate.max_content_length}",
            f"reject_generic_filler = {'true' if self.write_gate.reject_generic_filler else 'false'}",
            "",
            "# Dedup settings",
            "[dedup]",
            f"enabled = {'true' if self.dedup.enabled else 'false'}",
            f"simhash_threshold = {self.dedup.simhash_threshold}",
            f"embedding_threshold = {self.dedup.embedding_threshold}",
            f"embedding_ambiguous_low = {self.dedup.embedding_ambiguous_low}",
            f"llm_enabled = {'true' if self.dedup.llm_enabled else 'false'}",
            f'llm_provider = "{_sanitize_toml_str(self.dedup.llm_provider)}"',
            f'llm_model = "{_sanitize_toml_str(self.dedup.llm_model)}"',
            f"llm_max_pairs_per_encode = {self.dedup.llm_max_pairs_per_encode}",
            f'merge_strategy = "{_sanitize_toml_str(self.dedup.merge_strategy)}"',
            f"max_candidates = {self.dedup.max_candidates}",
            "",
            "# Tool memory auto-capture",
            "[tool_memory]",
            f"enabled = {'true' if self.tool_memory.enabled else 'false'}",
            f"min_duration_ms = {self.tool_memory.min_duration_ms}",
            f"blacklist = [{', '.join(repr(b) for b in self.tool_memory.blacklist)}]",
            f"cooccurrence_window_s = {self.tool_memory.cooccurrence_window_s}",
            f"min_frequency = {self.tool_memory.min_frequency}",
            f"max_buffer_lines = {self.tool_memory.max_buffer_lines}",
            f"process_batch_size = {self.tool_memory.process_batch_size}",
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
            "# Multi-device sync settings",
            "[sync]",
            f"enabled = {'true' if self.sync.enabled else 'false'}",
            f'hub_url = "{self.sync.hub_url}"',
            f"auto_sync = {'true' if self.sync.auto_sync else 'false'}",
            f"sync_interval_seconds = {self.sync.sync_interval_seconds}",
            f'conflict_strategy = "{self.sync.conflict_strategy}"',
            "",
            "# Storage backend: sqlite (default) or falkordb",
            f'storage_backend = "{_sanitize_toml_str(self.storage_backend)}"',
            "",
            '# FalkorDB settings (when storage_backend = "falkordb")',
            "[falkordb]",
            f'host = "{_sanitize_toml_str(self.falkordb.host)}"',
            f"port = {self.falkordb.port}",
            f'username = "{_sanitize_toml_str(self.falkordb.username)}"',
            "# Password omitted for security — use env NEURAL_MEMORY_FALKORDB_PASSWORD",
            'password = ""',
            "",
            '# PostgreSQL settings (when storage_backend = "postgres")',
            "[postgres]",
            f'host = "{_sanitize_toml_str(self.postgres.host)}"',
            f"port = {self.postgres.port}",
            f'database = "{_sanitize_toml_str(self.postgres.database)}"',
            f'user = "{_sanitize_toml_str(self.postgres.user)}"',
            'password = ""  # Use env NEURAL_MEMORY_POSTGRES_PASSWORD',
            "",
            "# Telegram backup integration",
            "# Bot token: set NMEM_TELEGRAM_BOT_TOKEN env var (never stored here)",
            "[telegram]",
            f"enabled = {'true' if self.telegram.enabled else 'false'}",
            f"chat_ids = [{', '.join(repr(cid) for cid in self.telegram.chat_ids)}]",
            f"max_file_size_mb = {self.telegram.max_file_size_mb}",
            f"backup_on_consolidation = {'true' if self.telegram.backup_on_consolidation else 'false'}",
            "",
            "# File watcher auto-ingestion",
            "[watcher]",
            f"enabled = {'true' if self.watcher.enabled else 'false'}",
            f"paths = [{', '.join(repr(p) for p in self.watcher.paths)}]",
            f"extensions = [{', '.join(repr(e) for e in self.watcher.extensions)}]",
            f"ignore_patterns = [{', '.join(repr(p) for p in self.watcher.ignore_patterns)}]",
            f"debounce_seconds = {self.watcher.debounce_seconds}",
            f"max_file_size_mb = {self.watcher.max_file_size_mb}",
            f"max_watched_dirs = {self.watcher.max_watched_dirs}",
            f'memory_type = "{_sanitize_toml_str(self.watcher.memory_type)}"',
            f'domain_tag = "{_sanitize_toml_str(self.watcher.domain_tag)}"',
            "",
            "# MCP tool tier (minimal/standard/full)",
            "[tool_tier]",
            f'tier = "{self.tool_tier.tier}"',
            "",
            "# MCP response compaction",
            "[response]",
            f"compact_mode = {'true' if self.response.compact_mode else 'false'}",
            f"max_list_items = {self.response.max_list_items}",
            f"strip_hints = {'true' if self.response.strip_hints else 'false'}",
            f"content_preview_length = {self.response.content_preview_length}",
            f"auto_compact_threshold = {self.response.auto_compact_threshold}",
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
_storage_cache: dict[str, NeuralStorage] = {}
_storage_lock: asyncio.Lock | None = None


def _get_storage_lock() -> asyncio.Lock:
    """Lazy-init asyncio.Lock (must be created inside a running event loop)."""
    global _storage_lock
    if _storage_lock is None:
        _storage_lock = asyncio.Lock()
    return _storage_lock


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
            logger.warning(
                "Found legacy config %s but could not read it", config_file, exc_info=True
            )
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
        logger.debug("Could not read current_brain from config.toml", exc_info=True)
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


async def get_shared_storage(brain_name: str | None = None) -> NeuralStorage:
    """Get storage for shared brain access.

    This is the main entry point for getting storage that works
    across CLI, MCP, and other tools. Storage instances are cached
    to avoid connection leaks.

    Respects config.storage_backend: "sqlite" (default) or "falkordb".

    Args:
        brain_name: Brain name, or use config's current_brain if None

    Returns:
        NeuralStorage instance, initialized and ready to use
    """
    config = get_config()

    # When no explicit brain is requested, resolve from env var or disk.
    #
    # Priority: env var > config.toml > in-memory config
    #
    # IMPORTANT: When NMEM_BRAIN / NEURALMEMORY_BRAIN is set, we use it
    # directly WITHOUT mutating config.current_brain. This ensures
    # process-level isolation for multi-agent setups where each Claude
    # Code session spawns its own MCP server process with a different
    # env var. Mutating the shared config object would cause cross-brain
    # contamination if the config singleton is ever shared.
    if brain_name is None:
        env_brain = os.environ.get("NEURALMEMORY_BRAIN") or os.environ.get("NMEM_BRAIN")
        if env_brain:
            name = env_brain
        else:
            disk_brain = _read_current_brain_from_toml()
            if disk_brain is not None and disk_brain != config.current_brain:
                logger = logging.getLogger(__name__)
                logger.info("Brain changed on disk: %s → %s", config.current_brain, disk_brain)
                config.current_brain = disk_brain
            name = config.current_brain
    else:
        name = brain_name

    # FalkorDB backend
    if config.storage_backend == "falkordb":
        return await _get_falkordb_storage(config, name)

    # PostgreSQL backend
    if config.storage_backend == "postgres":
        return await _get_postgres_storage(config, name)

    # Default: SQLite backend
    return await _get_sqlite_storage(config, name, brain_name)


async def _get_sqlite_storage(
    config: UnifiedConfig,
    name: str,
    brain_name: str | None,
) -> NeuralStorage:
    """Create or return cached SQLiteStorage (lock-protected against races)."""
    lock = _get_storage_lock()
    from neural_memory.core.brain import Brain
    from neural_memory.storage.sqlite_store import SQLiteStorage

    # Auto-migrate flat-layout DB → brains/ layout (one-time, non-blocking)
    _migrate_legacy_db(config, brain_name)

    db_path = config.get_brain_db_path(brain_name)
    cache_key = str(db_path)

    async with lock:
        # Return cached storage if available and still open
        if cache_key in _storage_cache:
            cached = _storage_cache[cache_key]
            if getattr(cached, "_conn", None) is not None:
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
        # Try by id first (normal case: brain_id == name),
        # then fallback to name lookup (handles brains with UUID ids from older versions)
        brain = await storage.get_brain(name)
        if brain is None:
            brain = await storage.find_brain_by_name(name)

        if brain is None:
            from neural_memory.core.brain import BrainConfig

            brain_config = BrainConfig(
                decay_rate=config.brain.decay_rate,
                reinforcement_delta=config.brain.reinforcement_delta,
                activation_threshold=config.brain.activation_threshold,
                max_spread_hops=config.brain.max_spread_hops,
                max_context_tokens=config.brain.max_context_tokens,
                freshness_weight=config.brain.freshness_weight,
                embedding_enabled=config.embedding.enabled,
                embedding_provider=config.embedding.provider,
                embedding_model=config.embedding.model,
                embedding_similarity_threshold=config.embedding.similarity_threshold,
            )
            brain = Brain.create(name=name, config=brain_config, brain_id=name)
            await storage.save_brain(brain)

        storage.set_brain(brain.id)
        _storage_cache[cache_key] = storage
        return storage


# Cached FalkorDB storage (single connection, multi-graph)
_falkordb_storage: NeuralStorage | None = None


async def _get_falkordb_storage(config: UnifiedConfig, name: str) -> NeuralStorage:
    """Create or return cached FalkorDBStorage."""
    global _falkordb_storage

    from neural_memory.core.brain import Brain
    from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage

    if _falkordb_storage is not None:
        # Verify connection is still alive with a PING
        db = getattr(_falkordb_storage, "_db", None)
        if db is not None:
            try:
                await db.connection.ping()
                _falkordb_storage.set_brain(name)
                return _falkordb_storage
            except Exception:
                logger.warning("FalkorDB connection lost, reconnecting")
        _falkordb_storage = None

    fdb_config = config.falkordb
    storage = FalkorDBStorage(
        host=fdb_config.host,
        port=fdb_config.port,
        username=fdb_config.username or None,
        password=fdb_config.password or None,
    )
    await storage.initialize()

    # Ensure brain exists and set context
    await storage.set_brain_with_indexes(name)
    # Try by id first, then fallback to name lookup (older brains may use UUID ids)
    brain = await storage.get_brain(name)
    if brain is None:
        brain = await storage.find_brain_by_name(name)

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
    _falkordb_storage = storage
    return storage


# Cached PostgreSQL storage (single pool, multi-brain)
_postgres_storage: NeuralStorage | None = None


async def _get_postgres_storage(config: UnifiedConfig, name: str) -> NeuralStorage:
    """Create or return cached PostgreSQLStorage."""
    global _postgres_storage

    from neural_memory.core.brain import Brain
    from neural_memory.storage.postgres.postgres_store import PostgreSQLStorage

    if _postgres_storage is not None:
        # Verify the pool is still alive before reusing
        pool = getattr(_postgres_storage, "_pool", None)
        if pool is None or pool.is_closing():
            logger.warning("PostgreSQL connection pool closed, reconnecting")
            _postgres_storage = None
        else:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("SELECT 1")
            except Exception:
                logger.warning("PostgreSQL connection lost, reconnecting")
                _postgres_storage = None

    if _postgres_storage is not None:
        brain = await _postgres_storage.get_brain(name)
        if brain is None:
            brain = await _postgres_storage.find_brain_by_name(name)
        if brain is None:
            from neural_memory.core.brain import BrainConfig

            brain_config = BrainConfig(
                decay_rate=config.brain.decay_rate,
                reinforcement_delta=config.brain.reinforcement_delta,
                activation_threshold=config.brain.activation_threshold,
                max_spread_hops=config.brain.max_spread_hops,
                max_context_tokens=config.brain.max_context_tokens,
                freshness_weight=config.brain.freshness_weight,
                embedding_enabled=config.embedding.enabled,
                embedding_provider=config.embedding.provider,
                embedding_model=config.embedding.model,
                embedding_similarity_threshold=config.embedding.similarity_threshold,
            )
            brain = Brain.create(name=name, config=brain_config, brain_id=name)
            await _postgres_storage.save_brain(brain)
        _postgres_storage.set_brain(brain.id)
        return _postgres_storage

    pg = config.postgres
    storage = PostgreSQLStorage(
        host=pg.host,
        port=pg.port,
        database=pg.database,
        user=pg.user,
        password=pg.password,
        embedding_dim=pg.embedding_dim,
    )
    await storage.initialize()

    storage.set_brain(name)
    brain = await storage.get_brain(name)
    if brain is None:
        brain = await storage.find_brain_by_name(name)

    if brain is None:
        from neural_memory.core.brain import BrainConfig

        brain_config = BrainConfig(
            decay_rate=config.brain.decay_rate,
            reinforcement_delta=config.brain.reinforcement_delta,
            activation_threshold=config.brain.activation_threshold,
            max_spread_hops=config.brain.max_spread_hops,
            max_context_tokens=config.brain.max_context_tokens,
            freshness_weight=config.brain.freshness_weight,
            embedding_enabled=config.embedding.enabled,
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            embedding_similarity_threshold=config.embedding.similarity_threshold,
        )
        brain = Brain.create(name=name, config=brain_config, brain_id=name)
        await storage.save_brain(brain)

    storage.set_brain(brain.id)
    _postgres_storage = storage
    return storage
