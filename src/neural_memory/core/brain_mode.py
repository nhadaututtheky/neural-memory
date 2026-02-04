"""Brain mode configuration for local/shared storage toggle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BrainMode(StrEnum):
    """Brain storage mode."""

    LOCAL = "local"
    """Store memories locally only (SQLite or in-memory)."""

    SHARED = "shared"
    """Connect to remote NeuralMemory server for real-time sharing."""

    HYBRID = "hybrid"
    """Write locally, sync to server periodically (offline-first)."""


class SyncStrategy(StrEnum):
    """Synchronization strategy for hybrid mode."""

    PUSH_ONLY = "push_only"
    """Only push local changes to server."""

    PULL_ONLY = "pull_only"
    """Only pull changes from server."""

    BIDIRECTIONAL = "bidirectional"
    """Full two-way sync."""

    ON_DEMAND = "on_demand"
    """Manual sync only."""


@dataclass(frozen=True)
class SharedConfig:
    """Configuration for shared storage mode."""

    server_url: str
    """URL of NeuralMemory server (e.g., http://localhost:8000)."""

    api_key: str | None = None
    """Optional API key for authentication."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    retry_count: int = 3
    """Number of retries on connection failure."""

    retry_delay: float = 1.0
    """Delay between retries in seconds."""

    def with_server_url(self, url: str) -> SharedConfig:
        """Create new config with different server URL."""
        return SharedConfig(
            server_url=url,
            api_key=self.api_key,
            timeout=self.timeout,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
        )

    def with_api_key(self, key: str | None) -> SharedConfig:
        """Create new config with different API key."""
        return SharedConfig(
            server_url=self.server_url,
            api_key=key,
            timeout=self.timeout,
            retry_count=self.retry_count,
            retry_delay=self.retry_delay,
        )


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid mode (offline-first with sync)."""

    local_path: str
    """Path to local SQLite database."""

    server_url: str
    """URL of NeuralMemory server for sync."""

    api_key: str | None = None
    """Optional API key for authentication."""

    sync_strategy: SyncStrategy = SyncStrategy.BIDIRECTIONAL
    """How to synchronize with server."""

    sync_interval_seconds: int = 60
    """How often to sync (0 = only manual)."""

    auto_sync_on_encode: bool = True
    """Automatically push new memories to server."""

    auto_sync_on_query: bool = False
    """Pull from server before queries."""

    conflict_resolution: str = "prefer_local"
    """How to resolve conflicts: 'prefer_local', 'prefer_remote', 'prefer_recent'."""


@dataclass(frozen=True)
class BrainModeConfig:
    """
    Configuration for brain storage mode.

    Determines how memories are stored and shared.

    Examples:
        # Local-only mode (default)
        config = BrainModeConfig(mode=BrainMode.LOCAL)

        # Shared mode - connect to server
        config = BrainModeConfig(
            mode=BrainMode.SHARED,
            shared=SharedConfig(server_url="http://localhost:8000"),
        )

        # Hybrid mode - offline-first with sync
        config = BrainModeConfig(
            mode=BrainMode.HYBRID,
            hybrid=HybridConfig(
                local_path="./brain.db",
                server_url="http://localhost:8000",
                sync_interval_seconds=300,
            ),
        )
    """

    mode: BrainMode = BrainMode.LOCAL
    """Current storage mode."""

    shared: SharedConfig | None = None
    """Configuration for shared mode."""

    hybrid: HybridConfig | None = None
    """Configuration for hybrid mode."""

    @classmethod
    def local(cls) -> BrainModeConfig:
        """Create local-only configuration."""
        return cls(mode=BrainMode.LOCAL)

    @classmethod
    def shared_mode(
        cls,
        server_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> BrainModeConfig:
        """
        Create shared configuration.

        Args:
            server_url: URL of NeuralMemory server
            api_key: Optional API key
            timeout: Request timeout in seconds
        """
        return cls(
            mode=BrainMode.SHARED,
            shared=SharedConfig(
                server_url=server_url,
                api_key=api_key,
                timeout=timeout,
            ),
        )

    @classmethod
    def hybrid_mode(
        cls,
        local_path: str,
        server_url: str,
        api_key: str | None = None,
        sync_interval: int = 60,
        strategy: SyncStrategy = SyncStrategy.BIDIRECTIONAL,
    ) -> BrainModeConfig:
        """
        Create hybrid (offline-first) configuration.

        Args:
            local_path: Path to local SQLite database
            server_url: URL of NeuralMemory server
            api_key: Optional API key
            sync_interval: Sync interval in seconds (0 = manual only)
            strategy: Synchronization strategy
        """
        return cls(
            mode=BrainMode.HYBRID,
            hybrid=HybridConfig(
                local_path=local_path,
                server_url=server_url,
                api_key=api_key,
                sync_strategy=strategy,
                sync_interval_seconds=sync_interval,
            ),
        )

    def is_local(self) -> bool:
        """Check if using local-only mode."""
        return self.mode == BrainMode.LOCAL

    def is_shared(self) -> bool:
        """Check if using shared (remote) mode."""
        return self.mode == BrainMode.SHARED

    def is_hybrid(self) -> bool:
        """Check if using hybrid (offline-first) mode."""
        return self.mode == BrainMode.HYBRID

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"mode": self.mode.value}

        if self.shared:
            result["shared"] = {
                "server_url": self.shared.server_url,
                "api_key": self.shared.api_key,
                "timeout": self.shared.timeout,
                "retry_count": self.shared.retry_count,
                "retry_delay": self.shared.retry_delay,
            }

        if self.hybrid:
            result["hybrid"] = {
                "local_path": self.hybrid.local_path,
                "server_url": self.hybrid.server_url,
                "api_key": self.hybrid.api_key,
                "sync_strategy": self.hybrid.sync_strategy.value,
                "sync_interval_seconds": self.hybrid.sync_interval_seconds,
                "auto_sync_on_encode": self.hybrid.auto_sync_on_encode,
                "auto_sync_on_query": self.hybrid.auto_sync_on_query,
                "conflict_resolution": self.hybrid.conflict_resolution,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainModeConfig:
        """Create from dictionary."""
        mode = BrainMode(data.get("mode", "local"))

        shared = None
        if "shared" in data:
            s = data["shared"]
            shared = SharedConfig(
                server_url=s["server_url"],
                api_key=s.get("api_key"),
                timeout=s.get("timeout", 30.0),
                retry_count=s.get("retry_count", 3),
                retry_delay=s.get("retry_delay", 1.0),
            )

        hybrid = None
        if "hybrid" in data:
            h = data["hybrid"]
            hybrid = HybridConfig(
                local_path=h["local_path"],
                server_url=h["server_url"],
                api_key=h.get("api_key"),
                sync_strategy=SyncStrategy(h.get("sync_strategy", "bidirectional")),
                sync_interval_seconds=h.get("sync_interval_seconds", 60),
                auto_sync_on_encode=h.get("auto_sync_on_encode", True),
                auto_sync_on_query=h.get("auto_sync_on_query", False),
                conflict_resolution=h.get("conflict_resolution", "prefer_local"),
            )

        return cls(mode=mode, shared=shared, hybrid=hybrid)
