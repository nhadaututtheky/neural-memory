"""Configuration management for NeuralMemory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Application configuration.

    Loaded from environment variables with sensible defaults.
    """

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Storage settings
    storage_backend: str = "memory"  # memory, sqlite, falkordb
    sqlite_path: str | None = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_username: str | None = None
    falkordb_password: str | None = None

    # Brain defaults
    default_decay_rate: float = 0.1
    default_activation_threshold: float = 0.2
    default_max_spread_hops: int = 4
    default_max_context_tokens: int = 1500

    # CORS settings
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:*", "http://127.0.0.1:*"]
    )

    # Trusted networks (CIDR notation, e.g. "172.16.0.0/12,192.168.0.0/16")
    # Allows non-localhost requests from these networks (for Docker/container deployments)
    trusted_networks: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""

        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default

        def get_list(key: str, default: list[str]) -> list[str]:
            value = os.getenv(key)
            if value is None:
                return default
            return [s.strip() for s in value.split(",")]

        return cls(
            host=os.getenv("NEURAL_MEMORY_HOST", "127.0.0.1"),
            port=get_int("NEURAL_MEMORY_PORT", 8000),
            debug=get_bool("NEURAL_MEMORY_DEBUG", False),
            storage_backend=os.getenv("NEURAL_MEMORY_STORAGE", "memory"),
            sqlite_path=os.getenv("NEURAL_MEMORY_SQLITE_PATH"),
            falkordb_host=os.getenv("NEURAL_MEMORY_FALKORDB_HOST", "localhost"),
            falkordb_port=get_int("NEURAL_MEMORY_FALKORDB_PORT", 6379),
            falkordb_username=os.getenv("NEURAL_MEMORY_FALKORDB_USERNAME"),
            falkordb_password=os.getenv("NEURAL_MEMORY_FALKORDB_PASSWORD"),
            default_decay_rate=get_float("NEURAL_MEMORY_DECAY_RATE", 0.1),
            default_activation_threshold=get_float("NEURAL_MEMORY_ACTIVATION_THRESHOLD", 0.2),
            default_max_spread_hops=get_int("NEURAL_MEMORY_MAX_SPREAD_HOPS", 4),
            default_max_context_tokens=get_int("NEURAL_MEMORY_MAX_CONTEXT_TOKENS", 1500),
            cors_origins=get_list(
                "NEURAL_MEMORY_CORS_ORIGINS",
                ["http://localhost:*", "http://127.0.0.1:*"],
            ),
            trusted_networks=get_list(
                "NEURAL_MEMORY_TRUSTED_NETWORKS",
                [],
            ),
        )


# Singleton config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
