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
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Storage settings
    storage_backend: str = "memory"  # memory, sqlite, neo4j
    sqlite_path: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None

    # Brain defaults
    default_decay_rate: float = 0.1
    default_activation_threshold: float = 0.2
    default_max_spread_hops: int = 4
    default_max_context_tokens: int = 1500

    # CORS settings
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

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
            return int(value)

        def get_float(key: str, default: float) -> float:
            value = os.getenv(key)
            if value is None:
                return default
            return float(value)

        def get_list(key: str, default: list[str]) -> list[str]:
            value = os.getenv(key)
            if value is None:
                return default
            return [s.strip() for s in value.split(",")]

        return cls(
            host=os.getenv("NEURAL_MEMORY_HOST", "0.0.0.0"),
            port=get_int("NEURAL_MEMORY_PORT", 8000),
            debug=get_bool("NEURAL_MEMORY_DEBUG", False),
            storage_backend=os.getenv("NEURAL_MEMORY_STORAGE", "memory"),
            sqlite_path=os.getenv("NEURAL_MEMORY_SQLITE_PATH"),
            neo4j_uri=os.getenv("NEURAL_MEMORY_NEO4J_URI"),
            neo4j_user=os.getenv("NEURAL_MEMORY_NEO4J_USER"),
            neo4j_password=os.getenv("NEURAL_MEMORY_NEO4J_PASSWORD"),
            default_decay_rate=get_float("NEURAL_MEMORY_DECAY_RATE", 0.1),
            default_activation_threshold=get_float(
                "NEURAL_MEMORY_ACTIVATION_THRESHOLD", 0.2
            ),
            default_max_spread_hops=get_int("NEURAL_MEMORY_MAX_SPREAD_HOPS", 4),
            default_max_context_tokens=get_int("NEURAL_MEMORY_MAX_CONTEXT_TOKENS", 1500),
            cors_origins=get_list("NEURAL_MEMORY_CORS_ORIGINS", ["*"]),
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
