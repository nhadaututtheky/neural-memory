"""OpenClaw configuration model and persistence.

Stores OpenClaw integration settings in ~/.neuralmemory/openclaw.json.
Covers: API keys, channel connections (Telegram/Discord),
function toggles, and security restrictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Sub-models ────────────────────────────────────────────────


class ApiKeyEntry(BaseModel):
    """Single API key entry for a provider."""

    provider: str
    key: str = ""
    label: str = ""
    enabled: bool = True


class TelegramConfig(BaseModel):
    """Telegram bot connection settings."""

    bot_token: str = ""
    chat_ids: list[str] = Field(default_factory=list)
    enabled: bool = False
    parse_mode: str = "Markdown"


class DiscordConfig(BaseModel):
    """Discord bot connection settings."""

    bot_token: str = ""
    guild_id: str = ""
    channel_ids: list[str] = Field(default_factory=list)
    enabled: bool = False


class FunctionConfig(BaseModel):
    """Single function toggle and settings."""

    name: str
    enabled: bool = True
    timeout_ms: int = 30000
    description: str = ""
    restricted: bool = False


class SecurityConfig(BaseModel):
    """Global security restrictions for OpenClaw."""

    sandbox_mode: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=list)
    max_tokens_per_request: int = 100000
    rate_limit_rpm: int = 60


# ── Root config model ─────────────────────────────────────────


class OpenClawConfig(BaseModel):
    """Complete OpenClaw configuration."""

    api_keys: list[ApiKeyEntry] = Field(default_factory=list)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    functions: list[FunctionConfig] = Field(default_factory=list)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Default functions ─────────────────────────────────────────

DEFAULT_FUNCTIONS: list[dict[str, Any]] = [
    {"name": "web_search", "enabled": True, "timeout_ms": 15000, "description": "Search the web"},
    {
        "name": "code_exec",
        "enabled": False,
        "timeout_ms": 30000,
        "description": "Execute code snippets",
        "restricted": True,
    },
    {"name": "file_read", "enabled": True, "timeout_ms": 5000, "description": "Read local files"},
    {
        "name": "file_write",
        "enabled": False,
        "timeout_ms": 5000,
        "description": "Write local files",
        "restricted": True,
    },
    {
        "name": "memory_recall",
        "enabled": True,
        "timeout_ms": 10000,
        "description": "Recall from NeuralMemory brain",
    },
    {
        "name": "memory_store",
        "enabled": True,
        "timeout_ms": 10000,
        "description": "Store to NeuralMemory brain",
    },
    {
        "name": "shell_exec",
        "enabled": False,
        "timeout_ms": 60000,
        "description": "Execute shell commands",
        "restricted": True,
    },
    {
        "name": "http_request",
        "enabled": True,
        "timeout_ms": 15000,
        "description": "Make HTTP requests",
    },
]


# ── Config manager ────────────────────────────────────────────


class OpenClawConfigManager:
    """Load/save OpenClaw config from ~/.neuralmemory/openclaw.json."""

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            from neural_memory.unified_config import get_neuralmemory_dir

            config_dir = get_neuralmemory_dir()
        self._path = config_dir / "openclaw.json"

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> OpenClawConfig:
        """Load config from disk, returning defaults if missing."""
        if not self._path.exists():
            return self._with_defaults(OpenClawConfig())

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            config = OpenClawConfig.model_validate(raw)
            return self._with_defaults(config)
        except Exception as exc:
            logger.warning("Failed to load OpenClaw config from %s: %s", self._path, exc)
            return self._with_defaults(OpenClawConfig())

    def save(self, config: OpenClawConfig) -> None:
        """Save config to disk with restrictive permissions (contains secrets)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="json")
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Restrict file permissions — config contains API keys and bot tokens
        try:
            self._path.chmod(0o600)
        except OSError:
            pass  # Windows may not support chmod

    def update_api_key(self, provider: str, key: str, label: str = "") -> OpenClawConfig:
        """Add or update an API key for a provider."""
        config = self.load()
        existing = [k for k in config.api_keys if k.provider != provider]
        existing.append(ApiKeyEntry(provider=provider, key=key, label=label, enabled=True))
        updated = config.model_copy(update={"api_keys": existing})
        self.save(updated)
        return updated

    def remove_api_key(self, provider: str) -> OpenClawConfig:
        """Remove an API key for a provider."""
        config = self.load()
        filtered = [k for k in config.api_keys if k.provider != provider]
        updated = config.model_copy(update={"api_keys": filtered})
        self.save(updated)
        return updated

    def update_telegram(self, telegram: TelegramConfig) -> OpenClawConfig:
        """Update Telegram configuration."""
        config = self.load()
        updated = config.model_copy(update={"telegram": telegram})
        self.save(updated)
        return updated

    def update_discord(self, discord: DiscordConfig) -> OpenClawConfig:
        """Update Discord configuration."""
        config = self.load()
        updated = config.model_copy(update={"discord": discord})
        self.save(updated)
        return updated

    def toggle_function(self, name: str, enabled: bool) -> OpenClawConfig:
        """Toggle a function on/off."""
        config = self.load()
        functions = []
        found = False
        for fn in config.functions:
            if fn.name == name:
                functions.append(fn.model_copy(update={"enabled": enabled}))
                found = True
            else:
                functions.append(fn)
        if not found:
            functions.append(FunctionConfig(name=name, enabled=enabled))
        updated = config.model_copy(update={"functions": functions})
        self.save(updated)
        return updated

    @staticmethod
    def _with_defaults(config: OpenClawConfig) -> OpenClawConfig:
        """Ensure default functions exist if none configured."""
        if not config.functions:
            defaults = [FunctionConfig.model_validate(f) for f in DEFAULT_FUNCTIONS]
            return config.model_copy(update={"functions": defaults})
        return config
