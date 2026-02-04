"""CLI configuration management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def get_default_data_dir() -> Path:
    """Get default data directory for neural-memory."""
    # Cross-platform: ~/.neural-memory/
    return Path.home() / ".neural-memory"


@dataclass
class CLIConfig:
    """CLI configuration."""

    data_dir: Path = field(default_factory=get_default_data_dir)
    current_brain: str = "default"
    default_depth: int | None = None  # Auto-detect
    default_max_tokens: int = 500
    json_output: bool = False

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

        return cls(
            data_dir=data_dir,
            current_brain=data.get("current_brain", "default"),
            default_depth=data.get("default_depth"),
            default_max_tokens=data.get("default_max_tokens", 500),
            json_output=data.get("json_output", False),
        )

    def save(self) -> None:
        """Save configuration to file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.data_dir / "config.json"

        data = {
            "current_brain": self.current_brain,
            "default_depth": self.default_depth,
            "default_max_tokens": self.default_max_tokens,
            "json_output": self.json_output,
            "updated_at": datetime.now().isoformat(),
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @property
    def brains_dir(self) -> Path:
        """Get brains directory."""
        return self.data_dir / "brains"

    def get_brain_path(self, brain_name: str | None = None) -> Path:
        """Get path to brain data file."""
        name = brain_name or self.current_brain
        return self.brains_dir / f"{name}.json"

    def list_brains(self) -> list[str]:
        """List available brains."""
        if not self.brains_dir.exists():
            return []
        return [p.stem for p in self.brains_dir.glob("*.json")]
