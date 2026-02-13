"""Static configuration presets for NeuralMemory.

Three built-in profiles that configure brain behavior, maintenance,
and retrieval for different use cases. Presets are static dicts
(not a plugin system) to keep the surface simple and predictable.

Usage:
    from neural_memory.config_presets import get_preset, list_presets, apply_preset

    preset = get_preset("max-recall")
    apply_preset(config, preset)
    config.save()
"""

from __future__ import annotations

from typing import Any

# ── Preset definitions ────────────────────────────────────────────

SAFE_COST: dict[str, dict[str, Any]] = {
    "brain": {
        "decay_rate": 0.2,
        "reinforcement_delta": 0.03,
        "activation_threshold": 0.3,
        "max_spread_hops": 2,
        "max_context_tokens": 500,
        "freshness_weight": 0.1,
    },
    "maintenance": {
        "auto_consolidate": True,
        "check_interval": 15,
        "auto_consolidate_strategies": ["prune", "merge"],
        "consolidate_cooldown_minutes": 30,
    },
    "eternal": {
        "max_context_tokens": 64_000,
    },
}

BALANCED: dict[str, dict[str, Any]] = {
    "brain": {
        "decay_rate": 0.1,
        "reinforcement_delta": 0.05,
        "activation_threshold": 0.2,
        "max_spread_hops": 4,
        "max_context_tokens": 1500,
        "freshness_weight": 0.0,
    },
    "maintenance": {
        "auto_consolidate": True,
        "check_interval": 25,
        "auto_consolidate_strategies": ["prune", "merge"],
        "consolidate_cooldown_minutes": 60,
    },
    "eternal": {
        "max_context_tokens": 128_000,
    },
}

MAX_RECALL: dict[str, dict[str, Any]] = {
    "brain": {
        "decay_rate": 0.05,
        "reinforcement_delta": 0.08,
        "activation_threshold": 0.15,
        "max_spread_hops": 6,
        "max_context_tokens": 3000,
        "freshness_weight": 0.0,
    },
    "maintenance": {
        "auto_consolidate": True,
        "check_interval": 50,
        "auto_consolidate_strategies": ["prune", "merge", "enrich", "mature"],
        "consolidate_cooldown_minutes": 120,
    },
    "eternal": {
        "max_context_tokens": 200_000,
    },
}

_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "safe-cost": SAFE_COST,
    "balanced": BALANCED,
    "max-recall": MAX_RECALL,
}

_DESCRIPTIONS: dict[str, str] = {
    "safe-cost": "Lower token usage, faster decay, aggressive pruning",
    "balanced": "Default settings — good all-around performance",
    "max-recall": "Maximum retention, deeper retrieval, conservative pruning",
}


# ── Public API ────────────────────────────────────────────────────


def list_presets() -> list[dict[str, str]]:
    """Return list of available presets with descriptions."""
    return [{"name": name, "description": _DESCRIPTIONS[name]} for name in _PRESETS]


def get_preset(name: str) -> dict[str, dict[str, Any]] | None:
    """Get a preset by name. Returns None if not found."""
    return _PRESETS.get(name)


def compute_diff(
    config: Any,
    preset: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the diff between current config and a preset.

    Returns a list of change dicts: {section, key, current, new}.
    Only includes values that would actually change.
    """
    changes: list[dict[str, Any]] = []
    section_getters = {
        "brain": lambda c: c.brain.to_dict(),
        "maintenance": lambda c: c.maintenance.to_dict(),
        "eternal": lambda c: c.eternal.to_dict(),
    }

    for section, values in preset.items():
        getter = section_getters.get(section)
        if getter is None:
            continue
        current_dict = getter(config)
        for key, new_val in values.items():
            current_val = current_dict.get(key)
            if current_val != new_val:
                changes.append(
                    {
                        "section": section,
                        "key": key,
                        "current": current_val,
                        "new": new_val,
                    }
                )

    return changes


def apply_preset(
    config: Any,
    preset: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a preset to a UnifiedConfig instance.

    Merges preset values into the config's nested sections,
    preserving any values not specified in the preset.

    Returns the list of changes applied (same format as compute_diff).
    """
    from dataclasses import replace

    from neural_memory.unified_config import BrainSettings, EternalConfig, MaintenanceConfig

    changes = compute_diff(config, preset)

    if "brain" in preset:
        merged = {**config.brain.to_dict(), **preset["brain"]}
        config.brain = BrainSettings.from_dict(merged)

    if "maintenance" in preset:
        merged = {**config.maintenance.to_dict(), **preset["maintenance"]}
        config.maintenance = replace(
            config.maintenance,
            **{
                k: v
                for k, v in MaintenanceConfig.from_dict(merged).to_dict().items()
                if k in preset["maintenance"]
            },
        )
        # Full merge: rebuild from merged dict
        config.maintenance = MaintenanceConfig.from_dict(merged)

    if "eternal" in preset:
        merged = {**config.eternal.to_dict(), **preset["eternal"]}
        config.eternal = EternalConfig.from_dict(merged)

    return changes
