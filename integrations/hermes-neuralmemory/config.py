"""Plugin configuration — port of the config section of integrations/neuralmemory/src/index.ts.

Faithful port: same defaults, same validation ranges, same regex.
Values are read from Hermes config.yaml: plugins.entries.neuralmemory
(env overrides kept for the MCP child process, matching upstream).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BRAIN_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-.]{1,64}$")
MAX_AUTO_CAPTURE_CHARS = 50_000


@dataclass(frozen=True)
class PluginConfig:
    python_path: str
    brain: str
    auto_context: bool
    auto_capture: bool
    auto_flush: bool
    auto_consolidate: bool
    context_depth: int
    max_context_tokens: int
    timeout: int
    init_timeout: int


DEFAULT_CONFIG = PluginConfig(
    python_path="python",
    brain="default",
    auto_context=True,
    auto_capture=True,
    auto_flush=True,
    auto_consolidate=True,
    context_depth=1,
    max_context_tokens=500,
    timeout=30_000,
    init_timeout=90_000,
)


def resolve_config(raw: dict | None) -> PluginConfig:
    """Merge raw config dict over defaults with the same validation as upstream."""
    merged = {**(raw or {})}

    def _str(key: str, default: str) -> str:
        v = merged.get(key)
        return v if isinstance(v, str) and v else default

    def _bool(key: str, default: bool) -> bool:
        v = merged.get(key)
        return v if isinstance(v, bool) else default

    def _int_range(key: str, default: int, lo: int, hi: int) -> int:
        v = merged.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
            return v
        return default

    def _num_range(key: str, default: int, lo: int, hi: int) -> int:
        v = merged.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                import math
                if math.isfinite(v) and lo <= v <= hi:
                    return int(v)
            except (TypeError, ValueError):
                pass
        return default

    brain = merged.get("brain")
    if not (isinstance(brain, str) and BRAIN_NAME_RE.match(brain)):
        brain = DEFAULT_CONFIG.brain

    return PluginConfig(
        python_path=_str("pythonPath", DEFAULT_CONFIG.python_path),
        brain=brain,
        auto_context=_bool("autoContext", DEFAULT_CONFIG.auto_context),
        auto_capture=_bool("autoCapture", DEFAULT_CONFIG.auto_capture),
        auto_flush=_bool("autoFlush", DEFAULT_CONFIG.auto_flush),
        auto_consolidate=_bool("autoConsolidate", DEFAULT_CONFIG.auto_consolidate),
        context_depth=_int_range("contextDepth", DEFAULT_CONFIG.context_depth, 0, 3),
        max_context_tokens=_int_range("maxContextTokens", DEFAULT_CONFIG.max_context_tokens, 100, 10_000),
        timeout=_num_range("timeout", DEFAULT_CONFIG.timeout, 5_000, 120_000),
        init_timeout=_num_range("initTimeout", DEFAULT_CONFIG.init_timeout, 10_000, 300_000),
    )


def load_plugin_config() -> PluginConfig:
    """Load config from plugins.entries.neuralmemory in Hermes config.yaml.

    Falls back to env vars (NEURALMEMORY_BRAIN) then defaults — mirrors how
    the OpenClaw plugin reads api.pluginConfig.
    """
    import os

    raw: dict = {}
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get("neuralmemory") or {}
        raw = dict(entry.get("config") or {})
    except Exception:
        # Fail open to defaults — a broken config read must never crash the agent.
        raw = {}

    if os.environ.get("NEURALMEMORY_BRAIN") and "brain" not in raw:
        raw["brain"] = os.environ["NEURALMEMORY_BRAIN"]

    return resolve_config(raw)
