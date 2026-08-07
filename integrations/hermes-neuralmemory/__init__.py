"""NeuralMemory — Hermes Memory Plugin.

Faithful port of the OpenClaw NeuralMemory plugin
(integrations/neuralmemory/, v1.17.0) to the Hermes plugin API.

Brain-inspired persistent memory for AI agents. Architecture:

  Hermes ←→ Plugin (Python) ←→ MCP stdio ←→ NeuralMemory (Python)

Port notes (full parity table in SPEC.md / README.md):
  - 5 fallback tools + 2 compat shims registered synchronously at
    register() time (Hermes freezes the tool list after register()).
  - Dynamic tools/list discovery runs at startup and is LOGGED
    (OpenClaw also freezes tools after register; same constraint).
  - before_prompt_build  → pre_llm_call   (context injection)
  - agent_end            → post_llm_call  (auto-capture)
  - before_reset         → on_session_reset (boundary flush)
  - before_compaction / gateway_start: no Hermes plugin-hook equivalent
    (documented, dropped)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcp_client import NeuralMemoryMcpClient

logger = logging.getLogger("hermes.plugins.neuralmemory")

# ── Singleton MCP client pool (port of the mcpClients Map in index.ts) ──
# Multiple register() calls share one connected client per (pythonPath, brain).
_mcp_clients: dict[str, NeuralMemoryMcpClient] = {}
_mcp_clients_lock = threading.Lock()


def _get_or_create_mcp_client(cfg) -> NeuralMemoryMcpClient:
    """Port of getOrCreateMcpClient(): keyed pool, thread-safe double-check."""
    from .mcp_client import NeuralMemoryMcpClient

    key = f"{cfg.python_path}::{cfg.brain}"
    existing = _mcp_clients.get(key)
    if existing is not None:
        logger.debug('Reusing existing MCP client for brain "%s"', cfg.brain)
        return existing
    with _mcp_clients_lock:
        existing = _mcp_clients.get(key)
        if existing is not None:
            return existing
        client = NeuralMemoryMcpClient(
            python_path=cfg.python_path,
            brain=cfg.brain,
            timeout=cfg.timeout,
            init_timeout=cfg.init_timeout,
        )
        _mcp_clients[key] = client
        return client

# Hermes plugin context object (duck-typed per the plugin contract).


def register(ctx) -> None:
    """Hermes plugin entry point — called once at plugin load (sync)."""
    from .config import load_plugin_config
    from .hooks import (
        make_post_llm_hook,
        make_pre_llm_hook,
        make_session_end_hook,
        make_session_reset_hook,
    )
    from .tools_proxy import (
        create_compatibility_tools,
        create_fallback_tools,
        create_tools_from_mcp,
    )

    cfg = load_plugin_config()

    # ── Singleton MCP client pool (port of getOrCreateMcpClient) ──
    # Module-level pool keyed by (pythonPath, brain), exactly mirroring the
    # upstream mcpClients Map. A bare lazy_singleton(_build_client) would be
    # keyed per factory CLOSURE — a new singleton on every register() call,
    # spawning duplicate MCP processes (reviewer finding M3).
    mcp = _get_or_create_mcp_client(cfg)

    # ── Register fallback + compat tools synchronously ─────────
    # Port of OpenClaw's sync register() + deferred MCP connection.
    # Fallback tools auto-reconnect MCP on first call.
    fallback_tools = create_fallback_tools(mcp)
    compat_tools = create_compatibility_tools(mcp)

    for t in fallback_tools + compat_tools:
        ctx.register_tool(
            name=t["name"],
            toolset="neuralmemory",
            schema={"name": t["name"], "description": t["description"],
                    "parameters": t["parameters"]},
            handler=t["handler"],
            description=t["description"],
        )

    logger.info(
        "Registered %d NeuralMemory tools + %d compat shims (sync)",
        len(fallback_tools), len(compat_tools),
    )

    # ── Background connect + dynamic tool discovery ────────────
    # Port of service.start(): connect MCP, then log discovered tools.
    # Daemon thread so plugin load never blocks Hermes startup.
    def _startup() -> None:
        try:
            mcp.ensure_connected()
            logger.info("NeuralMemory MCP connected at startup")
            try:
                dynamic = create_tools_from_mcp(mcp)
                logger.info("NeuralMemory MCP discovered %d tools", len(dynamic))
            except Exception as err:
                logger.warning("Tool discovery failed: %s", err)
        except Exception as err:
            logger.warning("NeuralMemory MCP startup connect failed: %s "
                           "(tools auto-reconnect on first call)", err)

    threading.Thread(target=_startup, daemon=True,
                     name="nmem-startup").start()

    # ── Hooks ──────────────────────────────────────

    # Tool awareness + auto-context before each turn
    # (port of before_prompt_build → Hermes pre_llm_call)
    ctx.register_hook("pre_llm_call", make_pre_llm_hook(mcp, cfg, fallback_tools))

    # Auto-capture after successful turns
    # (port of agent_end → Hermes post_llm_call)
    if cfg.auto_capture:
        ctx.register_hook("post_llm_call", make_post_llm_hook(mcp, cfg))

    # Flush at session boundary (/new and /reset)
    # (port of before_reset → Hermes on_session_reset)
    if cfg.auto_flush:
        ctx.register_hook("on_session_reset", make_session_reset_hook(mcp, cfg))

    # MCP process teardown at session end
    # (port of service.stop() → Hermes on_session_end; also evicts the pool entry
    # so a later register() builds a fresh client — reviewer MINOR-2)
    ctx.register_hook("on_session_end", make_session_end_hook(mcp, cfg))

    logger.info(
        "NeuralMemory registered (brain: %s, autoContext: %s, autoCapture: %s, "
        "autoFlush: %s) — MCP connects in background; tools auto-reconnect on first call",
        cfg.brain, cfg.auto_context, cfg.auto_capture, cfg.auto_flush,
    )
