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

logger = logging.getLogger("hermes.plugins.neuralmemory")

# Hermes plugin context object (duck-typed per the plugin contract).


def register(ctx) -> None:
    """Hermes plugin entry point — called once at plugin load (sync)."""
    from .config import load_plugin_config
    from .mcp_client import NeuralMemoryMcpClient
    from .tools_proxy import create_compatibility_tools, create_fallback_tools, create_tools_from_mcp
    from .hooks import (make_post_llm_hook, make_pre_llm_hook,
                        make_session_reset_hook)

    cfg = load_plugin_config()

    # ── Singleton MCP client (port of getOrCreateMcpClient) ──
    # Hermes is multi-threaded; use the official thread-safe helper when
    # available, else a locked fallback.
    def _build_client() -> NeuralMemoryMcpClient:
        return NeuralMemoryMcpClient(
            python_path=cfg.python_path,
            brain=cfg.brain,
            timeout=cfg.timeout,
            init_timeout=cfg.init_timeout,
        )

    try:
        from plugins.plugin_utils import lazy_singleton
        get_client = lazy_singleton(_build_client)
        mcp = get_client()
    except Exception:  # noqa: BLE001 — plugin_utils unavailable outside Hermes core
        mcp = _build_client()

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
            except Exception as err:  # noqa: BLE001
                logger.warning("Tool discovery failed: %s", err)
        except Exception as err:  # noqa: BLE001
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

    logger.info(
        "NeuralMemory registered (brain: %s, autoContext: %s, autoCapture: %s, "
        "autoFlush: %s) — MCP connects in background; tools auto-reconnect on first call",
        cfg.brain, cfg.auto_context, cfg.auto_capture, cfg.auto_flush,
    )
