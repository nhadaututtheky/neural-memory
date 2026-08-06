"""Hooks + prompt hygiene — port of the hook section of integrations/neuralmemory/src/index.ts.

Faithful port: stripPromptMetadata and sanitizeAutoCapture carry the same
regex passes in the same order with the same fallback logic. Hook wiring is
adapted to Hermes hook names (documented in SPEC.md parity table).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from .config import MAX_AUTO_CAPTURE_CHARS, PluginConfig
from .mcp_client import NeuralMemoryMcpClient

logger = logging.getLogger("hermes.plugins.neuralmemory")

# ── Prompt metadata stripping (port of stripPromptMetadata) ──────
#
# Stripping order matters — later passes clean up residue from earlier ones.

_RE_JSON_META = re.compile(
    r'^\{[\s\S]*?"(?:conversation|message_id|sender_id|sender|chat_id|update_id)"[\s\S]*?\}$',
    re.MULTILINE,
)
_RE_NM_SECTIONS = re.compile(
    r'^#{1,3}\s*(?:Relevant Memories|Related Information|Relevant Context|Neural Memory)'
    r'[\s\S]*?(?=\n#{1,3}\s|\n\n(?![-•*\s])|$)',
    re.MULTILINE | re.IGNORECASE,
)
_RE_NEURON_BULLETS = re.compile(
    r'^-\s*\[(?:concept|entity|decision|error|preference|insight|memory|fact|workflow|instruction|pattern)\].*$',
    re.MULTILINE | re.IGNORECASE,
)
_RE_NM_WRAPPERS = re.compile(r'^\[NeuralMemory\s*[—–-].*\]$', re.MULTILINE)
_RE_META_LABELS = re.compile(
    r'^(?:Conversation info|Sender|Context|System)\s*\(.*?\)\s*:?\s*$',
    re.MULTILINE | re.IGNORECASE,
)
_RE_EXPORT_LINES = re.compile(r'^export\s+\w+=.*$', re.MULTILINE)


def strip_prompt_metadata(raw: str) -> str:
    """Port of stripPromptMetadata() — same 7 passes, same fallback."""
    cleaned = raw

    # 1. Remove JSON blocks (Telegram metadata, conversation info)
    cleaned = _RE_JSON_META.sub("", cleaned)
    # 2. Remove NeuralMemory context sections (## Relevant Memories, etc.)
    #    The |$ ensures sections at end-of-string are also stripped.
    cleaned = _RE_NM_SECTIONS.sub("", cleaned)
    # 3. Remove neuron-type bullet lines injected by NM context
    cleaned = _RE_NEURON_BULLETS.sub("", cleaned)
    # 4. Remove [NeuralMemory — ...] wrapper lines
    cleaned = _RE_NM_WRAPPERS.sub("", cleaned)
    # 5. Remove metadata labels (untrusted metadata lines)
    cleaned = _RE_META_LABELS.sub("", cleaned)
    # 6. Remove env/export lines
    cleaned = _RE_EXPORT_LINES.sub("", cleaned)
    # 7. Collapse whitespace runs
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # Fallback: if everything was stripped, use last non-empty line of raw
    if not cleaned:
        lines = [line for line in raw.split("\n") if line.strip()]
        cleaned = lines[-1].strip() if lines else raw.strip()

    return cleaned


# ── Auto-capture sanitization (port of sanitizeAutoCapture) ──────

_RE_NM_SECTION_HEADERS = re.compile(
    r'^#{1,3}\s*(?:Relevant Memories|Related Information|Relevant Context|Neural Memory)\b.*$',
    re.MULTILINE | re.IGNORECASE,
)
_RE_ACK_LINES = re.compile(
    r'^(?:OK|Sure|Done|Got it|Understood|Noted|Alright|I see|Thanks|Thank you|Okay)\.?\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def sanitize_auto_capture(raw: str) -> str:
    """Port of sanitizeAutoCapture() — defense-in-depth before nmem_auto."""
    cleaned = raw

    # Strip NM context section headers
    cleaned = _RE_NM_SECTION_HEADERS.sub("", cleaned)
    # Strip [NeuralMemory — ...] wrapper lines
    cleaned = _RE_NM_WRAPPERS.sub("", cleaned)
    # Strip neuron-type bullet lines (- [concept] ..., - [error] ...)
    cleaned = _RE_NEURON_BULLETS.sub("", cleaned)
    # Strip metadata labels
    cleaned = re.sub(
        r'^(?:Conversation info|Sender|Context)\s*\(.*?\)\s*:?\s*$',
        "", cleaned, flags=re.MULTILINE | re.IGNORECASE,
    )
    # Strip short acknowledgement lines (< 20 chars, common filler)
    cleaned = _RE_ACK_LINES.sub("", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return cleaned


# ── Tool awareness block (port of buildToolInstructions) ─────────

def build_tool_instructions(tools: list[dict]) -> str:
    """Port of buildToolInstructions() — tool list built from registered tools."""
    tool_list = "\n".join(
        f"- {t['name']}: {t['description'][:100]}" for t in tools
    )

    return (
        "Neural Memory gives you persistent memory across sessions. Use it proactively — "
        "each session starts fresh, so without explicit saves ALL discoveries are lost.\n\n"
        "These are TOOL CALLS, not CLI commands. Do NOT run \"nmem remember\" in terminal.\n\n"
        "## Available Tools\n"
        f"{tool_list}\n\n"
        "nmem_* is your primary memory system. memory_search/memory_get are legacy aliases for nmem_recall.\n\n"
        "## WHEN TO RECALL\n"
        "- New session starts → nmem_recall(\"current project context\")\n"
        "- User references past event → nmem_recall(\"<that topic>\")\n"
        "- Prefix queries with project name for precision\n\n"
        "## WHEN TO SAVE\n"
        "After each task: did you make a decision (type=\"decision\", priority=7), fix a bug "
        "(type=\"error\", priority=7), learn a preference (type=\"preference\", priority=8), "
        "or discover an insight (type=\"insight\", priority=6)?\n\n"
        "Save with: nmem_remember(content=\"Chose X over Y because Z\", type=\"decision\", "
        "priority=7, tags=[\"project\", \"topic\"])\n\n"
        "## CONTENT QUALITY\n"
        "- Max 1-3 sentences. Use causal language: \"Chose X because Y\", \"Root cause was X, fixed by Y\".\n"
        "- Always include project name + topic in tags (lowercase).\n"
        "- For temporary scratch notes: nmem_remember(content=\"...\", ephemeral=true) — auto-expires, never synced.\n\n"
        "## SESSION END\n"
        "nmem_auto(action=\"process\", text=\"<brief session summary>\")\n\n"
        "## COMPACT MODE\n"
        "All tools support compact=true (saves 60-80% tokens) and token_budget=N."
    )


# ── Hook factories ──────────────────────────────

def make_pre_llm_hook(mcp: NeuralMemoryMcpClient, cfg: PluginConfig,
                      tools: list[dict]) -> Callable[..., Any]:
    """Port of the before_prompt_build hook.

    Adaptation (SPEC.md row 7): OpenClaw appends tool instructions to the
    SYSTEM prompt on every prompt build; Hermes pre_llm_call injects into the
    USER message (prompt-cache preserving). Instructions are therefore
    injected on the first turn of each session only (same intent: fresh
    session awareness, survives /new), and auto-recall context on every turn.
    """
    instructions = build_tool_instructions(tools)

    def hook(session_id: str = "", user_message: str = "",
             conversation_history: list | None = None,
             is_first_turn: bool = False, model: str = "", platform: str = "",
             **kwargs: Any) -> dict | None:
        parts: list[str] = []

        if is_first_turn:
            parts.append(instructions)

        if cfg.auto_context and mcp.connected:
            try:
                query = strip_prompt_metadata(user_message)
                raw = mcp.call_tool("nmem_recall", {
                    "query": query,
                    "depth": cfg.context_depth,
                    "max_tokens": cfg.max_context_tokens,
                    "clean_for_prompt": True,
                })
                data = json.loads(raw) if raw else {}
                if isinstance(data, dict) and data.get("answer") and (data.get("confidence") or 0) > 0.1:
                    parts.append(f"[NeuralMemory — relevant context]\n{data['answer']}")
            except Exception as err:  # noqa: BLE001 — recall must never break a turn
                logger.warning("Auto-context failed: %s", err)

        if not parts:
            return None
        return {"context": "\n\n".join(parts)}

    return hook


def make_post_llm_hook(mcp: NeuralMemoryMcpClient,
                       cfg: PluginConfig) -> Callable[..., None]:
    """Port of the agent_end hook → Hermes post_llm_call.

    post_llm_call fires only on successful turns (documented: does not fire
    on interruption), matching the upstream `if (!ev.success) return` guard.
    """

    def hook(session_id: str = "", user_message: str = "",
             assistant_response: str = "", conversation_history: list | None = None,
             model: str = "", platform: str = "", **kwargs: Any) -> None:
        if not mcp.connected:
            return
        try:
            # Last 5 messages, assistant-role string contents only (upstream behavior)
            messages = (conversation_history or [])[-5:]
            contents: list[str] = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "assistant" \
                        and isinstance(m.get("content"), str):
                    contents.append(m["content"])
            if not contents and assistant_response:
                contents = [assistant_response]
            raw_text = "\n".join(contents)[:MAX_AUTO_CAPTURE_CHARS]

            # Strip NM context noise and short acknowledgements before re-ingest
            text = sanitize_auto_capture(raw_text)

            if len(text) > 50:
                mcp.call_tool("nmem_auto", {"action": "process", "text": text})
        except Exception as err:  # noqa: BLE001
            logger.warning("Auto-capture failed: %s", err)

    return hook


def make_session_reset_hook(mcp: NeuralMemoryMcpClient,
                           cfg: PluginConfig) -> Callable[..., None]:
    """Port of the before_reset hook → Hermes on_session_reset (flush at session boundary)."""

    def hook(**kwargs: Any) -> None:
        if not mcp.connected:
            return
        try:
            mcp.call_tool("nmem_auto", {"action": "process",
                                        "text": "[session boundary — reset]"})
        except Exception as err:  # noqa: BLE001
            logger.warning("Session boundary flush failed: %s", err)

    return hook
