"""MCP handler mixin for tool result offload tools.

Phase 1 of plan-agent-ergonomics: reduces context bloat by storing large
tool results as ephemeral neurons (24h TTL) and returning a compact ref +
summary that the agent can drill into via ``nmem_inflate`` when needed.

No LLM calls, no compression — pure store/lookup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.token_budget import TOKEN_RATIO
from neural_memory.mcp.tool_handler_utils import _get_brain_or_error

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Hard cap on offload content — same ceiling as nmem_remember to keep storage sane
_MAX_CONTENT_LEN = 100_000

# Caps on caller-controlled string fields (handler-side, schema is only advisory)
_MAX_TOOL_NAME_LEN = 100
_MAX_EXPLICIT_SUMMARY_LEN = 500

# Preview length for auto-generated summaries
_SUMMARY_PREVIEW_LEN = 200

# Hard cap on the final summary string returned to caller — keeps the
# offload contract ("summary is small") true even with long tool_names.
_MAX_SUMMARY_LEN = 300


def _estimate_tokens(content: str) -> int:
    """Rough token estimate — uses whichever of (words x ratio) or (chars / 4) is larger.

    The dual estimate guards against pathological inputs (e.g. a 5000-char run
    of identical bytes with no whitespace) where word count under-reports cost.
    """
    words = len(content.split())
    word_based = int(words * TOKEN_RATIO)
    char_based = len(content) // 4  # ~4 chars/token rule of thumb for English
    return max(1, word_based, char_based)


def _build_summary(content: str, tool_name: str) -> str:
    """Generate a compact preview + size hint for an offloaded payload.

    Output is hard-capped at ``_MAX_SUMMARY_LEN`` to keep the offload
    contract honest regardless of tool_name length.
    """
    preview = content[:_SUMMARY_PREVIEW_LEN].replace("\n", " ").strip()
    if len(content) > _SUMMARY_PREVIEW_LEN:
        preview += "…"
    line_count = content.count("\n") + 1
    byte_count = len(content)
    summary = f"[{tool_name}] {preview} (~{line_count} lines, {byte_count}B)"
    if len(summary) > _MAX_SUMMARY_LEN:
        summary = summary[: _MAX_SUMMARY_LEN - 1] + "…"
    return summary


class OffloadHandler:
    """Mixin: tool result offload + inflate tools."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _offload(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a large tool result as an ephemeral neuron, return a compact ref.

        Args:
            content: Raw tool output to offload (required, ≤100k chars). The
                content is sanitized for prompt-injection markers and run
                through the auto-redactor before storage (same pipeline as
                nmem_remember) so leaked secrets in tool output are scrubbed.
            tool_name: Name of the tool that produced the output (required,
                truncated to 100 chars).
            summary: Caller-provided summary (optional; auto-generated if
                absent, max 500 chars).
            (ttl is fixed at 24h via the ephemeral expiry handler — no
            ttl_hours arg is accepted.)

        Returns:
            ``{ref_id, summary, token_saved, redacted}`` on success,
            ``{error}`` on failure. ``redacted`` is True when sensitive
            content was scrubbed.
        """
        content = args.get("content")
        tool_name_raw = args.get("tool_name") or "unknown"
        tool_name = str(tool_name_raw)[:_MAX_TOOL_NAME_LEN]
        explicit_summary_raw = args.get("summary")
        explicit_summary = (
            str(explicit_summary_raw)[:_MAX_EXPLICIT_SUMMARY_LEN]
            if isinstance(explicit_summary_raw, str)
            else None
        )

        if not content or not isinstance(content, str):
            return {"error": "content is required and must be a non-empty string"}
        if len(content) > _MAX_CONTENT_LEN:
            return {"error": f"Content too long ({len(content)} chars). Max: {_MAX_CONTENT_LEN}."}

        try:
            storage = await self.get_storage()
            _brain, err = await _get_brain_or_error(storage)
            if err:
                return err

            # Defense in depth — tool output is a common vector for accidental
            # secret capture (API keys in grep, tokens in curl logs, etc).
            # Mirror the remember_handler safety pipeline.
            from neural_memory.safety.input_firewall import sanitize_explicit_content
            from neural_memory.safety.sensitive import auto_redact_content

            content = sanitize_explicit_content(content)
            try:
                redact_severity = int(self.config.safety.auto_redact_min_severity)
            except (TypeError, ValueError, AttributeError):
                redact_severity = 3
            redacted_content, redacted_matches, _hash = auto_redact_content(
                content, min_severity=redact_severity
            )
            redacted = bool(redacted_matches)
            if redacted:
                content = redacted_content
                logger.info(
                    "nmem_offload: auto-redacted %d sensitive matches for tool=%s",
                    len(redacted_matches),
                    tool_name,
                )

            summary = explicit_summary or _build_summary(content, tool_name)
            token_estimate = _estimate_tokens(content)
            summary_tokens = _estimate_tokens(summary)
            token_saved = max(0, token_estimate - summary_tokens)

            neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=content,
                metadata={
                    "_source": "tool_offload",
                    "_tool_name": tool_name,
                    "_summary": summary,
                    "_offload_token_estimate": token_estimate,
                    "_offload_redacted": redacted,
                },
                ephemeral=True,
            )
            await storage.add_neuron(neuron)

            return {
                "ref_id": neuron.id,
                "summary": summary,
                "token_saved": token_saved,
                "redacted": redacted,
            }
        except Exception:
            logger.error("Offload failed for tool=%s", tool_name, exc_info=True)
            return {"error": "Offload failed"}

    async def _inflate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Retrieve full content of a previously offloaded neuron by ref_id.

        Args:
            ref_id: Neuron ID returned by ``nmem_offload`` (required)

        Returns:
            ``{content, tool_name, summary}`` on success, ``{error}`` on failure.
        """
        ref_id = args.get("ref_id")
        if not ref_id or not isinstance(ref_id, str):
            return {"error": "ref_id is required and must be a string"}

        try:
            storage = await self.get_storage()
            neuron = await storage.get_neuron(ref_id)
            if neuron is None:
                return {"error": f"ref_id not found or expired: {ref_id}"}

            meta = neuron.metadata or {}
            if meta.get("_source") != "tool_offload":
                # Don't allow inflate to peek at arbitrary neurons — only offload payloads.
                return {"error": f"ref_id {ref_id} is not an offloaded payload"}

            return {
                "content": neuron.content,
                "tool_name": meta.get("_tool_name", "unknown"),
                "summary": meta.get("_summary", ""),
            }
        except Exception:
            logger.error("Inflate failed for ref_id=%s", ref_id, exc_info=True)
            return {"error": "Inflate failed"}
