"""Stop hook: capture memories when a Claude Code session ends.

Called by Claude Code when the session terminates normally.
Reads the conversation transcript, detects memorable content,
and saves it to the brain — preventing memory loss on session end.

Unlike the PreCompact hook (which uses emergency/aggressive settings),
this hook runs at normal confidence thresholds since there is no urgency.

Usage as Claude Code hook:
    Receives JSON on stdin with `transcript_path` field.
    Outputs status to stderr (stdout reserved for hook response).

Usage standalone:
    echo '{"transcript_path": "/path/to/transcript.jsonl"}' | python -m neural_memory.hooks.stop
    python -m neural_memory.hooks.stop --transcript /path/to/transcript.jsonl
    python -m neural_memory.hooks.stop --text "Some text to capture"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max lines to read from transcript tail (increased for longer sessions)
MAX_TRANSCRIPT_LINES = 150
# Max characters to process
MAX_CAPTURE_CHARS = 100_000
# Normal confidence threshold (not emergency)
DEFAULT_CONFIDENCE = 0.7
# Max chars for session summary extraction
_SUMMARY_TAIL_CHARS = 8_000


def read_hook_input() -> dict[str, Any]:
    """Read Claude Code hook JSON from stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def _get_entry_role(entry: dict[str, Any]) -> str:
    """Extract the role from a JSONL transcript entry.

    Claude Code transcript entries may nest role in different places:
    - Top-level: {"role": "user", "content": ...}
    - Nested: {"message": {"role": "assistant", ...}}
    - Tool results: {"type": "tool_result", ...} or role == "tool"

    Returns:
        One of "user", "assistant", or "tool".
    """
    # Direct role field
    role: str = str(entry.get("role", ""))
    if role in ("user", "assistant", "tool"):
        return role

    # Nested message object
    message = entry.get("message")
    if isinstance(message, dict):
        role = str(message.get("role", ""))
        if role in ("user", "assistant", "tool"):
            return role

    # Tool result heuristic: type field or tool_use_id present
    if entry.get("type") in ("tool_result", "tool_use"):
        return "tool"
    if "tool_use_id" in entry:
        return "tool"

    # Content list with tool_use blocks → assistant (but tool-heavy, skip)
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("tool_use", "tool_result"):
                return "tool"

    # Default: treat as user (will go through text extraction + length filter)
    return "user"


# Patterns that indicate memory-worthy assistant output.
# These match the same signals auto_capture.py detects, plus common
# session-end patterns like summaries and status updates.
_MEMORY_MARKER_RE = re.compile(
    r"(?i)"
    r"(?:decided|chose|selected|opted|switched|migrated)"
    r"|(?:root cause|bug|fixed|resolved|solved|workaround)"
    r"|(?:lesson learned|takeaway|key insight|turns out|realized|discovered)"
    r"|(?:TODO|FIXME|need to|should|must|remember to)"
    r"|(?:prefer|always use|never use|don't use|avoid)"
    r"|(?:quyết định|chọn|lỗi|sửa|bài học|hóa ra|cần phải)"
    r"|(?:saved|committed|pushed|deployed|released|shipped)"
    r"|(?:v\d+\.\d+)"
)


def _has_memory_markers(text: str) -> bool:
    """Check if assistant text contains explicit memory-worthy markers.

    Returns True if the text mentions decisions, errors, insights,
    preferences, TODOs, or session-end signals worth capturing.
    """
    return bool(_MEMORY_MARKER_RE.search(text))


def read_transcript_tail(transcript_path: str, max_lines: int = MAX_TRANSCRIPT_LINES) -> str:
    """Read the last N entries from a JSONL transcript and extract text content.

    Applies role-based filtering: user messages get full capture,
    assistant messages are only included if they contain explicit
    memory markers. Tool results are skipped entirely.
    """
    path = Path(transcript_path)
    if not path.exists() or not path.is_file():
        return ""

    lines: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-max_lines:]

        for raw_line in tail:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
                role = _get_entry_role(entry)

                # Skip tool results entirely — data, not decisions
                if role == "tool":
                    continue

                text = _extract_text(entry)
                if not text or len(text) <= 20:
                    continue

                # Assistant messages: only include if they contain
                # explicit memory markers (decisions, root causes, etc.)
                if role == "assistant" and not _has_memory_markers(text):
                    continue

                lines.append(text)
            except json.JSONDecodeError:
                continue
    except OSError:
        return ""

    joined = "\n\n".join(lines)
    if len(joined) > MAX_CAPTURE_CHARS:
        joined = joined[-MAX_CAPTURE_CHARS:]
    return joined


def _extract_text(entry: dict[str, Any]) -> str:
    """Extract text content from a transcript entry."""
    content = entry.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    message = entry.get("message")
    if isinstance(message, dict):
        return _extract_text(message)

    text = entry.get("text", "")
    return text if isinstance(text, str) else ""


def _extract_session_summary(text: str) -> str:
    """Extract a brief session summary from transcript tail.

    Looks for common session-end signals (decisions, completions, errors fixed)
    and produces a 1-3 sentence summary. Falls back to a generic summary
    from the last portion of the transcript.

    Args:
        text: Full transcript text.

    Returns:
        A short summary string, or empty string if nothing meaningful found.
    """
    tail = text[-_SUMMARY_TAIL_CHARS:] if len(text) > _SUMMARY_TAIL_CHARS else text
    lines = [line.strip() for line in tail.split("\n") if line.strip() and len(line.strip()) > 15]
    if not lines:
        return ""

    # Take last ~10 meaningful lines and join as summary context
    summary_lines = lines[-10:]
    summary = " ".join(summary_lines)

    # Truncate to a reasonable length
    if len(summary) > 500:
        summary = summary[:500]

    return f"Session activity: {summary}"


async def _embedding_dedup(
    items: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Remove semantic near-duplicates using embedding cosine similarity.

    Effective for Vietnamese text where different word forms produce
    different simhashes but carry the same meaning.

    Falls back gracefully (returns original list) if no embedding
    provider is available.

    Args:
        items: Detected memory candidates from analyze_text_for_memories().
        similarity_threshold: Pairs above this similarity are considered duplicates.

    Returns:
        Filtered list with semantic duplicates removed (keeps higher-confidence item).
    """
    if len(items) <= 1:
        return items

    try:
        from neural_memory.engine.semantic_discovery import _auto_detect_provider

        provider_name, model_name = _auto_detect_provider()
    except (RuntimeError, Exception):
        logger.debug("No embedding provider available, skipping semantic dedup")
        return items

    try:
        from neural_memory.engine.embedding.provider import EmbeddingProvider

        embed_provider: EmbeddingProvider
        if provider_name == "sentence_transformer":
            from neural_memory.engine.embedding.sentence_transformer import (
                SentenceTransformerEmbedding,
            )

            embed_provider = SentenceTransformerEmbedding(model_name=model_name)
        elif provider_name == "ollama":
            from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

            embed_provider = OllamaEmbedding(model=model_name)
        else:
            # Skip API-based providers in stop hook (rate limits, latency)
            logger.debug("Skipping API-based embedding provider %s in stop hook", provider_name)
            return items

        contents = [item["content"] for item in items]
        embeddings = await embed_provider.embed_batch(contents)

        # Mark indices to remove (keep higher-confidence item in each duplicate pair)
        remove: set[int] = set()
        for i in range(len(embeddings)):
            if i in remove:
                continue
            for j in range(i + 1, len(embeddings)):
                if j in remove:
                    continue
                sim = await embed_provider.similarity(embeddings[i], embeddings[j])
                if sim >= similarity_threshold:
                    # Remove the lower-confidence candidate
                    if items[i]["confidence"] >= items[j]["confidence"]:
                        remove.add(j)
                    else:
                        remove.add(i)
                        break  # i is removed, stop comparing

        filtered = [item for idx, item in enumerate(items) if idx not in remove]
        if len(filtered) < len(items):
            logger.debug(
                "Embedding dedup removed %d/%d candidates",
                len(items) - len(filtered),
                len(items),
            )
        return filtered
    except Exception:
        logger.debug("Embedding dedup failed, using original list", exc_info=True)
        return items


async def capture_text(text: str) -> dict[str, Any]:
    """Detect and save memorable content from session transcript.

    Uses normal (non-emergency) confidence thresholds since this runs
    at clean session end, not under compaction pressure.

    Always saves at least a session summary (type=context) even if no
    specific patterns are detected, ensuring every session leaves a trace.
    """
    from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.mcp.auto_capture import analyze_text_for_memories
    from neural_memory.safety.input_firewall import check_content

    # Gate 1: Input firewall — block garbage/adversarial content
    fw = check_content(text)
    if fw.blocked:
        logger.debug("Stop hook: input firewall blocked — %s", fw.reason)
        return {"saved": 0, "message": f"Input blocked: {fw.reason}"}
    if fw.sanitized:
        text = fw.sanitized
    from neural_memory.safety.sensitive import auto_redact_content
    from neural_memory.unified_config import get_config, get_shared_storage
    from neural_memory.utils.timeutils import utcnow

    config = get_config()
    storage = await get_shared_storage(config.current_brain)

    try:
        detected = analyze_text_for_memories(
            text,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            capture_preferences=True,
        )

        # Use configured threshold (or DEFAULT_CONFIDENCE as floor)
        threshold = max(config.auto.min_confidence, DEFAULT_CONFIDENCE)
        eligible = (
            [item for item in detected if item["confidence"] >= threshold] if detected else []
        )

        # Embedding-based semantic dedup (graceful fallback if unavailable)
        if len(eligible) > 1:
            eligible = await _embedding_dedup(eligible)

        brain = await storage.get_brain(config.current_brain)
        if not brain:
            return {"error": "No brain configured", "saved": 0}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        auto_redact_severity = config.safety.auto_redact_min_severity
        saved: list[str] = []

        # Write gate check for stop hook (auto-capture path)
        write_gate_cfg = config.write_gate
        gate_enabled = write_gate_cfg.enabled

        for item in eligible:
            try:
                content = item["content"]

                # Apply write gate if enabled (uses auto_capture threshold)
                if gate_enabled:
                    from neural_memory.engine.quality_scorer import check_write_gate

                    gate_result = check_write_gate(
                        content,
                        gate_config=write_gate_cfg,
                        is_auto_capture=True,
                        memory_type=item.get("type"),
                    )
                    if gate_result.rejected:
                        logger.debug(
                            "Stop hook write gate rejected: %s",
                            gate_result.rejection_reason,
                        )
                        continue

                redacted_content, matches, _ = auto_redact_content(
                    content, min_severity=auto_redact_severity
                )
                if matches:
                    logger.debug("Auto-redacted %d matches in stop-hook memory", len(matches))

                result = await encoder.encode(
                    content=redacted_content,
                    timestamp=utcnow(),
                    tags={"stop_hook", "session_end"},
                )

                mem_type_str = item.get("type", "fact")
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    mem_type = MemoryType.FACT

                typed_mem = TypedMemory.create(
                    fiber_id=result.fiber.id,
                    memory_type=mem_type,
                    priority=Priority.from_int(item.get("priority", 5)),
                    source="stop_hook",
                    tags={"stop_hook", "session_end"},
                )
                await storage.add_typed_memory(typed_mem)
                saved.append(redacted_content[:60])
            except Exception:
                logger.debug("Failed to save stop-hook memory", exc_info=True)
                continue

        # Always save a session summary if no patterns were detected
        if not saved:
            summary = _extract_session_summary(text)
            if summary and len(summary) > 30:
                # Apply write gate to session summary too
                if gate_enabled:
                    from neural_memory.engine.quality_scorer import check_write_gate

                    gate_result = check_write_gate(
                        summary,
                        gate_config=write_gate_cfg,
                        is_auto_capture=True,
                        memory_type="context",
                    )
                    if gate_result.rejected:
                        logger.debug(
                            "Stop hook session summary rejected: %s",
                            gate_result.rejection_reason,
                        )
                        summary = None  # type: ignore[assignment]

                if summary:
                    try:
                        redacted_summary, _, _ = auto_redact_content(
                            summary, min_severity=auto_redact_severity
                        )
                        result = await encoder.encode(
                            content=redacted_summary,
                            timestamp=utcnow(),
                            tags={"stop_hook", "session_end", "session_summary"},
                        )
                        typed_mem = TypedMemory.create(
                            fiber_id=result.fiber.id,
                            memory_type=MemoryType.CONTEXT,
                            priority=Priority.from_int(4),
                            source="stop_hook",
                            tags={"stop_hook", "session_end", "session_summary"},
                        )
                        await storage.add_typed_memory(typed_mem)
                        saved.append(redacted_summary[:60])
                    except Exception:
                        logger.debug("Failed to save session summary", exc_info=True)

        await storage.batch_save()

        return {
            "saved": len(saved),
            "memories": saved,
            "message": f"Session end: captured {len(saved)} memories"
            if saved
            else "No memories saved",
        }
    finally:
        await storage.close()


def main() -> None:
    """Entry point for Stop hook or standalone CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuralMemory Stop hook — capture memories on session end"
    )
    parser.add_argument("--transcript", "-t", help="Path to JSONL transcript file")
    parser.add_argument("--text", help="Direct text to capture (alternative to transcript)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    text = ""

    if args.text:
        text = args.text
    elif args.transcript:
        text = read_transcript_tail(args.transcript)
    else:
        hook_input = read_hook_input()
        transcript_path = hook_input.get("transcript_path", "")
        if transcript_path:
            # Validate stdin transcript path is within Claude data directory
            resolved = Path(transcript_path).resolve()
            allowed_dir = Path.home().resolve() / ".claude"
            if not resolved.is_relative_to(allowed_dir):
                logger.warning("Transcript path outside allowed directory, skipping")
                sys.exit(0)
            text = read_transcript_tail(transcript_path)
        else:
            sys.exit(0)

    if not text or len(text.strip()) < 50:
        print("No substantial content to capture", file=sys.stderr)
        sys.exit(0)

    try:
        result = asyncio.run(capture_text(text))
        saved = result.get("saved", 0)
        if saved > 0:
            print(
                f"[NeuralMemory] Session end: captured {saved} memories",
                file=sys.stderr,
            )
        else:
            print(
                f"[NeuralMemory] Session end: {result.get('message', 'no memories')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[NeuralMemory] Stop hook error: {exc}", file=sys.stderr)
        sys.exit(0)  # Never block session termination


if __name__ == "__main__":
    main()
