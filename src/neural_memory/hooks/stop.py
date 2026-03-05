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


def read_transcript_tail(transcript_path: str, max_lines: int = MAX_TRANSCRIPT_LINES) -> str:
    """Read the last N entries from a JSONL transcript and extract text content."""
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
                text = _extract_text(entry)
                if text and len(text) > 20:
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

        brain = await storage.get_brain(config.current_brain)
        if not brain:
            return {"error": "No brain configured", "saved": 0}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        auto_redact_severity = config.safety.auto_redact_min_severity
        saved: list[str] = []

        for item in eligible:
            try:
                content = item["content"]
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
        print("No substantial content to capture", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    try:
        result = asyncio.run(capture_text(text))
        saved = result.get("saved", 0)
        if saved > 0:
            print(  # noqa: T201
                f"[NeuralMemory] Session end: captured {saved} memories",
                file=sys.stderr,
            )
        else:
            print(  # noqa: T201
                f"[NeuralMemory] Session end: {result.get('message', 'no memories')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[NeuralMemory] Stop hook error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(0)  # Never block session termination


if __name__ == "__main__":
    main()
