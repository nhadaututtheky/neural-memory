"""PreCompact hook: auto-flush memories before context compaction.

Called by Claude Code before context window is compressed.
Reads the conversation transcript, detects memorable content,
and saves it to the brain — preventing memory loss from compaction.

Usage as Claude Code hook:
    Receives JSON on stdin with `transcript_path` field.
    Outputs status to stderr (stdout reserved for hook response).

Usage standalone:
    echo '{"transcript_path": "/path/to/transcript.jsonl"}' | python -m neural_memory.hooks.pre_compact
    python -m neural_memory.hooks.pre_compact --transcript /path/to/transcript.jsonl
    python -m neural_memory.hooks.pre_compact --text "Some text to flush"
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max lines to read from transcript tail
MAX_TRANSCRIPT_LINES = 80
# Max characters to flush (matches MCP constant)
MAX_FLUSH_CHARS = 100_000
# Emergency confidence threshold
EMERGENCY_THRESHOLD = 0.5
# Priority boost for emergency-captured memories
PRIORITY_BOOST = 2


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
    """Read the last N entries from a JSONL transcript and extract text content.

    Handles multiple transcript formats:
    - {"role": "...", "content": "text"}
    - {"role": "...", "content": [{"type": "text", "text": "..."}]}
    - {"type": "human|assistant", "message": {"content": [...]}}
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
                text = _extract_text(entry)
                if text and len(text) > 20:  # Skip trivial entries
                    lines.append(text)
            except json.JSONDecodeError:
                continue
    except OSError:
        return ""

    joined = "\n\n".join(lines)
    # Truncate to max flush size
    if len(joined) > MAX_FLUSH_CHARS:
        joined = joined[-MAX_FLUSH_CHARS:]
    return joined


def _extract_text(entry: dict[str, Any]) -> str:
    """Extract text content from a transcript entry."""
    # Format: {"role": "...", "content": "text"}  # noqa: ERA001
    content = entry.get("content")
    if isinstance(content, str):
        return content

    # Format: {"role": "...", "content": [{"type": "text", "text": "..."}]}  # noqa: ERA001
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    # Format: {"type": "...", "message": {"content": [...]}}  # noqa: ERA001
    message = entry.get("message")
    if isinstance(message, dict):
        return _extract_text(message)

    # Fallback: direct text field
    text = entry.get("text", "")
    return text if isinstance(text, str) else ""


async def flush_text(text: str) -> dict[str, Any]:
    """Detect and save memorable content from text.

    Uses the same auto-capture pipeline as the MCP server's flush action,
    but runs standalone without the MCP server.
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
        # Detect ALL memory types with emergency settings
        detected = analyze_text_for_memories(
            text,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            capture_preferences=True,
        )

        if not detected:
            return {"saved": 0, "message": "No memorable content detected"}

        # Emergency threshold: more aggressive than normal
        eligible = [item for item in detected if item["confidence"] >= EMERGENCY_THRESHOLD]
        if not eligible:
            return {"saved": 0, "message": "No memories met emergency threshold"}

        # Boost priority for emergency-captured memories
        boosted = [
            {**item, "priority": min(item.get("priority", 5) + PRIORITY_BOOST, 10)}
            for item in eligible
        ]

        # Get brain for encoder
        brain = await storage.get_brain(config.current_brain)
        if not brain:
            return {"error": "No brain configured", "saved": 0}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        auto_redact_severity = config.safety.auto_redact_min_severity
        saved: list[str] = []

        for item in boosted:
            try:
                # Auto-redact sensitive content
                content = item["content"]
                redacted_content, matches, _ = auto_redact_content(
                    content, min_severity=auto_redact_severity
                )
                if matches:
                    logger.debug("Auto-redacted %d matches in flush memory", len(matches))

                # Encode into neural graph
                result = await encoder.encode(
                    content=redacted_content,
                    timestamp=utcnow(),
                    tags={"emergency_flush", "pre_compact"},
                )

                # Create typed memory metadata
                mem_type_str = item.get("type", "fact")
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    mem_type = MemoryType.FACT

                typed_mem = TypedMemory.create(
                    fiber_id=result.fiber.id,
                    memory_type=mem_type,
                    priority=Priority.from_int(item.get("priority", 5)),
                    source="pre_compact_hook",
                    tags={"emergency_flush", "pre_compact"},
                )
                await storage.add_typed_memory(typed_mem)
                saved.append(redacted_content[:60])
            except Exception:
                logger.debug("Failed to save flush memory", exc_info=True)
                continue

        await storage.batch_save()

        return {
            "saved": len(saved),
            "memories": saved,
            "mode": "emergency_flush",
            "threshold": EMERGENCY_THRESHOLD,
            "message": f"Emergency flush: captured {len(saved)} memories"
            if saved
            else "No memories saved",
        }
    finally:
        await storage.close()


def main() -> None:
    """Entry point for PreCompact hook or standalone CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuralMemory PreCompact hook — flush memories before compaction"
    )
    parser.add_argument(
        "--transcript",
        "-t",
        help="Path to JSONL transcript file",
    )
    parser.add_argument(
        "--text",
        help="Direct text to flush (alternative to transcript)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    text = ""

    if args.text:
        # Direct text input
        text = args.text
    elif args.transcript:
        # Read from transcript file
        text = read_transcript_tail(args.transcript)
    else:
        # Read Claude Code hook input from stdin
        hook_input = read_hook_input()
        transcript_path = hook_input.get("transcript_path", "")
        if transcript_path:
            text = read_transcript_tail(transcript_path)
        else:
            # No transcript path — nothing to flush
            sys.exit(0)

    if not text or len(text.strip()) < 50:
        # Too little content to analyze
        print("No substantial content to flush", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    try:
        result = asyncio.run(flush_text(text))
        saved = result.get("saved", 0)
        if saved > 0:
            print(  # noqa: T201
                f"[NeuralMemory] Pre-compact flush: captured {saved} memories",
                file=sys.stderr,
            )
        else:
            print(  # noqa: T201
                f"[NeuralMemory] Pre-compact flush: {result.get('message', 'no memories')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[NeuralMemory] Flush error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(0)  # Don't block compaction on errors


if __name__ == "__main__":
    main()
