#!/usr/bin/env python3
"""NeuralMemory Documentation Chatbot v2 — Cognitive Gradio UI.

A self-answering chatbot that uses NeuralMemory's full cognitive pipeline:
- Spreading activation retrieval (RRF + graph expansion)
- Conversation memory (remembers within session)
- Cognitive reasoning (hypothesize when uncertain)
- Source citations (provenance tracking)

No LLM needed — the brain IS the answer.

Usage (local):
    python chatbot/app.py                     # Launch locally
    python chatbot/app.py --port 7861         # Custom port
    python chatbot/app.py --share             # Create public URL

HuggingFace Spaces: set app_file=app.py, sdk=gradio in README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from neural_memory import BrainConfig
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage

# ── Config ─────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "brain" / "docs.db"
BRAIN_NAME = "neuralmemory-docs"

MAX_CONVERSATION_TURNS = 20
MAX_CONTEXT_TOKENS = 2000
LOW_CONFIDENCE_THRESHOLD = 0.3
MEDIUM_CONFIDENCE_THRESHOLD = 0.6

EXAMPLE_QUERIES = [
    "How do I install NeuralMemory?",
    "What is spreading activation and how does it work?",
    "How do I configure embeddings with Gemini?",
    "What MCP tools are available?",
    "How does memory consolidation work?",
    "What is the difference between CLI and MCP?",
    "How to set up cloud sync between devices?",
    "What cognitive reasoning tools does NM have?",
]

# ── State ──────────────────────────────────────────────────

_storage: SQLiteStorage | None = None
_config: BrainConfig | None = None
_pipeline: ReflexPipeline | None = None
_encoder: MemoryEncoder | None = None


async def get_pipeline() -> tuple[ReflexPipeline, SQLiteStorage, BrainConfig]:
    """Lazy-init the storage + pipeline on first query."""
    global _storage, _config, _pipeline, _encoder

    if _pipeline is not None and _storage is not None and _config is not None:
        return _pipeline, _storage, _config

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Brain not found at {DB_PATH}. "
            "Run `python chatbot/train_docs_brain.py` first."
        )

    _storage = SQLiteStorage(str(DB_PATH))
    await _storage.initialize()

    brain = await _storage.find_brain_by_name(BRAIN_NAME)
    if brain is None:
        raise ValueError(f"Brain '{BRAIN_NAME}' not found in database.")

    _storage.set_brain(brain.id)
    _config = brain.config or BrainConfig()

    _pipeline = ReflexPipeline(_storage, _config)
    _encoder = MemoryEncoder(_storage, _config)
    return _pipeline, _storage, _config


# ── Session Memory ─────────────────────────────────────────


def _session_id(request: gr.Request | None = None) -> str:
    """Generate a stable session ID from request or fallback."""
    if request and hasattr(request, "session_hash") and request.session_hash:
        return f"chat-{request.session_hash[:12]}"
    return f"chat-{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"


async def _remember_exchange(
    query: str, answer: str, confidence: float, session_id: str
) -> None:
    """Encode a Q&A exchange into the brain for conversation context."""
    if _encoder is None:
        return

    content = f"User asked: {query}\nAnswer ({confidence:.0%} confidence): {answer[:500]}"
    try:
        await _encoder.encode(
            content,
            metadata={
                "type": "conversation",
                "session_id": session_id,
                "confidence": confidence,
            },
            tags={"conversation", "chatbot", session_id},
        )
    except Exception:
        pass  # Non-critical — don't break chat if encoding fails


# ── Retrieval Engine ───────────────────────────────────────


async def _retrieve(
    query: str,
    conversation_context: str,
    depth: DepthLevel,
) -> dict[str, Any]:
    """Run full cognitive retrieval pipeline.

    Returns dict with: context, confidence, neurons, latency, sources, reasoning.
    """
    pipeline, storage, config = await get_pipeline()

    # Build enriched query with conversation context
    enriched_query = query
    if conversation_context:
        enriched_query = f"{conversation_context}\n\nCurrent question: {query}"

    # Primary retrieval — deep depth for better results
    result = await pipeline.query(
        enriched_query,
        depth=depth,
        max_tokens=MAX_CONTEXT_TOKENS,
    )

    primary_confidence = result.confidence or 0.0
    context = result.context or ""
    sources: list[str] = []

    # Extract source fibers for citation
    if result.fibers_matched:
        for fid in result.fibers_matched[:5]:
            try:
                fiber = await storage.get_fiber(fid)
                if fiber and fiber.metadata:
                    src = fiber.metadata.get("source_file") or fiber.metadata.get(
                        "source", ""
                    )
                    if src and src not in sources:
                        sources.append(str(src))
            except Exception:
                continue

    # Cognitive reasoning for low-confidence answers
    reasoning_notes: list[str] = []

    if primary_confidence < LOW_CONFIDENCE_THRESHOLD and context:
        reasoning_notes.append(
            "Low confidence — this topic may not be well-covered in the documentation."
        )
        # Try broader search with DEEP depth
        if depth != DepthLevel.DEEP:
            deep_result = await pipeline.query(
                query, depth=DepthLevel.DEEP, max_tokens=MAX_CONTEXT_TOKENS
            )
            if (deep_result.confidence or 0.0) > primary_confidence:
                context = deep_result.context or context
                primary_confidence = deep_result.confidence or primary_confidence
                reasoning_notes.append(
                    f"Deep search improved confidence: {primary_confidence:.0%}"
                )
                if deep_result.fibers_matched:
                    for fid in deep_result.fibers_matched[:3]:
                        try:
                            fiber = await storage.get_fiber(fid)
                            if fiber and fiber.metadata:
                                src = fiber.metadata.get(
                                    "source_file"
                                ) or fiber.metadata.get("source", "")
                                if src and src not in sources:
                                    sources.append(str(src))
                        except Exception:
                            continue

    elif primary_confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        reasoning_notes.append("High-confidence match from documentation.")

    return {
        "context": context,
        "confidence": primary_confidence,
        "neurons": result.neurons_activated or 0,
        "latency_ms": result.latency_ms or 0,
        "sources": sources,
        "reasoning": reasoning_notes,
        "depth_used": result.depth_used.name if result.depth_used else depth.name,
        "score_breakdown": result.score_breakdown or {},
    }


# ── Chat Handler ───────────────────────────────────────────


def _format_confidence_badge(confidence: float) -> str:
    """Generate HTML confidence badge."""
    if confidence >= 0.7:
        color, label = "#10b981", "High"
    elif confidence >= 0.4:
        color, label = "#f59e0b", "Medium"
    else:
        color, label = "#ef4444", "Low"

    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:12px;font-weight:600;font-size:13px;">'
        f"{label} — {confidence:.0%}</span>"
    )


def _format_sources(sources: list[str]) -> str:
    """Format source list as markdown."""
    if not sources:
        return ""
    items = "\n".join(f"- `{s}`" for s in sources[:5])
    return f"\n\n**Sources:**\n{items}"


def _format_stats(data: dict[str, Any]) -> str:
    """Format retrieval stats."""
    parts = [
        f"Depth: {data.get('depth_used', 'N/A')}",
        f"Neurons: {data.get('neurons', 0)}",
        f"Latency: {data.get('latency_ms', 0):.0f}ms",
    ]
    breakdown = data.get("score_breakdown", {})
    if breakdown:
        for key, val in list(breakdown.items())[:3]:
            parts.append(f"{key}: {val:.2f}")
    return " | ".join(parts)


def _build_conversation_context(history: list[dict[str, str]]) -> str:
    """Extract recent conversation context for enriched queries."""
    if not history:
        return ""

    recent = history[-(MAX_CONVERSATION_TURNS * 2) :]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            # Truncate long assistant messages
            lines.append(f"Assistant: {content[:200]}")

    if not lines:
        return ""
    return "Previous conversation:\n" + "\n".join(lines[-6:])


async def chat_respond(
    message: str,
    history: list[dict[str, str]],
    depth_label: str,
    request: gr.Request | None = None,
) -> tuple[list[dict[str, str]], str, str, str]:
    """Handle a chat message and return updated history + metadata.

    Returns:
        (history, confidence_html, stats_text, sources_md)
    """
    if not message.strip():
        return history, "", "", ""

    depth_map: dict[str, DepthLevel] = {
        "Quick": DepthLevel.INSTANT,
        "Normal": DepthLevel.CONTEXT,
        "Deep": DepthLevel.DEEP,
    }
    depth = depth_map.get(depth_label, DepthLevel.CONTEXT)
    sid = _session_id(request)

    # Add user message to history
    history = [*history, {"role": "user", "content": message}]

    try:
        conversation_context = _build_conversation_context(history[:-1])
        data = await _retrieve(message, conversation_context, depth)

        context = data["context"]
        confidence = data["confidence"]
        sources = data["sources"]
        reasoning = data["reasoning"]

        # Build response
        if not context or context.strip() == "":
            response = (
                "I couldn't find relevant documentation for this question. "
                "Try rephrasing or check the "
                "[docs](https://nhadaututheky.github.io/neural-memory/) directly."
            )
        else:
            response = context
            # Add reasoning notes if any
            if reasoning:
                notes = " | ".join(reasoning)
                response += f"\n\n> *{notes}*"

            response += _format_sources(sources)

        # Add assistant message to history
        history = [*history, {"role": "assistant", "content": response}]

        # Encode exchange into brain for session memory
        await _remember_exchange(message, context[:300], confidence, sid)

        confidence_html = _format_confidence_badge(confidence)
        stats = _format_stats(data)
        sources_md = _format_sources(sources)

    except FileNotFoundError as e:
        history = [*history, {"role": "assistant", "content": str(e)}]
        confidence_html = ""
        stats = ""
        sources_md = ""
    except Exception as e:
        error_msg = f"Error retrieving answer: {e}"
        history = [*history, {"role": "assistant", "content": error_msg}]
        confidence_html = ""
        stats = f"Error: {e}"
        sources_md = ""

    return history, confidence_html, stats, sources_md


# ── Gradio UI ──────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Build the Gradio chat interface."""
    css = """
    .stats-box { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .source-box { font-size: 13px; }
    .badge-box { min-height: 36px; display: flex; align-items: center; }
    """

    with gr.Blocks(
        title="NeuralMemory Docs",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="violet"),
        css=css,
    ) as app:
        gr.Markdown(
            """
# NeuralMemory Documentation Assistant

Ask questions about NeuralMemory — powered by **spreading activation**, not an LLM.

**What's different:** This chatbot remembers your conversation, uses cognitive reasoning
for uncertain answers, and cites its sources. No AI hallucinations — only real docs.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=480,
                    show_copy_button=True,
                    avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/brain_1f9e0.png"),
                    placeholder="Ask me anything about NeuralMemory...",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="How do I install NeuralMemory?",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    depth_select = gr.Radio(
                        choices=["Quick", "Normal", "Deep"],
                        value="Normal",
                        label="Depth",
                        scale=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Retrieval Info")
                confidence_badge = gr.HTML(
                    elem_classes=["badge-box"], label="Confidence"
                )
                stats_text = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["stats-box"],
                    lines=3,
                )
                sources_md = gr.Markdown(
                    label="Sources",
                    elem_classes=["source-box"],
                )
                clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Accordion("Example questions", open=False):
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUERIES],
                inputs=[msg_input],
                label="",
            )

        gr.Markdown(
            """
---
*Powered by [NeuralMemory](https://github.com/nhadaututtheky/neural-memory)
— brain-inspired persistent memory for AI agents.
Spreading activation retrieval • Conversation memory • Cognitive reasoning • Source citations*
"""
        )

        # Event handlers
        submit_args = {
            "fn": chat_respond,
            "inputs": [msg_input, chatbot, depth_select],
            "outputs": [chatbot, confidence_badge, stats_text, sources_md],
        }

        send_btn.click(**submit_args).then(
            fn=lambda: "", outputs=[msg_input]
        )
        msg_input.submit(**submit_args).then(
            fn=lambda: "", outputs=[msg_input]
        )
        clear_btn.click(
            fn=lambda: ([], "", "", ""),
            outputs=[chatbot, confidence_badge, stats_text, sources_md],
        )

    return app


# ── Main ───────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralMemory Docs Chatbot v2")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
