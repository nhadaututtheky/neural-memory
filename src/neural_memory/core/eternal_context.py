"""Three-tier eternal context system for NeuralMemory.

Manages persistent context across AI sessions using a 3-tier model:

- **Tier 1 (Critical)**: Project identity, tech stack, key decisions.
  Never deleted. ~200-500 tokens.

- **Tier 2 (Important)**: Current feature, task, progress, errors.
  Kept within session. ~300-800 tokens.

- **Tier 3 (Context)**: Conversation summaries, recent files, queries.
  Temporary. ~500-2000 tokens.

All state is stored as lightweight JSON files, not in SQLite.
Actual memory data stays in the neural graph via SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from neural_memory.core.brain_persistence import BrainPersistence

logger = logging.getLogger(__name__)

# Token estimation ratio (words * ratio ≈ tokens)
_TOKEN_RATIO = 1.3


@dataclass(frozen=True)
class BrainState:
    """Tier 1 — CRITICAL. Never deleted. ~200-500 tokens.

    Stores project identity and key decisions that must survive
    across all sessions indefinitely.
    """

    project_name: str = ""
    tech_stack: tuple[str, ...] = ()
    key_decisions: tuple[dict[str, str], ...] = ()
    instructions: tuple[str, ...] = ()
    version: str = "1.0"


@dataclass(frozen=True)
class SessionState:
    """Tier 2 — IMPORTANT. Kept within session. ~300-800 tokens.

    Stores current working state: feature, task, progress, errors.
    Saved on workflow events and session boundaries.
    """

    feature: str = ""
    task: str = ""
    progress: float = 0.0
    errors_history: tuple[dict[str, Any], ...] = ()
    pending_tasks: tuple[str, ...] = ()
    branch: str = ""
    commit: str = ""
    started_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ContextSnapshot:
    """Tier 3 — CONTEXT. Temporary. ~500-2000 tokens.

    Stores ephemeral conversation state: summaries, recent files,
    recent queries, counters.
    """

    conversation_summary: tuple[str, ...] = ()
    recent_files: tuple[str, ...] = ()
    recent_queries: tuple[str, ...] = ()
    message_count: int = 0
    token_estimate: int = 0


class EternalContext:
    """Orchestrates the 3-tier eternal context lifecycle.

    Manages load/save, tier updates, injection formatting,
    and context capacity estimation.
    """

    def __init__(self, brain_id: str, persistence: BrainPersistence | None = None) -> None:
        self._brain_id = brain_id
        self._persistence = persistence or BrainPersistence(brain_id)
        self._brain: BrainState = BrainState()
        self._session: SessionState = SessionState()
        self._context: ContextSnapshot = ContextSnapshot()
        self._loaded = False

    @property
    def brain(self) -> BrainState:
        """Current Tier 1 (Critical) state."""
        return self._brain

    @property
    def session(self) -> SessionState:
        """Current Tier 2 (Important) state."""
        return self._session

    @property
    def context(self) -> ContextSnapshot:
        """Current Tier 3 (Context) state."""
        return self._context

    @property
    def is_loaded(self) -> bool:
        """Whether state has been loaded from persistence."""
        return self._loaded

    # ──────────────────── Load / Save ────────────────────

    def load(self) -> None:
        """Load all 3 tiers from file persistence."""
        try:
            self._brain = self._persistence.load_brain_state()
            self._session = self._persistence.load_session_state()
            self._context = self._persistence.load_context()
            self._loaded = True
        except Exception as e:
            logger.warning("Failed to load eternal context: %s", e)
            self._loaded = True  # Mark loaded even on failure (use defaults)

    def save(self, tiers: tuple[int, ...] = (1, 2, 3)) -> None:
        """Persist specified tiers to files.

        Args:
            tiers: Which tiers to save. (1, 2, 3) = all.
        """
        try:
            if 1 in tiers:
                self._persistence.save_brain_state(self._brain)
            if 2 in tiers:
                self._persistence.save_session_state(self._session)
            if 3 in tiers:
                self._persistence.save_context(self._context)
        except Exception as e:
            logger.error("Failed to save eternal context: %s", e)

    def save_with_snapshot(self) -> None:
        """Save all tiers and create a timestamped snapshot."""
        self.save(tiers=(1, 2, 3))
        try:
            self._persistence.create_snapshot(self._brain, self._session)
        except Exception as e:
            logger.warning("Failed to create snapshot: %s", e)

    # ──────────────────── Tier 1 updates ────────────────────

    def update_brain(self, **kwargs: Any) -> None:
        """Update Tier 1 (Critical) state immutably."""
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "tech_stack" and isinstance(value, (list, tuple)):
                updates[key] = tuple(value)
            elif key == "key_decisions" and isinstance(value, (list, tuple)):
                updates[key] = tuple(dict(d) for d in value)
            elif key == "instructions" and isinstance(value, (list, tuple)):
                updates[key] = tuple(value)
            elif hasattr(self._brain, key):
                updates[key] = value
        if updates:
            self._brain = replace(self._brain, **updates)

    def add_decision(self, decision: str, reason: str = "") -> None:
        """Append a key decision to Tier 1."""
        entry = {
            "decision": decision,
            "reason": reason,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        # Deduplicate: skip if same decision text already exists
        existing = {d.get("decision", "").lower() for d in self._brain.key_decisions}
        if decision.lower() in existing:
            return
        self._brain = replace(
            self._brain,
            key_decisions=(*self._brain.key_decisions, entry),
        )

    # ──────────────────── Tier 2 updates ────────────────────

    def update_session(self, **kwargs: Any) -> None:
        """Update Tier 2 (Important) state immutably."""
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("errors_history", "pending_tasks") and isinstance(value, (list, tuple)):
                updates[key] = tuple(value)
            elif hasattr(self._session, key):
                updates[key] = value
        updates["updated_at"] = datetime.now().isoformat()
        if not self._session.started_at:
            updates["started_at"] = datetime.now().isoformat()
        self._session = replace(self._session, **updates)

    def add_error(self, error: str, fixed: bool = False) -> None:
        """Append an error to Tier 2 history."""
        entry: dict[str, Any] = {
            "error": error,
            "fixed": fixed,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._session = replace(
            self._session,
            errors_history=(*self._session.errors_history, entry),
            updated_at=datetime.now().isoformat(),
        )

    def mark_error_fixed(self, error_substring: str) -> bool:
        """Mark an existing error as fixed (by substring match).

        Returns True if an error was found and marked.
        """
        error_lower = error_substring.lower()
        updated = []
        found = False
        for entry in self._session.errors_history:
            if (
                not found
                and error_lower in entry.get("error", "").lower()
                and not entry.get("fixed", False)
            ):
                updated.append({**entry, "fixed": True})
                found = True
            else:
                updated.append(dict(entry))
        if found:
            self._session = replace(
                self._session,
                errors_history=tuple(updated),
                updated_at=datetime.now().isoformat(),
            )
        return found

    # ──────────────────── Tier 3 updates ────────────────────

    def update_context(self, **kwargs: Any) -> None:
        """Update Tier 3 (Context) state immutably."""
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in ("conversation_summary", "recent_files", "recent_queries") and isinstance(
                value, (list, tuple)
            ):
                updates[key] = tuple(value)
            elif hasattr(self._context, key):
                updates[key] = value
        if updates:
            self._context = replace(self._context, **updates)

    def add_summary(self, summary: str) -> None:
        """Append a conversation summary to Tier 3 (max 20 entries)."""
        entries = (*self._context.conversation_summary, summary)
        if len(entries) > 20:
            entries = entries[-20:]
        self._context = replace(self._context, conversation_summary=entries)

    def add_recent_file(self, filepath: str) -> None:
        """Track a recently accessed file in Tier 3 (max 10)."""
        # Deduplicate and keep recent
        files = [f for f in self._context.recent_files if f != filepath]
        files.append(filepath)
        if len(files) > 10:
            files = files[-10:]
        self._context = replace(self._context, recent_files=tuple(files))

    def add_query(self, query: str) -> None:
        """Track a recent query in Tier 3 (max 10)."""
        queries = (*self._context.recent_queries, query)
        if len(queries) > 10:
            queries = queries[-10:]
        self._context = replace(
            self._context,
            recent_queries=queries,
            message_count=self._context.message_count + 1,
        )

    def increment_message_count(self) -> int:
        """Increment message counter and return new count."""
        new_count = self._context.message_count + 1
        self._context = replace(self._context, message_count=new_count)
        return new_count

    # ──────────────────── Injection formatting ────────────────────

    def get_injection(self, level: int = 1) -> str:
        """Format context for system prompt injection.

        Args:
            level: Loading level (1=instant, 2=on-demand, 3=deep).

        Returns:
            Formatted context string for injection.
        """
        parts: list[str] = []

        # Level 1: Always loaded (~500 tokens)
        parts.append("## Project Context")
        if self._brain.project_name:
            parts.append(f"- Project: {self._brain.project_name}")
        if self._brain.tech_stack:
            parts.append(f"- Stack: {', '.join(self._brain.tech_stack)}")
        if self._session.feature:
            parts.append(f"- Current: {self._session.feature}")
        if self._session.task:
            pct = int(self._session.progress * 100) if self._session.progress else 0
            task_line = f"- Task: {self._session.task}"
            if pct > 0:
                task_line += f" ({pct}%)"
            parts.append(task_line)
        # Active errors
        active_errors = [e for e in self._session.errors_history if not e.get("fixed", False)]
        if active_errors:
            parts.append(f"- Active errors: {len(active_errors)}")
            for err in active_errors[-3:]:
                parts.append(f"  - {err.get('error', 'unknown')}")

        if level < 2:
            return "\n".join(parts)

        # Level 2: On-demand (~1000 additional tokens)
        if self._brain.key_decisions:
            parts.append("\n## Key Decisions")
            for d in self._brain.key_decisions[-5:]:
                line = f"- {d.get('decision', '')}"
                if d.get("reason"):
                    line += f" — {d['reason']}"
                parts.append(line)

        if self._session.pending_tasks:
            parts.append("\n## Pending Tasks")
            for task in self._session.pending_tasks[-5:]:
                parts.append(f"- {task}")

        if self._brain.instructions:
            parts.append("\n## Instructions")
            for inst in self._brain.instructions[-5:]:
                parts.append(f"- {inst}")

        if self._session.branch:
            parts.append(f"\n## Git: {self._session.branch}")
            if self._session.commit:
                parts.append(f"- Commit: {self._session.commit}")

        if level < 3:
            return "\n".join(parts)

        # Level 3: Deep dive (~2000+ additional tokens)
        if self._context.conversation_summary:
            parts.append("\n## Conversation Summary")
            for s in self._context.conversation_summary[-10:]:
                parts.append(f"- {s}")

        if self._context.recent_files:
            parts.append("\n## Recent Files")
            for f in self._context.recent_files:
                parts.append(f"- {f}")

        if self._context.recent_queries:
            parts.append("\n## Recent Queries")
            for q in self._context.recent_queries[-5:]:
                parts.append(f"- {q}")

        all_errors = self._session.errors_history
        if all_errors:
            parts.append("\n## Error History")
            for err in all_errors[-10:]:
                status = "fixed" if err.get("fixed") else "open"
                parts.append(f"- [{status}] {err.get('error', '')} ({err.get('date', '')})")

        return "\n".join(parts)

    # ──────────────────── Capacity estimation ────────────────────

    def estimate_context_usage(self, max_tokens: int = 128_000) -> float:
        """Estimate fraction of context window used (0.0 to 1.0).

        Uses the stored token_estimate from Tier 3, plus estimates
        of the eternal context overhead itself.
        """
        if max_tokens <= 0:
            return 0.0

        # Overhead from eternal context injection
        injection = self.get_injection(level=3)
        injection_tokens = int(len(injection.split()) * _TOKEN_RATIO)

        # Session token estimate (from external tracking)
        session_tokens = self._context.token_estimate

        total = injection_tokens + session_tokens
        return min(1.0, total / max_tokens)

    # ──────────────────── Compact ────────────────────

    def compact(self) -> str:
        """Summarize Tier 3 into Tier 2 and clear Tier 3.

        Returns a compact summary of what was in Tier 3.
        """
        summaries = list(self._context.conversation_summary)
        queries = list(self._context.recent_queries)

        # Build compact summary
        parts: list[str] = []
        if summaries:
            parts.append(f"Previous session ({len(summaries)} items): " + "; ".join(summaries[-5:]))
        if queries:
            parts.append("Recent queries: " + ", ".join(queries[-5:]))

        compact_text = " | ".join(parts) if parts else "No context to compact."

        # Add to pending tasks as a reference
        if summaries:
            note = f"Session summary: {'; '.join(summaries[-3:])}"
            self._session = replace(
                self._session,
                pending_tasks=(*self._session.pending_tasks, note)[-10:],
                updated_at=datetime.now().isoformat(),
            )

        # Clear Tier 3
        self._context = ContextSnapshot()

        return compact_text

    # ──────────────────── Log ────────────────────

    def log(self, entry: str) -> None:
        """Append an entry to the session log."""
        try:
            self._persistence.append_log(entry)
        except Exception as e:
            logger.warning("Failed to write log: %s", e)
