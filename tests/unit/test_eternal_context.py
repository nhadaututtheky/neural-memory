"""Tests for the eternal context system.

Tests cover:
- BrainPersistence: file I/O for all 3 tiers, snapshots, log, cleanup
- TriggerEngine: pattern detection for all trigger types
- EternalContext: 3-tier lifecycle, injection formatting, capacity estimation
"""

from __future__ import annotations

import json
from pathlib import Path

from neural_memory.core.brain_persistence import BrainPersistence
from neural_memory.core.eternal_context import (
    BrainState,
    ContextSnapshot,
    EternalContext,
    SessionState,
)
from neural_memory.core.trigger_engine import (
    TriggerType,
    check_triggers,
    estimate_session_tokens,
)

# ──────────────────── BrainPersistence Tests ────────────────────


class TestBrainPersistence:
    """Tests for file-based persistence of eternal context state."""

    def test_save_load_brain_state(self, tmp_path: Path) -> None:
        """Round-trip save/load of Tier 1 (Critical) state."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        state = BrainState(
            project_name="MyApp",
            tech_stack=("Next.js", "Prisma"),
            key_decisions=(
                {"decision": "Use PostgreSQL", "reason": "Team familiarity", "date": "2024-01-15"},
            ),
            instructions=("Always use type hints",),
        )
        p.save_brain_state(state)
        loaded = p.load_brain_state()

        assert loaded.project_name == "MyApp"
        assert loaded.tech_stack == ("Next.js", "Prisma")
        assert len(loaded.key_decisions) == 1
        assert loaded.key_decisions[0]["decision"] == "Use PostgreSQL"
        assert loaded.instructions == ("Always use type hints",)

    def test_save_load_session_state(self, tmp_path: Path) -> None:
        """Round-trip save/load of Tier 2 (Important) state."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        state = SessionState(
            feature="Authentication",
            task="Login form",
            progress=0.65,
            errors_history=({"error": "CORS", "fixed": True, "date": "2024-01-15"},),
            pending_tasks=("Add tests",),
            branch="feat/auth",
        )
        p.save_session_state(state)
        loaded = p.load_session_state()

        assert loaded.feature == "Authentication"
        assert loaded.task == "Login form"
        assert loaded.progress == 0.65
        assert len(loaded.errors_history) == 1
        assert loaded.errors_history[0]["fixed"] is True
        assert loaded.branch == "feat/auth"

    def test_save_load_context(self, tmp_path: Path) -> None:
        """Round-trip save/load of Tier 3 (Context) state."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        snapshot = ContextSnapshot(
            conversation_summary=("Discussed auth options",),
            recent_files=("src/login.tsx",),
            recent_queries=("how does auth work?",),
            message_count=42,
            token_estimate=15000,
        )
        p.save_context(snapshot)
        loaded = p.load_context()

        assert loaded.conversation_summary == ("Discussed auth options",)
        assert loaded.recent_files == ("src/login.tsx",)
        assert loaded.message_count == 42
        assert loaded.token_estimate == 15000

    def test_load_missing_files_returns_defaults(self, tmp_path: Path) -> None:
        """Loading from non-existent files returns default states."""
        p = BrainPersistence("nonexistent", base_dir=tmp_path)

        brain = p.load_brain_state()
        assert brain.project_name == ""
        assert brain.tech_stack == ()

        session = p.load_session_state()
        assert session.feature == ""

        context = p.load_context()
        assert context.message_count == 0

    def test_corrupted_json_returns_defaults(self, tmp_path: Path) -> None:
        """Corrupted JSON files return defaults instead of raising."""
        p = BrainPersistence("corrupt", base_dir=tmp_path)
        p.ensure_dirs()

        # Write invalid JSON
        (p.directory / "brain.json").write_text("{invalid json", encoding="utf-8")
        brain = p.load_brain_state()
        assert brain.project_name == ""

    def test_snapshot_create_and_load(self, tmp_path: Path) -> None:
        """Create a snapshot and load it back."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        brain = BrainState(project_name="TestProject")
        session = SessionState(feature="Testing", task="Write tests")

        path = p.create_snapshot(brain, session)
        assert path.exists()

        loaded_brain, loaded_session = p.load_snapshot(path)
        assert loaded_brain.project_name == "TestProject"
        assert loaded_session.feature == "Testing"

    def test_snapshot_list(self, tmp_path: Path) -> None:
        """List snapshots returns sorted paths."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        brain = BrainState()
        session = SessionState()

        p.create_snapshot(brain, session)

        snapshots = p.list_snapshots()
        assert len(snapshots) >= 1
        assert all(s.suffix == ".json" for s in snapshots)

    def test_snapshot_cleanup(self, tmp_path: Path) -> None:
        """Cleanup removes snapshots older than retention period."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        p.ensure_dirs()

        # Create an old snapshot manually
        old_path = p._snapshots_dir / "2020-01-01_120000.json"
        old_data = {
            "created_at": "2020-01-01T12:00:00",
            "brain_id": "test-brain",
            "brain": {"project_name": ""},
            "session": {"feature": ""},
        }
        with open(old_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        # Create a recent snapshot
        p.create_snapshot(BrainState(), SessionState())

        deleted = p.cleanup_snapshots(retention_days=7)
        assert deleted == 1
        assert not old_path.exists()

    def test_session_log_append_and_read(self, tmp_path: Path) -> None:
        """Append to session log and read back."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        p.append_log("Started session")
        p.append_log("Made a decision")

        lines = p.read_log(tail=10)
        assert len(lines) == 2
        assert "Started session" in lines[0]
        assert "Made a decision" in lines[1]

    def test_read_log_empty(self, tmp_path: Path) -> None:
        """Reading log from non-existent file returns empty list."""
        p = BrainPersistence("test-brain", base_dir=tmp_path)
        assert p.read_log() == []


# ──────────────────── TriggerEngine Tests ────────────────────


class TestTriggerEngine:
    """Tests for auto-save trigger detection."""

    def test_no_trigger_on_normal_text(self) -> None:
        """Normal conversation text should not trigger."""
        result = check_triggers("How do I implement a login form?", message_count=3)
        assert result.triggered is False

    def test_no_trigger_on_short_text(self) -> None:
        """Very short text should not trigger patterns."""
        result = check_triggers("ok", message_count=3)
        assert result.triggered is False

    def test_user_leaving_english(self) -> None:
        """Detect English user-leaving patterns."""
        result = check_triggers("Alright, bye! I'll continue tomorrow.", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.USER_LEAVING
        assert 1 in result.save_tiers and 2 in result.save_tiers and 3 in result.save_tiers

    def test_user_leaving_vietnamese(self) -> None:
        """Detect Vietnamese user-leaving patterns."""
        result = check_triggers("ok tạm nghỉ nhé, mai làm tiếp", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.USER_LEAVING

    def test_milestone_english(self) -> None:
        """Detect English milestone/workflow completion."""
        result = check_triggers("All tests pass, feature complete!", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.WORKFLOW_END
        assert 1 in result.save_tiers and 2 in result.save_tiers

    def test_milestone_vietnamese(self) -> None:
        """Detect Vietnamese milestone patterns."""
        result = check_triggers("hoàn thành rồi, deploy xong", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.WORKFLOW_END

    def test_error_fixed(self) -> None:
        """Detect error-fixed patterns."""
        result = check_triggers("Fixed the CORS issue by adding the right headers", message_count=5)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.ERROR_FIXED

    def test_decision_made(self) -> None:
        """Detect decision patterns."""
        result = check_triggers(
            "We decided to use PostgreSQL instead of MongoDB for this project",
            message_count=5,
        )
        assert result.triggered is True
        assert result.trigger_type == TriggerType.DECISION_MADE

    def test_checkpoint_every_n_messages(self) -> None:
        """Checkpoint triggers at multiples of interval."""
        result = check_triggers("just a normal message", message_count=15)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.CHECKPOINT
        assert 2 in result.save_tiers

    def test_no_checkpoint_at_non_multiple(self) -> None:
        """No checkpoint at non-multiples."""
        result = check_triggers("just a normal message", message_count=14)
        assert result.triggered is False

    def test_context_warning(self) -> None:
        """Context capacity warning when tokens > threshold."""
        result = check_triggers(
            "some text here",
            message_count=5,
            token_estimate=110_000,
            max_tokens=128_000,
        )
        assert result.triggered is True
        assert result.trigger_type == TriggerType.CONTEXT_WARNING
        assert 1 in result.save_tiers and 2 in result.save_tiers and 3 in result.save_tiers

    def test_context_warning_priority_over_patterns(self) -> None:
        """Context warning has highest priority even if text matches other patterns."""
        result = check_triggers(
            "bye, I'm done for today",
            message_count=15,
            token_estimate=120_000,
            max_tokens=128_000,
        )
        assert result.trigger_type == TriggerType.CONTEXT_WARNING

    def test_estimate_session_tokens(self) -> None:
        """Token estimation formula: messages*150 + code*5 + errors*300."""
        tokens = estimate_session_tokens(message_count=100, code_lines=200, error_count=2)
        assert tokens == 100 * 150 + 200 * 5 + 2 * 300
        assert tokens == 16600

    def test_estimate_session_tokens_messages_only(self) -> None:
        """Token estimation with messages only."""
        tokens = estimate_session_tokens(message_count=50)
        assert tokens == 7500


# ──────────────────── EternalContext Tests ────────────────────


class TestEternalContext:
    """Tests for the 3-tier eternal context manager."""

    def test_load_save_roundtrip(self, tmp_path: Path) -> None:
        """Load and save complete state round-trip."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.update_brain(project_name="TestApp", tech_stack=["Python", "FastAPI"])
        ctx.update_session(feature="Auth", task="Login")
        ctx.add_summary("Discussed auth options")

        ctx.save()

        # Create new context and load
        ctx2 = EternalContext("test", persistence)
        ctx2.load()

        assert ctx2.brain.project_name == "TestApp"
        assert ctx2.brain.tech_stack == ("Python", "FastAPI")
        assert ctx2.session.feature == "Auth"
        assert ctx2.session.task == "Login"
        assert "Discussed auth options" in ctx2.context.conversation_summary

    def test_add_decision(self, tmp_path: Path) -> None:
        """Add a key decision to Tier 1."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_decision("Use Redis", "Low latency")
        assert len(ctx.brain.key_decisions) == 1
        assert ctx.brain.key_decisions[0]["decision"] == "Use Redis"
        assert ctx.brain.key_decisions[0]["reason"] == "Low latency"

    def test_add_decision_dedup(self, tmp_path: Path) -> None:
        """Duplicate decisions are not added twice."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_decision("Use Redis", "Low latency")
        ctx.add_decision("Use Redis", "Different reason")
        assert len(ctx.brain.key_decisions) == 1

    def test_add_error(self, tmp_path: Path) -> None:
        """Add error to Tier 2 history."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_error("CORS blocked", fixed=False)
        assert len(ctx.session.errors_history) == 1
        assert ctx.session.errors_history[0]["error"] == "CORS blocked"
        assert ctx.session.errors_history[0]["fixed"] is False

    def test_mark_error_fixed(self, tmp_path: Path) -> None:
        """Mark an existing error as fixed."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_error("CORS blocked", fixed=False)
        ctx.add_error("404 not found", fixed=False)

        found = ctx.mark_error_fixed("CORS")
        assert found is True
        assert ctx.session.errors_history[0]["fixed"] is True
        assert ctx.session.errors_history[1]["fixed"] is False

    def test_mark_error_fixed_not_found(self, tmp_path: Path) -> None:
        """Marking non-existent error returns False."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        found = ctx.mark_error_fixed("nonexistent")
        assert found is False

    def test_update_session_sets_timestamps(self, tmp_path: Path) -> None:
        """Session updates set started_at and updated_at."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.update_session(feature="Auth")
        assert ctx.session.started_at != ""
        assert ctx.session.updated_at != ""

    def test_get_injection_level1(self, tmp_path: Path) -> None:
        """Level 1 injection includes project + current task."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.update_brain(project_name="MyApp", tech_stack=["Next.js"])
        ctx.update_session(feature="Auth", task="Login", progress=0.65)

        injection = ctx.get_injection(level=1)
        assert "MyApp" in injection
        assert "Next.js" in injection
        assert "Auth" in injection
        assert "Login" in injection
        assert "65%" in injection

    def test_get_injection_level2(self, tmp_path: Path) -> None:
        """Level 2 injection includes decisions and pending tasks."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.update_brain(project_name="MyApp")
        ctx.add_decision("Use NextAuth", "Simple setup")
        ctx.update_session(pending_tasks=["Add tests", "Deploy"])

        injection = ctx.get_injection(level=2)
        assert "NextAuth" in injection
        assert "Add tests" in injection
        assert "Deploy" in injection

    def test_get_injection_level3(self, tmp_path: Path) -> None:
        """Level 3 injection includes conversation summary and files."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_summary("Discussed auth options")
        ctx.add_recent_file("src/login.tsx")
        ctx.add_query("how does auth work?")

        injection = ctx.get_injection(level=3)
        assert "Discussed auth options" in injection
        assert "src/login.tsx" in injection
        assert "how does auth work?" in injection

    def test_estimate_context_usage(self, tmp_path: Path) -> None:
        """Context usage estimation returns reasonable values."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        # With no content, usage should be very low
        usage = ctx.estimate_context_usage(max_tokens=128_000)
        assert 0.0 <= usage <= 0.01

        # With significant token estimate
        ctx.update_context(token_estimate=100_000)
        usage = ctx.estimate_context_usage(max_tokens=128_000)
        assert usage > 0.7

    def test_compact(self, tmp_path: Path) -> None:
        """Compact moves Tier 3 summaries to Tier 2 and clears Tier 3."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_summary("Discussed auth options")
        ctx.add_summary("Implemented login form")
        ctx.add_query("how does auth work?")

        summary = ctx.compact()
        assert "auth options" in summary.lower() or "login form" in summary.lower()
        # Tier 3 should be cleared
        assert ctx.context.conversation_summary == ()
        assert ctx.context.message_count == 0
        # Tier 2 should have gained a pending task note
        assert len(ctx.session.pending_tasks) > 0

    def test_add_recent_file_dedup(self, tmp_path: Path) -> None:
        """Duplicate files are deduplicated, keeping most recent."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_recent_file("src/a.ts")
        ctx.add_recent_file("src/b.ts")
        ctx.add_recent_file("src/a.ts")

        assert ctx.context.recent_files == ("src/b.ts", "src/a.ts")

    def test_add_recent_file_max_10(self, tmp_path: Path) -> None:
        """Recent files capped at 10."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        for i in range(15):
            ctx.add_recent_file(f"src/file{i}.ts")

        assert len(ctx.context.recent_files) == 10

    def test_add_summary_max_20(self, tmp_path: Path) -> None:
        """Conversation summaries capped at 20."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        for i in range(25):
            ctx.add_summary(f"Summary {i}")

        assert len(ctx.context.conversation_summary) == 20

    def test_increment_message_count(self, tmp_path: Path) -> None:
        """Message counter increments correctly."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        assert ctx.increment_message_count() == 1
        assert ctx.increment_message_count() == 2
        assert ctx.context.message_count == 2

    def test_save_with_snapshot(self, tmp_path: Path) -> None:
        """save_with_snapshot creates both files and snapshot."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.update_brain(project_name="TestApp")
        ctx.save_with_snapshot()

        # Verify files exist
        assert (persistence.directory / "brain.json").exists()
        assert (persistence.directory / "session.json").exists()
        assert (persistence.directory / "context.json").exists()
        assert len(persistence.list_snapshots()) == 1

    def test_log(self, tmp_path: Path) -> None:
        """Log entries are written to session_log.txt."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.log("Test entry")
        lines = persistence.read_log()
        assert len(lines) == 1
        assert "Test entry" in lines[0]

    def test_injection_with_active_errors(self, tmp_path: Path) -> None:
        """Level 1 injection shows active (unfixed) errors."""
        persistence = BrainPersistence("test", base_dir=tmp_path)
        ctx = EternalContext("test", persistence)

        ctx.add_error("CORS blocked", fixed=False)
        ctx.add_error("404 not found", fixed=True)

        injection = ctx.get_injection(level=1)
        assert "CORS" in injection
        assert "Active errors: 1" in injection
