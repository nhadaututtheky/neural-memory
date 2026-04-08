"""Integration tests for causal auto-inclusion + anti-redundancy in retrieval pipeline."""

from __future__ import annotations

from neural_memory.core.brain import BrainConfig
from neural_memory.engine.session_state import SessionManager


class TestBrainConfigCausalFields:
    """Verify new BrainConfig fields for causal integration."""

    def test_defaults(self) -> None:
        config = BrainConfig()
        assert config.causal_auto_include is True
        assert config.causal_auto_include_max_hops == 2
        assert config.anti_redundancy_penalty == 0.3

    def test_override(self) -> None:
        config = BrainConfig(
            causal_auto_include=False,
            causal_auto_include_max_hops=4,
            anti_redundancy_penalty=0.5,
        )
        assert config.causal_auto_include is False
        assert config.causal_auto_include_max_hops == 4
        assert config.anti_redundancy_penalty == 0.5

    def test_with_updates(self) -> None:
        config = BrainConfig()
        updated = config.with_updates(anti_redundancy_penalty=0.1)
        assert updated.anti_redundancy_penalty == 0.1
        # Original unchanged (immutable)
        assert config.anti_redundancy_penalty == 0.3


class TestSessionStateAttentionIntegration:
    """Integration: attention set works across SessionManager lifecycle."""

    def setup_method(self) -> None:
        SessionManager.reset()

    def teardown_method(self) -> None:
        SessionManager.reset()

    def test_surfaced_fibers_persist_across_queries(self) -> None:
        mgr = SessionManager.get_instance()
        session = mgr.get_or_create("test-session")

        # First query: surface fibers f1, f2
        session.record_surfaced(["f1", "f2"])

        # Second access: same session retains state
        session2 = mgr.get_or_create("test-session")
        assert session2.is_surfaced("f1")
        assert session2.is_surfaced("f2")

    def test_different_sessions_independent(self) -> None:
        mgr = SessionManager.get_instance()
        s1 = mgr.get_or_create("session-a")
        s2 = mgr.get_or_create("session-b")

        s1.record_surfaced(["f1"])
        assert s1.is_surfaced("f1")
        assert not s2.is_surfaced("f1")

    def test_session_expiry_clears_attention_set(self) -> None:
        mgr = SessionManager.get_instance()
        session = mgr.get_or_create("test-session")
        session.record_surfaced(["f1", "f2"])

        # Force expiry
        session.last_active = 0.0  # long ago
        mgr._expire_stale(now=999999999.0)

        # Session should be gone
        assert mgr.get("test-session") is None

    def test_anti_redundancy_penalty_value(self) -> None:
        """Anti-redundancy penalty should be multiplicative, not exclusion."""
        config = BrainConfig()
        # 0.3 means surfaced fibers get 30% of normal score
        assert 0 < config.anti_redundancy_penalty < 1.0


class TestCausalSupplementFormat:
    """Integration: causal supplement text format is LLM-friendly."""

    def test_supplement_lines_are_labeled(self) -> None:
        from neural_memory.core.synapse import SynapseType
        from neural_memory.engine.causal_inclusion import format_causal_supplement
        from neural_memory.engine.causal_traversal import CausalChain, CausalStep

        chains = [
            CausalChain(
                seed_neuron_id="n1",
                direction="causes",
                steps=(CausalStep("n2", "Config was wrong", SynapseType.CAUSED_BY, 0.9, 0),),
                total_weight=0.9,
            ),
            CausalChain(
                seed_neuron_id="n1",
                direction="effects",
                steps=(CausalStep("n3", "Deploy failed", SynapseType.LEADS_TO, 0.85, 0),),
                total_weight=0.85,
            ),
        ]
        text = format_causal_supplement(chains, max_chars=5000)
        lines = text.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("[Caused by]")
        assert lines[1].startswith("[Led to]")
