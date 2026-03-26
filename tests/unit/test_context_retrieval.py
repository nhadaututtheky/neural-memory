"""Tests for context-dependent retrieval scoring."""

from __future__ import annotations

import pytest

from neural_memory.engine.context_retrieval import (
    ContextFingerprint,
    ContextFingerprintStep,
    build_context_fingerprint,
    context_match_score,
)


class TestContextFingerprint:
    def test_defaults(self) -> None:
        fp = ContextFingerprint()
        assert fp.project_name == ""
        assert fp.dominant_topics == ()
        assert fp.active_entities == ()

    def test_to_dict_round_trip(self) -> None:
        fp = ContextFingerprint(
            project_name="myproject",
            dominant_topics=("python", "async"),
            active_entities=("UserService",),
        )
        d = fp.to_dict()
        restored = ContextFingerprint.from_dict(d)
        assert restored == fp

    def test_from_dict_empty(self) -> None:
        assert ContextFingerprint.from_dict({}) == ContextFingerprint()
        assert ContextFingerprint.from_dict(None) == ContextFingerprint()  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        fp = ContextFingerprint()
        with pytest.raises(AttributeError):
            fp.project_name = "x"  # type: ignore[misc]


class TestBuildContextFingerprint:
    def test_basic(self) -> None:
        fp = build_context_fingerprint(
            tags={"python", "async", "testing"},
            entities=["UserService", "DB"],
            project_name="neural-memory",
        )
        assert fp.project_name == "neural-memory"
        assert "async" in fp.dominant_topics
        assert "UserService" in fp.active_entities

    def test_empty_tags(self) -> None:
        fp = build_context_fingerprint(tags=set())
        assert fp.dominant_topics == ()

    def test_truncates_at_10(self) -> None:
        tags = {f"tag{i}" for i in range(20)}
        fp = build_context_fingerprint(tags=tags)
        assert len(fp.dominant_topics) == 10

    def test_entities_truncated(self) -> None:
        entities = [f"ent{i}" for i in range(20)]
        fp = build_context_fingerprint(tags=set(), entities=entities)
        assert len(fp.active_entities) == 10


class TestContextMatchScore:
    def test_no_context_neutral(self) -> None:
        a = ContextFingerprint()
        b = ContextFingerprint()
        assert context_match_score(a, b) == 1.0

    def test_same_project_boost(self) -> None:
        a = ContextFingerprint(project_name="myproject", dominant_topics=("x",))
        b = ContextFingerprint(project_name="myproject", dominant_topics=("y",))
        score = context_match_score(a, b)
        assert score > 1.0  # project match gives +0.2

    def test_different_project_penalty(self) -> None:
        a = ContextFingerprint(project_name="proj-a", dominant_topics=("x",))
        b = ContextFingerprint(project_name="proj-b", dominant_topics=("y",))
        score = context_match_score(a, b)
        assert score < 1.0  # -0.1 penalty

    def test_topic_overlap_boost(self) -> None:
        a = ContextFingerprint(dominant_topics=("python", "async", "testing"))
        b = ContextFingerprint(dominant_topics=("python", "async", "deploy"))
        score = context_match_score(a, b)
        assert score > 1.0  # Jaccard overlap boost

    def test_entity_overlap_boost(self) -> None:
        a = ContextFingerprint(
            dominant_topics=("x",),
            active_entities=("UserService", "DB"),
        )
        b = ContextFingerprint(
            dominant_topics=("x",),
            active_entities=("UserService", "Cache"),
        )
        score = context_match_score(a, b)
        assert score > 1.0

    def test_clamped_range(self) -> None:
        # Even with max boosts, should not exceed 1.5
        a = ContextFingerprint(
            project_name="same",
            dominant_topics=("a", "b", "c"),
            active_entities=("X", "Y"),
        )
        score = context_match_score(a, a)
        assert 0.5 <= score <= 1.5

    def test_case_insensitive_project(self) -> None:
        a = ContextFingerprint(project_name="MyProject", dominant_topics=("x",))
        b = ContextFingerprint(project_name="myproject", dominant_topics=("y",))
        score = context_match_score(a, b)
        assert score > 1.0  # case-insensitive match


class TestContextFingerprintStep:
    def test_name(self) -> None:
        step = ContextFingerprintStep()
        assert step.name == "context_fingerprint"

    @pytest.mark.asyncio
    async def test_disabled(self) -> None:
        from unittest.mock import MagicMock

        step = ContextFingerprintStep()
        config = MagicMock()
        config.context_retrieval_enabled = False
        ctx = MagicMock()
        result = await step.execute(ctx, MagicMock(), config)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_stores_fingerprint(self) -> None:
        from unittest.mock import MagicMock

        step = ContextFingerprintStep()
        config = MagicMock()
        config.context_retrieval_enabled = True

        ctx = MagicMock()
        ctx.effective_metadata = {}
        ctx.entity_neurons = []
        ctx.merged_tags = {"python", "async"}
        ctx.auto_tags = set()

        await step.execute(ctx, MagicMock(), config)

        assert "_context_fingerprint" in ctx.effective_metadata
        fp = ctx.effective_metadata["_context_fingerprint"]
        assert "dominant_topics" in fp
        assert "python" in fp["dominant_topics"]
