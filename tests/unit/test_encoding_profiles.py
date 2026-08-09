"""Tests for lean/cognitive encoding profiles (Phase 5)."""

from __future__ import annotations

import pytest

from neural_memory.engine.encoding_profiles import (
    EncodingProfile,
    build_lean_pipeline,
    parse_encoding_profile,
    resolve_profile,
)


class TestParseProfile:
    def test_none_is_cognitive_compat(self) -> None:
        assert parse_encoding_profile(None) is EncodingProfile.COGNITIVE

    def test_empty_is_cognitive(self) -> None:
        assert parse_encoding_profile("") is EncodingProfile.COGNITIVE

    def test_lean(self) -> None:
        assert parse_encoding_profile("lean") is EncodingProfile.LEAN

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            parse_encoding_profile("turbo")


class TestResolveProfile:
    def test_missing_key_cognitive_sync(self) -> None:
        r = resolve_profile(configured=None, async_enrichment=None)
        assert r.profile is EncodingProfile.COGNITIVE
        assert r.async_enrichment is False
        assert r.reason == "cognitive_compat"

    def test_lean_enables_async(self) -> None:
        r = resolve_profile(configured="lean", async_enrichment=None)
        assert r.profile is EncodingProfile.LEAN
        assert r.async_enrichment is True

    def test_decision_forces_cognitive(self) -> None:
        r = resolve_profile(configured="lean", async_enrichment=True, memory_type="decision")
        assert r.profile is EncodingProfile.COGNITIVE
        assert r.async_enrichment is False

    def test_high_priority_forces_cognitive(self) -> None:
        r = resolve_profile(configured="lean", async_enrichment=True, priority=9)
        assert r.profile is EncodingProfile.COGNITIVE


class TestLeanPipeline:
    def test_lean_has_fewer_steps_than_cognitive(self) -> None:
        lean = build_lean_pipeline()
        assert len(lean.steps) <= 10
        names = [s.name for s in lean.steps]
        assert "create_anchor" in names
        assert "build_fiber" in names
        # Heavy stages deferred
        assert "relation_extraction" not in names
        assert "temporal_binding" not in names
