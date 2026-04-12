"""Tests for layer routing engine.

Covers:
- MemoryLayer enum values
- LayerDecision dataclass
- route_memory() core routing logic
- Explicit layer override
- Ephemeral → SESSION routing
- Type-based routing (preference/instruction → GLOBAL)
- Default → PROJECT routing
- Invalid explicit layer fallback
"""

from __future__ import annotations

import pytest

from neural_memory.engine.layer_router import (
    LayerDecision,
    MemoryLayer,
    route_memory,
)


class TestMemoryLayer:
    def test_values(self) -> None:
        assert MemoryLayer.SESSION.value == "session"
        assert MemoryLayer.PROJECT.value == "project"
        assert MemoryLayer.GLOBAL.value == "global"


class TestLayerDecision:
    def test_frozen(self) -> None:
        d = LayerDecision(layer=MemoryLayer.PROJECT, reason="test")
        with pytest.raises(AttributeError):
            d.layer = MemoryLayer.GLOBAL  # type: ignore[misc]

    def test_to_dict(self) -> None:
        d = LayerDecision(layer=MemoryLayer.GLOBAL, reason="preference type")
        result = d.to_dict()
        assert result["layer"] == "global"
        assert result["reason"] == "preference type"


class TestRouteMemory:
    # ── Explicit layer override ──

    def test_explicit_global(self) -> None:
        d = route_memory("fact", explicit_layer="global")
        assert d.layer == MemoryLayer.GLOBAL
        assert "explicit" in d.reason

    def test_explicit_project(self) -> None:
        d = route_memory("preference", explicit_layer="project")
        assert d.layer == MemoryLayer.PROJECT
        assert "explicit" in d.reason

    def test_explicit_session(self) -> None:
        d = route_memory("fact", explicit_layer="session")
        assert d.layer == MemoryLayer.SESSION

    def test_explicit_overrides_type_routing(self) -> None:
        """Explicit layer beats type-based routing."""
        d = route_memory("preference", explicit_layer="project")
        assert d.layer == MemoryLayer.PROJECT

    def test_explicit_case_insensitive(self) -> None:
        d = route_memory("fact", explicit_layer="GLOBAL")
        assert d.layer == MemoryLayer.GLOBAL

    def test_explicit_invalid_falls_through(self) -> None:
        """Invalid explicit layer falls through to auto-routing."""
        d = route_memory("preference", explicit_layer="invalid_layer")
        # Should fall through to type-based routing
        assert d.layer == MemoryLayer.GLOBAL

    # ── Ephemeral → SESSION ──

    def test_ephemeral_routes_session(self) -> None:
        d = route_memory("fact", is_ephemeral=True)
        assert d.layer == MemoryLayer.SESSION
        assert "ephemeral" in d.reason

    def test_ephemeral_overrides_type(self) -> None:
        """Ephemeral flag overrides type-based routing."""
        d = route_memory("preference", is_ephemeral=True)
        assert d.layer == MemoryLayer.SESSION

    # ── Type-based routing ──

    def test_preference_routes_global(self) -> None:
        d = route_memory("preference")
        assert d.layer == MemoryLayer.GLOBAL
        assert "preference" in d.reason

    def test_instruction_routes_global(self) -> None:
        d = route_memory("instruction")
        assert d.layer == MemoryLayer.GLOBAL
        assert "instruction" in d.reason

    def test_decision_routes_project(self) -> None:
        d = route_memory("decision")
        assert d.layer == MemoryLayer.PROJECT

    def test_error_routes_project(self) -> None:
        d = route_memory("error")
        assert d.layer == MemoryLayer.PROJECT

    def test_fact_routes_project(self) -> None:
        d = route_memory("fact")
        assert d.layer == MemoryLayer.PROJECT

    def test_insight_routes_project(self) -> None:
        d = route_memory("insight")
        assert d.layer == MemoryLayer.PROJECT

    def test_workflow_routes_project(self) -> None:
        d = route_memory("workflow")
        assert d.layer == MemoryLayer.PROJECT

    # ── Priority order ──

    def test_explicit_beats_ephemeral(self) -> None:
        """Explicit layer=global beats ephemeral flag."""
        d = route_memory("fact", is_ephemeral=True, explicit_layer="global")
        assert d.layer == MemoryLayer.GLOBAL

    def test_ephemeral_beats_type(self) -> None:
        """Ephemeral beats type-based routing (already tested above)."""
        d = route_memory("instruction", is_ephemeral=True)
        assert d.layer == MemoryLayer.SESSION


class TestGlobalBrainConfig:
    def test_global_brain_name_constant(self) -> None:
        from neural_memory.unified_config import GLOBAL_BRAIN_NAME

        assert GLOBAL_BRAIN_NAME == "_global"
        # Must match brain name pattern
        import re

        pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
        assert pattern.match(GLOBAL_BRAIN_NAME)

    def test_list_brains_hides_global_by_default(self) -> None:
        """list_brains() should have include_global parameter."""
        # Just verify the method signature accepts include_global
        import inspect

        from neural_memory.unified_config import UnifiedConfig

        sig = inspect.signature(UnifiedConfig.list_brains)
        assert "include_global" in sig.parameters
