"""Tests for the Evolution dashboard REST API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neural_memory.server.routes.dashboard_api import router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def mock_storage() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def client(mock_storage: AsyncMock) -> TestClient:
    app = _make_app()
    from neural_memory.server.routes.dashboard_api import get_storage

    app.dependency_overrides[get_storage] = lambda: mock_storage
    return TestClient(app)


def _make_evolution(
    *,
    brain_name: str = "test-brain",
    proficiency_level: str = "junior",
    proficiency_index: int = 15,
    with_stage_dist: bool = True,
    with_closest: bool = False,
) -> MagicMock:
    """Create a mock BrainEvolution object."""
    evo = MagicMock()
    evo.brain_name = brain_name
    evo.proficiency_level = MagicMock(value=proficiency_level)
    evo.proficiency_index = proficiency_index
    evo.maturity_level = 0.35
    evo.plasticity = 0.6
    evo.density = 0.45
    evo.activity_score = 0.8
    evo.semantic_ratio = 0.1
    evo.reinforcement_days = 2.5
    evo.topology_coherence = 0.65
    evo.plasticity_index = 0.55
    evo.knowledge_density = 1.2
    evo.total_neurons = 100
    evo.total_synapses = 250
    evo.total_fibers = 50
    evo.fibers_at_semantic = 5
    evo.fibers_at_episodic = 20

    if with_stage_dist:
        sd = MagicMock()
        sd.short_term = 10
        sd.working = 15
        sd.episodic = 20
        sd.semantic = 5
        sd.total = 50
        evo.stage_distribution = sd
    else:
        evo.stage_distribution = None

    if with_closest:
        prog = MagicMock()
        prog.fiber_id = "fiber-abc-123"
        prog.stage = "EPISODIC"
        prog.days_in_stage = 5.2
        prog.days_required = 7.0
        prog.reinforcement_days = 2
        prog.reinforcement_required = 3
        prog.progress_pct = 0.74
        prog.next_step = "Reinforce 1 more day"
        evo.closest_to_semantic = (prog,)
    else:
        evo.closest_to_semantic = ()

    return evo


class TestEvolutionEndpoint:
    def test_success(self, client: TestClient) -> None:
        """Evolution endpoint returns valid response."""
        evo = _make_evolution()
        with (
            patch("neural_memory.unified_config.get_config") as mock_cfg,
            patch("neural_memory.engine.brain_evolution.EvolutionEngine") as mock_engine_cls,
        ):
            mock_cfg.return_value.current_brain = "test-brain"
            mock_engine = AsyncMock()
            mock_engine.analyze = AsyncMock(return_value=evo)
            mock_engine_cls.return_value = mock_engine

            resp = client.get("/api/dashboard/evolution")

        assert resp.status_code == 200
        data = resp.json()
        assert data["brain"] == "test-brain"
        assert data["proficiency_level"] == "junior"
        assert data["proficiency_index"] == 15
        assert data["total_fibers"] == 50
        assert data["fibers_at_semantic"] == 5

    def test_with_stage_distribution(self, client: TestClient) -> None:
        """Stage distribution sub-object is correctly mapped."""
        evo = _make_evolution(with_stage_dist=True)
        with (
            patch("neural_memory.unified_config.get_config") as mock_cfg,
            patch("neural_memory.engine.brain_evolution.EvolutionEngine") as mock_engine_cls,
        ):
            mock_cfg.return_value.current_brain = "test-brain"
            mock_engine = AsyncMock()
            mock_engine.analyze = AsyncMock(return_value=evo)
            mock_engine_cls.return_value = mock_engine

            resp = client.get("/api/dashboard/evolution")

        data = resp.json()
        sd = data["stage_distribution"]
        assert sd is not None
        assert sd["short_term"] == 10
        assert sd["working"] == 15
        assert sd["episodic"] == 20
        assert sd["semantic"] == 5
        assert sd["total"] == 50

    def test_with_closest_to_semantic(self, client: TestClient) -> None:
        """Closest to semantic list items are correctly mapped."""
        evo = _make_evolution(with_closest=True)
        with (
            patch("neural_memory.unified_config.get_config") as mock_cfg,
            patch("neural_memory.engine.brain_evolution.EvolutionEngine") as mock_engine_cls,
        ):
            mock_cfg.return_value.current_brain = "test-brain"
            mock_engine = AsyncMock()
            mock_engine.analyze = AsyncMock(return_value=evo)
            mock_engine_cls.return_value = mock_engine

            resp = client.get("/api/dashboard/evolution")

        data = resp.json()
        closest = data["closest_to_semantic"]
        assert len(closest) == 1
        item = closest[0]
        assert item["fiber_id"] == "fiber-abc-123"
        assert item["stage"] == "EPISODIC"
        assert item["progress_pct"] == 0.74
        assert item["next_step"] == "Reinforce 1 more day"

    def test_no_stage_distribution(self, client: TestClient) -> None:
        """No stage_distribution when brain has no data."""
        evo = _make_evolution(with_stage_dist=False)
        with (
            patch("neural_memory.unified_config.get_config") as mock_cfg,
            patch("neural_memory.engine.brain_evolution.EvolutionEngine") as mock_engine_cls,
        ):
            mock_cfg.return_value.current_brain = "test-brain"
            mock_engine = AsyncMock()
            mock_engine.analyze = AsyncMock(return_value=evo)
            mock_engine_cls.return_value = mock_engine

            resp = client.get("/api/dashboard/evolution")

        data = resp.json()
        assert data["stage_distribution"] is None

    def test_engine_failure(self, client: TestClient) -> None:
        """Returns 500 when EvolutionEngine raises."""
        with (
            patch("neural_memory.unified_config.get_config") as mock_cfg,
            patch("neural_memory.engine.brain_evolution.EvolutionEngine") as mock_engine_cls,
        ):
            mock_cfg.return_value.current_brain = "test-brain"
            mock_engine = AsyncMock()
            mock_engine.analyze = AsyncMock(side_effect=RuntimeError("analysis failed"))
            mock_engine_cls.return_value = mock_engine

            resp = client.get("/api/dashboard/evolution")

        assert resp.status_code == 500
        assert "Evolution analysis failed" in resp.json()["detail"]
