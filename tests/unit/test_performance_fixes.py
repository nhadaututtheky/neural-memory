"""Tests for v1.0.0 performance and intelligence improvements."""

from __future__ import annotations

import asyncio
import base64
import json
import zlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# -- Guarded imports ---------------------------------------------------------
# Some modules are being built by parallel agents and may not exist yet.

try:
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.engine.activation import ActivationResult

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

try:
    from neural_memory.engine.consolidation import (
        ConsolidationConfig,
        ConsolidationEngine,
    )

    _CONSOLIDATION_AVAILABLE = True
except ImportError:
    _CONSOLIDATION_AVAILABLE = False

try:
    from neural_memory.engine.encoder import MemoryEncoder  # noqa: F401

    _ENCODER_AVAILABLE = True
except ImportError:
    _ENCODER_AVAILABLE = False

try:
    from neural_memory.engine.retrieval import ReflexPipeline

    _RETRIEVAL_AVAILABLE = True
except ImportError:
    _RETRIEVAL_AVAILABLE = False

try:
    from neural_memory.engine.reflex_activation import CoActivation

    _COACTIVATION_AVAILABLE = True
except ImportError:
    _COACTIVATION_AVAILABLE = False

try:
    from neural_memory.storage.sqlite_versioning import _decompress_snapshot

    _VERSIONING_AVAILABLE = True
except ImportError:
    _VERSIONING_AVAILABLE = False

try:
    from neural_memory.mcp.server import handle_message

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


requires_core = pytest.mark.skipif(not _CORE_AVAILABLE, reason="core imports not available yet")
requires_retrieval = pytest.mark.skipif(
    not _RETRIEVAL_AVAILABLE, reason="ReflexPipeline not available yet"
)
requires_consolidation = pytest.mark.skipif(
    not _CONSOLIDATION_AVAILABLE, reason="consolidation imports not available yet"
)
requires_encoder = pytest.mark.skipif(
    not _ENCODER_AVAILABLE, reason="MemoryEncoder not available yet"
)
requires_versioning = pytest.mark.skipif(
    not _VERSIONING_AVAILABLE, reason="_decompress_snapshot not available yet"
)
requires_mcp = pytest.mark.skipif(not _MCP_AVAILABLE, reason="MCP server imports not available yet")
requires_coactivation = pytest.mark.skipif(
    not _COACTIVATION_AVAILABLE, reason="CoActivation not available yet"
)


# -- Retrieval: Embedding Fallback -------------------------------------------


@requires_retrieval
@requires_core
class TestEmbeddingFallback:
    """Test that embedding provider is wired into anchor finding."""

    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        storage.find_fibers_batch = AsyncMock(return_value=[])
        storage.get_neurons_batch = AsyncMock(return_value={})
        storage.get_fibers = AsyncMock(return_value=[])
        storage.get_synapses_for_neurons = AsyncMock(return_value={})
        return storage

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.max_context_tokens = 500
        config.max_spread_hops = 3
        config.activation_threshold = 0.1
        config.lateral_inhibition_k = 10
        config.lateral_inhibition_factor = 0.3
        config.reinforcement_delta = 0.05
        config.hebbian_threshold = 0.5
        config.hebbian_delta = 0.1
        config.hebbian_initial_weight = 0.3
        return config

    def test_pipeline_accepts_embedding_provider(self, mock_storage, mock_config):
        """ReflexPipeline should accept optional embedding_provider."""
        mock_provider = AsyncMock()
        pipeline = ReflexPipeline(
            storage=mock_storage,
            config=mock_config,
            embedding_provider=mock_provider,
        )
        assert pipeline._embedding_provider is mock_provider

    def test_pipeline_none_embedding_by_default(self, mock_storage, mock_config):
        """ReflexPipeline defaults to None embedding provider."""
        pipeline = ReflexPipeline(
            storage=mock_storage,
            config=mock_config,
        )
        assert pipeline._embedding_provider is None

    @pytest.mark.asyncio
    async def test_embedding_anchors_called_when_no_other_anchors(self, mock_storage, mock_config):
        """Embedding fallback fires when no substring anchors found."""
        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_provider.similarity = AsyncMock(return_value=0.85)

        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="authentication system",
            metadata={"_embedding": [0.1, 0.2, 0.3]},
        )
        mock_storage.find_neurons = AsyncMock(return_value=[neuron])

        pipeline = ReflexPipeline(
            storage=mock_storage,
            config=mock_config,
            embedding_provider=mock_provider,
        )

        anchors = await pipeline._find_embedding_anchors("auth login")
        assert len(anchors) > 0
        mock_provider.embed.assert_called_once_with("auth login")

    @pytest.mark.asyncio
    async def test_embedding_anchors_empty_without_provider(self, mock_storage, mock_config):
        """No embedding anchors when provider is None."""
        pipeline = ReflexPipeline(
            storage=mock_storage,
            config=mock_config,
            embedding_provider=None,
        )
        result = await pipeline._find_embedding_anchors("test query")
        assert result == []


# -- Retrieval: Query Expansion ----------------------------------------------


@requires_retrieval
class TestQueryExpansion:
    """Test query term expansion for better recall."""

    @pytest.fixture
    def pipeline(self):
        storage = AsyncMock()
        config = MagicMock()
        config.max_context_tokens = 500
        config.max_spread_hops = 3
        config.activation_threshold = 0.1
        config.lateral_inhibition_k = 10
        config.lateral_inhibition_factor = 0.3
        config.reinforcement_delta = 0.05
        config.hebbian_threshold = 0.5
        config.hebbian_delta = 0.1
        config.hebbian_initial_weight = 0.3
        return ReflexPipeline(storage=storage, config=config)

    def test_short_keyword_gets_suffix(self, pipeline):
        """Short keywords get expanded with common suffixes."""
        expanded = pipeline._expand_query_terms(["auth"])
        assert len(expanded) > 1
        assert "auth" in expanded

    def test_long_keyword_gets_stem(self, pipeline):
        """Long keywords with known suffixes get stemmed."""
        expanded = pipeline._expand_query_terms(["authentication"])
        stems = [t for t in expanded if t != "authentication"]
        assert len(stems) > 0

    def test_no_duplicates(self, pipeline):
        """Expansion should not produce duplicates."""
        expanded = pipeline._expand_query_terms(["test", "testing"])
        lowers = [t.lower() for t in expanded]
        assert len(lowers) == len(set(lowers))

    def test_empty_input(self, pipeline):
        """Empty input returns empty."""
        expanded = pipeline._expand_query_terms([])
        assert expanded == []


# -- Retrieval: Cluster-Aware Lateral Inhibition -----------------------------


@requires_retrieval
@requires_core
class TestClusterAwareLateralInhibition:
    """Test that lateral inhibition preserves cluster diversity."""

    @pytest.fixture
    def pipeline(self):
        storage = AsyncMock()
        config = MagicMock()
        config.max_context_tokens = 500
        config.max_spread_hops = 3
        config.activation_threshold = 0.1
        config.lateral_inhibition_k = 4
        config.lateral_inhibition_factor = 0.3
        config.reinforcement_delta = 0.05
        config.hebbian_threshold = 0.5
        config.hebbian_delta = 0.1
        config.hebbian_initial_weight = 0.3
        return ReflexPipeline(storage=storage, config=config)

    def test_preserves_diversity_across_clusters(self, pipeline):
        """Winners from different clusters should both survive."""
        activations = {
            "n1": ActivationResult(
                neuron_id="n1",
                activation_level=0.9,
                hop_distance=1,
                path=["a1"],
                source_anchor="a1",
            ),
            "n2": ActivationResult(
                neuron_id="n2",
                activation_level=0.8,
                hop_distance=1,
                path=["a1"],
                source_anchor="a1",
            ),
            "n3": ActivationResult(
                neuron_id="n3",
                activation_level=0.7,
                hop_distance=1,
                path=["a1"],
                source_anchor="a1",
            ),
            "n4": ActivationResult(
                neuron_id="n4",
                activation_level=0.85,
                hop_distance=1,
                path=["a2"],
                source_anchor="a2",
            ),
            "n5": ActivationResult(
                neuron_id="n5",
                activation_level=0.75,
                hop_distance=1,
                path=["a2"],
                source_anchor="a2",
            ),
            "n6": ActivationResult(
                neuron_id="n6",
                activation_level=0.6,
                hop_distance=1,
                path=["a2"],
                source_anchor="a2",
            ),
        }
        result = pipeline._apply_lateral_inhibition(activations)
        cluster_a1 = [
            nid
            for nid, a in result.items()
            if a.source_anchor == "a1" and a.activation_level >= 0.5
        ]
        cluster_a2 = [
            nid
            for nid, a in result.items()
            if a.source_anchor == "a2" and a.activation_level >= 0.5
        ]
        assert len(cluster_a1) >= 1
        assert len(cluster_a2) >= 1

    def test_small_set_unchanged(self, pipeline):
        """When activations <= K, nothing is suppressed."""
        activations = {
            "n1": ActivationResult(
                neuron_id="n1",
                activation_level=0.9,
                hop_distance=1,
                path=[],
                source_anchor="a1",
            ),
            "n2": ActivationResult(
                neuron_id="n2",
                activation_level=0.8,
                hop_distance=1,
                path=[],
                source_anchor="a2",
            ),
        }
        result = pipeline._apply_lateral_inhibition(activations)
        assert result == activations


# -- Consolidation: Inverted Index Merge -------------------------------------


@requires_consolidation
@requires_core
class TestInvertedIndexMerge:
    """Test that consolidation merge uses inverted index."""

    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        storage._get_brain_id = MagicMock(return_value="brain1")
        storage._current_brain_id = "brain1"
        return storage

    @pytest.mark.asyncio
    async def test_merge_overlapping_fibers(self, mock_storage):
        """Fibers sharing neurons should be merged."""
        shared_neurons = {"n1", "n2", "n3"}
        fiber_a = Fiber(
            id="fa",
            neuron_ids=shared_neurons | {"n4"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=datetime.now(),
        )
        fiber_b = Fiber(
            id="fb",
            neuron_ids=shared_neurons | {"n5"},
            synapse_ids=set(),
            anchor_neuron_id="n2",
            created_at=datetime.now(),
        )
        mock_storage.get_fibers = AsyncMock(return_value=[fiber_a, fiber_b])
        mock_storage.add_fiber = AsyncMock()
        mock_storage.delete_fiber = AsyncMock()

        config = ConsolidationConfig(merge_overlap_threshold=0.4)
        engine = ConsolidationEngine(mock_storage, config)
        report = await engine.run(strategies=["merge"])
        assert report.fibers_merged >= 2

    @pytest.mark.asyncio
    async def test_no_merge_disjoint_fibers(self, mock_storage):
        """Fibers with no shared neurons should NOT merge."""
        fiber_a = Fiber(
            id="fa",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=datetime.now(),
        )
        fiber_b = Fiber(
            id="fb",
            neuron_ids={"n3", "n4"},
            synapse_ids=set(),
            anchor_neuron_id="n3",
            created_at=datetime.now(),
        )
        mock_storage.get_fibers = AsyncMock(return_value=[fiber_a, fiber_b])

        config = ConsolidationConfig(merge_overlap_threshold=0.4)
        engine = ConsolidationEngine(mock_storage, config)
        report = await engine.run(strategies=["merge"])
        assert report.fibers_merged == 0


# -- Encoder: Meaningful Pathways --------------------------------------------


@requires_encoder
@requires_core
class TestEncoderPathways:
    """Test that _build_pathway builds multi-node fiber pathways."""

    def test_build_pathway_order(self):
        """Pathway should follow time -> entity -> concept -> anchor order."""
        from neural_memory.engine.pipeline_steps import _build_pathway

        time_n = Neuron.create(type=NeuronType.TIME, content="2024-01-01")
        entity_n = Neuron.create(type=NeuronType.ENTITY, content="Alice")
        concept_n = Neuron.create(type=NeuronType.CONCEPT, content="meeting")
        anchor_n = Neuron.create(type=NeuronType.CONCEPT, content="full content")

        pathway = _build_pathway(
            time_neurons=[time_n],
            entity_neurons=[entity_n],
            concept_neurons=[concept_n],
            anchor_neuron=anchor_n,
        )
        assert len(pathway) == 4
        assert pathway[0] == time_n.id
        assert pathway[1] == entity_n.id
        assert pathway[2] == concept_n.id
        assert pathway[3] == anchor_n.id

    def test_build_pathway_deduplicates(self):
        """Pathway should not contain duplicates."""
        from neural_memory.engine.pipeline_steps import _build_pathway

        shared = Neuron.create(type=NeuronType.CONCEPT, content="shared")
        pathway = _build_pathway(
            time_neurons=[],
            entity_neurons=[shared],
            concept_neurons=[shared],
            anchor_neuron=shared,
        )
        assert len(pathway) == 1
        assert pathway[0] == shared.id


# -- Storage: Compressed Snapshots -------------------------------------------


@requires_versioning
class TestCompressedSnapshots:
    """Test zlib compression for version snapshots."""

    def test_compress_decompress_roundtrip(self):
        """Compressed data should decompress to original."""
        original = json.dumps({"neurons": [{"id": "n1"}] * 100})
        compressed = base64.b64encode(zlib.compress(original.encode("utf-8"), level=6)).decode(
            "ascii"
        )

        result = _decompress_snapshot(compressed)
        assert result == original

    def test_decompress_legacy_uncompressed(self):
        """Legacy uncompressed data should be returned as-is."""
        legacy_data = json.dumps({"neurons": []})
        result = _decompress_snapshot(legacy_data)
        assert result == legacy_data

    def test_compression_actually_saves_space(self):
        """Compression should reduce size for typical snapshot data."""
        original = json.dumps(
            {"neurons": [{"id": f"n{i}", "content": "test" * 50} for i in range(100)]}
        )
        compressed = base64.b64encode(zlib.compress(original.encode("utf-8"), level=6)).decode(
            "ascii"
        )
        assert len(compressed) < len(original)


# -- MCP: Tool Call Timeout --------------------------------------------------


@requires_mcp
class TestMCPTimeout:
    """Test that MCP tool calls have timeout protection."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        """Timed-out tool calls should return JSON-RPC error."""

        class FakeServer:
            async def call_tool(self, name, args):
                await asyncio.sleep(60)
                return {}

        server = FakeServer()

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nmem_remember", "arguments": {"content": "test"}},
        }

        # Use a very short timeout so the test completes quickly
        with patch("neural_memory.mcp.server._TOOL_CALL_TIMEOUT", 0.5):
            result = await handle_message(server, message)
            assert result is not None
            assert "error" in result
            assert "timed out" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_normal_tool_call_succeeds(self):
        """Normal (fast) tool calls should succeed."""
        server = MagicMock()

        async def fast_tool(*args, **kwargs):
            return {"status": "ok"}

        server.call_tool = fast_tool

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nmem_remember", "arguments": {"content": "test"}},
        }

        result = await handle_message(server, message)
        assert "error" not in result
        assert result["result"]["content"][0]["type"] == "text"


# -- Batch Co-Activation Lookups ---------------------------------------------


@requires_retrieval
@requires_core
@requires_coactivation
class TestBatchCoActivation:
    """Test batch synapse lookups in co-activation processing."""

    @pytest.fixture
    def mock_storage(self):
        storage = AsyncMock()
        storage.find_fibers_batch = AsyncMock(return_value=[])
        storage.get_neurons_batch = AsyncMock(return_value={})
        storage.get_synapses_for_neurons = AsyncMock(return_value={})
        return storage

    @pytest.fixture
    def pipeline(self, mock_storage):
        config = MagicMock()
        config.max_context_tokens = 500
        config.max_spread_hops = 3
        config.activation_threshold = 0.1
        config.lateral_inhibition_k = 10
        config.lateral_inhibition_factor = 0.3
        config.reinforcement_delta = 0.05
        config.hebbian_threshold = 0.3
        config.hebbian_delta = 0.1
        config.hebbian_initial_weight = 0.3
        return ReflexPipeline(storage=mock_storage, config=config)

    @pytest.mark.asyncio
    async def test_batch_lookup_used(self, pipeline, mock_storage):
        """Should use get_synapses_for_neurons instead of per-pair queries."""
        co = CoActivation(
            neuron_ids=frozenset(("n1", "n2")),
            temporal_window_ms=100,
            binding_strength=0.8,
            source_anchors=["a1"],
        )
        activations = {
            "n1": ActivationResult(
                neuron_id="n1",
                activation_level=0.9,
                hop_distance=1,
                path=[],
                source_anchor="a1",
            ),
            "n2": ActivationResult(
                neuron_id="n2",
                activation_level=0.8,
                hop_distance=1,
                path=[],
                source_anchor="a1",
            ),
        }

        await pipeline._defer_co_activated([co], activations=activations)
        mock_storage.get_synapses_for_neurons.assert_called_once()
