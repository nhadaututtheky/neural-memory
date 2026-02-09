"""Tests for doc_trainer: doc-to-brain training pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.doc_trainer import DocTrainer, TrainingConfig, TrainingResult


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage that accepts neuron/synapse/fiber operations."""
    storage = AsyncMock()
    storage.add_neuron = AsyncMock()
    storage.add_synapse = AsyncMock()
    storage.add_fiber = AsyncMock()
    storage.save_maturation = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers = AsyncMock(return_value=[])
    storage.disable_auto_save = MagicMock()
    storage.enable_auto_save = MagicMock()
    storage.batch_save = AsyncMock()
    storage._current_brain_id = "test-brain"
    return storage


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock BrainConfig."""
    return MagicMock()


class TestTrainingConfig:
    """Tests for TrainingConfig defaults."""

    def test_defaults(self) -> None:
        """Verify default configuration values."""
        tc = TrainingConfig()
        assert tc.domain_tag == ""
        assert tc.brain_name == ""
        assert tc.min_chunk_words == 20
        assert tc.max_chunk_words == 500
        assert tc.memory_type == "reference"
        assert tc.consolidate is True
        assert tc.extensions == (".md",)

    def test_custom_values(self) -> None:
        """Custom values are preserved."""
        tc = TrainingConfig(
            domain_tag="react",
            brain_name="react-expert",
            min_chunk_words=10,
            max_chunk_words=300,
            memory_type="fact",
            consolidate=False,
            extensions=(".md", ".mdx"),
        )
        assert tc.domain_tag == "react"
        assert tc.brain_name == "react-expert"
        assert tc.consolidate is False


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_frozen(self) -> None:
        """TrainingResult is immutable."""
        result = TrainingResult(
            files_processed=1,
            chunks_encoded=5,
            chunks_skipped=0,
            neurons_created=10,
            synapses_created=20,
            hierarchy_synapses=3,
            enrichment_synapses=2,
            brain_name="test",
        )
        with pytest.raises(AttributeError):
            result.files_processed = 99  # type: ignore[misc]

    def test_chunks_failed_default(self) -> None:
        """chunks_failed defaults to 0."""
        result = TrainingResult(
            files_processed=1,
            chunks_encoded=5,
            chunks_skipped=0,
        )
        assert result.chunks_failed == 0
        assert result.brain_name == "current"

    def test_chunks_failed_set(self) -> None:
        """chunks_failed can be set explicitly."""
        result = TrainingResult(
            files_processed=1,
            chunks_encoded=5,
            chunks_skipped=2,
            chunks_failed=3,
        )
        assert result.chunks_failed == 3


class TestDocTrainer:
    """Tests for DocTrainer pipeline."""

    @pytest.mark.asyncio
    async def test_train_single_file(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Training a single file encodes chunks and creates hierarchy."""
        md = (
            "# Guide\n\n"
            + " ".join(["This is the guide content word"] * 8)
            + "\n\n## Setup\n\n"
            + " ".join(["Setup instructions for the project word"] * 8)
        )
        doc = tmp_path / "guide.md"
        doc.write_text(md, encoding="utf-8")

        trainer = DocTrainer(mock_storage, mock_config)

        # Patch consolidation to avoid full engine run
        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_file(doc)

        assert result.files_processed == 1
        assert result.chunks_encoded >= 1
        assert result.neurons_created >= 1
        assert result.chunks_failed == 0
        # add_neuron called for encoded chunks + heading hierarchy neurons
        assert mock_storage.add_neuron.call_count >= 1

    @pytest.mark.asyncio
    async def test_train_directory(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Training a directory processes multiple files."""
        for name in ("a.md", "b.md"):
            content = f"# {name}\n\n" + " ".join(["content word here now"] * 8)
            (tmp_path / name).write_text(content, encoding="utf-8")

        trainer = DocTrainer(mock_storage, mock_config)
        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_directory(tmp_path)

        assert result.files_processed == 2
        assert result.chunks_encoded >= 2

    @pytest.mark.asyncio
    async def test_domain_tag_applied(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Domain tag is passed to encoder via tags."""
        md = "# Test\n\n" + " ".join(["content for domain tagging test word"] * 8)
        (tmp_path / "test.md").write_text(md, encoding="utf-8")

        tc = TrainingConfig(domain_tag="kubernetes", consolidate=False)
        trainer = DocTrainer(mock_storage, mock_config)

        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_file(tmp_path / "test.md", tc)

        assert result.chunks_encoded >= 1

    @pytest.mark.asyncio
    async def test_heading_hierarchy_synapses(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """CONTAINS synapses are created for heading hierarchy."""
        md = (
            "# Root\n\n"
            + " ".join(["root content text word"] * 8)
            + "\n\n## Child\n\n"
            + " ".join(["child content text word"] * 8)
        )
        (tmp_path / "hierarchy.md").write_text(md, encoding="utf-8")

        trainer = DocTrainer(mock_storage, mock_config)
        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_file(tmp_path / "hierarchy.md")

        # Should have hierarchy synapses: Root → Child, Child → chunk anchor(s)
        assert result.hierarchy_synapses >= 1

    @pytest.mark.asyncio
    async def test_consolidation_runs(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """ENRICH consolidation runs when enabled."""
        md = "# Test\n\n" + " ".join(["content for consolidation test word"] * 8)
        (tmp_path / "test.md").write_text(md, encoding="utf-8")

        tc = TrainingConfig(consolidate=True)
        trainer = DocTrainer(mock_storage, mock_config)

        with patch.object(trainer, "_run_enrichment", return_value=5) as mock_enrich:
            result = await trainer.train_file(tmp_path / "test.md", tc)

        mock_enrich.assert_called_once()
        assert result.enrichment_synapses == 5

    @pytest.mark.asyncio
    async def test_consolidation_skipped_when_disabled(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """ENRICH consolidation is skipped when disabled."""
        md = "# Test\n\n" + " ".join(["content for no consolidation test word"] * 8)
        (tmp_path / "test.md").write_text(md, encoding="utf-8")

        tc = TrainingConfig(consolidate=False)
        trainer = DocTrainer(mock_storage, mock_config)

        with patch.object(trainer, "_run_enrichment", return_value=0) as mock_enrich:
            result = await trainer.train_file(tmp_path / "test.md", tc)

        mock_enrich.assert_not_called()
        assert result.enrichment_synapses == 0

    @pytest.mark.asyncio
    async def test_empty_directory(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Empty directory returns zero-count result gracefully."""
        trainer = DocTrainer(mock_storage, mock_config)
        result = await trainer.train_directory(tmp_path)

        assert result.files_processed == 0
        assert result.chunks_encoded == 0
        assert result.neurons_created == 0
        assert result.chunks_failed == 0
        assert result.brain_name == "current"

    @pytest.mark.asyncio
    async def test_unreadable_file_skipped(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Files that can't be read are skipped with a warning."""
        bad = tmp_path / "binary.md"
        bad.write_bytes(b"\xff\xfe" + b"\x00" * 100)

        trainer = DocTrainer(mock_storage, mock_config)
        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_file(bad)

        # Should not crash, just return 0 chunks
        assert result.files_processed == 0 or result.chunks_encoded == 0

    @pytest.mark.asyncio
    async def test_per_chunk_error_isolation(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """One failing chunk doesn't abort the entire batch."""
        md = (
            "# Good Section One\n\n"
            + " ".join(["first section has good content for testing"] * 6)
            + "\n\n# Good Section Two\n\n"
            + " ".join(["second section also has good content word"] * 6)
        )
        (tmp_path / "test.md").write_text(md, encoding="utf-8")

        trainer = DocTrainer(mock_storage, mock_config)

        call_count = 0
        original_encode = trainer._encoder.encode

        async def encode_fail_first(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated encoding failure")
            return await original_encode(*args, **kwargs)

        with (
            patch.object(trainer._encoder, "encode", side_effect=encode_fail_first),
            patch.object(trainer, "_run_enrichment", return_value=0),
        ):
            result = await trainer.train_file(tmp_path / "test.md")

        # First chunk failed, second succeeded
        assert result.chunks_failed >= 1
        assert result.chunks_encoded >= 1

    @pytest.mark.asyncio
    async def test_encoder_called_with_skip_flags(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Encoder is called with skip_conflicts=True, skip_time_neurons=True."""
        md = "# Test\n\n" + " ".join(["content checking encoder skip flags word"] * 8)
        (tmp_path / "test.md").write_text(md, encoding="utf-8")

        trainer = DocTrainer(mock_storage, mock_config)

        encode_calls: list[dict[str, object]] = []
        original_encode = trainer._encoder.encode

        async def capture_encode(*args: object, **kwargs: object) -> object:
            encode_calls.append(dict(kwargs))
            return await original_encode(*args, **kwargs)

        with (
            patch.object(trainer._encoder, "encode", side_effect=capture_encode),
            patch.object(trainer, "_run_enrichment", return_value=0),
        ):
            await trainer.train_file(tmp_path / "test.md")

        assert len(encode_calls) >= 1
        for call_kwargs in encode_calls:
            assert call_kwargs.get("skip_conflicts") is True
            assert call_kwargs.get("skip_time_neurons") is True

    @pytest.mark.asyncio
    async def test_custom_extensions(
        self, mock_storage: AsyncMock, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Custom extensions filter correctly in directory training."""
        (tmp_path / "doc.md").write_text(
            "# MD\n\n" + " ".join(["markdown content here word"] * 8),
            encoding="utf-8",
        )
        (tmp_path / "doc.rst").write_text(
            "# RST\n\n" + " ".join(["restructured text content word"] * 8),
            encoding="utf-8",
        )

        tc = TrainingConfig(extensions=(".rst",), consolidate=False)
        trainer = DocTrainer(mock_storage, mock_config)

        with patch.object(trainer, "_run_enrichment", return_value=0):
            result = await trainer.train_directory(tmp_path, tc)

        # Only .rst file should be processed
        assert result.files_processed == 1
