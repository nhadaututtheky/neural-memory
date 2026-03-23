"""Tests for trainer-friendly markdown preparation."""

from __future__ import annotations

import textwrap

from neural_memory.engine.training_markdown import (
    extract_training_annotations,
    prepare_markdown_for_training,
)


class TestPrepareMarkdownForTraining:
    """Tests for canonical markdown preparation."""

    def test_maps_common_headings_to_canonical_sections(self) -> None:
        """Installation and architecture headings are regrouped predictably."""
        source = textwrap.dedent("""\
            # Cache System

            ## Architecture

            The cache system contains a coordinator, a local store, and a sync worker.
            These components coordinate writes and reads so hot data remains available.

            ## Installation

            Install the package, create the cache directory, and start the service.
            The setup flow is short, but it establishes the worker topology.
        """)

        result = prepare_markdown_for_training(source, source_name="cache.md")

        assert "## Concepts" in result.output_text
        assert "### Architecture" in result.output_text
        assert "## Procedures" in result.output_text
        assert "### Installation" in result.output_text

    def test_synthesizes_relationships_from_relational_language(self) -> None:
        """Relational sentences are lifted into an explicit relationship section."""
        source = textwrap.dedent("""\
            # Pipeline

            ## Flow

            The parser feeds the normalizer because raw markdown varies widely.
            After normalization, the chunker creates sections and links them.
        """)

        result = prepare_markdown_for_training(source, source_name="pipeline.md")

        assert "Relationships" in result.synthesized_sections
        assert "## Relationships" in result.output_text
        assert "parser feeds the normalizer" in result.output_text.lower()

    def test_lifts_frontmatter_into_visible_reference_content(self) -> None:
        """Frontmatter metadata is kept as visible markdown for training."""
        source = textwrap.dedent("""\
            ---
            title: Agent Notes
            category: docs
            ---

            ## Usage

            Use the agent carefully and keep the notes synchronized with the source.
            This section contains enough words to survive chunking during training.
        """)

        result = prepare_markdown_for_training(source, source_name="agent-notes.md")

        assert "Reference" in result.synthesized_sections
        assert "### Frontmatter Metadata" in result.output_text
        assert "- category: docs" in result.output_text

    def test_adds_overview_when_missing(self) -> None:
        """Files without an overview get a generated one."""
        source = textwrap.dedent("""\
            ## Commands

            Run the worker and inspect logs when failures appear in the queue.
            This short runbook is meant for operators who already know the basics.
        """)

        result = prepare_markdown_for_training(source, source_name="ops.md")

        assert "Overview" in result.synthesized_sections
        assert "## Overview" in result.output_text
        assert "# Ops" in result.output_text

    def test_short_sections_are_grouped_into_collected_notes(self) -> None:
        """Tiny sections are merged so they clear the trainer chunk threshold."""
        source = textwrap.dedent("""\
            # Tiny Notes

            ## Flags

            Fast mode.

            ## Limits

            Ten retries.

            ## Usage

            Use the retry command when jobs stall and confirm the worker resumes
            processing after the backoff window expires successfully.
        """)

        result = prepare_markdown_for_training(source, source_name="tiny.md")

        assert "### Collected Notes" in result.output_text
        assert "- Flags: Fast mode." in result.output_text
        assert "- Limits: Ten retries." in result.output_text

    def test_outputs_training_frontmatter_with_tags_and_labels(self) -> None:
        """Prepared markdown includes machine-readable tags and labels."""
        source = textwrap.dedent("""\
            # Cache Pipeline

            ## Architecture

            The cache pipeline uses a parser, a normalizer, and a sync worker.
            These components coordinate reads and writes across the system.
        """)

        result = prepare_markdown_for_training(source, source_name="cache-pipeline.md")

        assert result.output_text.startswith("---\n")
        assert "nm_tags:" in result.output_text
        assert "nm_labels:" in result.output_text
        assert "trainer-ready" in result.labels
        assert len(result.topic_tags) > 0

    def test_extracts_training_annotations_from_frontmatter(self) -> None:
        """Frontmatter annotations can be read back by the trainer."""
        source = textwrap.dedent("""\
            ---
            title: Cache Pipeline
            nm_tags: cache, parser, sync-worker
            nm_labels: concepts, procedures, trainer-ready
            ---

            # Cache Pipeline

            ## Usage

            Use the cache pipeline carefully when retry traffic spikes.
        """)

        annotations = extract_training_annotations(source, source_name="cache.md")

        assert annotations.title == "Cache Pipeline"
        assert annotations.topic_tags == ("cache", "parser", "sync-worker")
        assert annotations.labels == ("concepts", "procedures", "trainer-ready")
