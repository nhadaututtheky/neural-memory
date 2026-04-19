"""Tests for markdown brain export formatter."""

from __future__ import annotations

from neural_memory.cli.markdown_export import snapshot_to_markdown


def _make_snapshot(
    *,
    neurons: list | None = None,
    synapses: list | None = None,
    fibers: list | None = None,
    typed_memories: list | None = None,
) -> dict:
    return {
        "brain_id": "brain-1",
        "brain_name": "test-brain",
        "exported_at": "2026-03-02T12:00:00",
        "version": "4.52.2",
        "neurons": neurons or [],
        "synapses": synapses or [],
        "fibers": fibers or [],
        "config": {},
        "metadata": {"typed_memories": typed_memories or []},
    }


class TestSnapshotToMarkdown:
    """Test the markdown export formatter."""

    def test_empty_brain(self) -> None:
        snapshot = _make_snapshot()
        result = snapshot_to_markdown(snapshot)
        assert "# Brain: test-brain" in result
        assert "Neurons: 0" in result
        assert "Fibers: 0" in result

    def test_header_contains_counts(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "test"}],
            synapses=[{"id": "s1", "source_id": "n1", "target_id": "n2", "type": "related_to"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-02",
                }
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "Neurons: 1" in result
        assert "Synapses: 1" in result
        assert "Fibers: 1" in result

    def test_groups_by_memory_type(self) -> None:
        snapshot = _make_snapshot(
            neurons=[
                {"id": "n1", "type": "concept", "content": "PostgreSQL decision"},
                {"id": "n2", "type": "concept", "content": "Auth bug fix"},
            ],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                },
                {
                    "id": "f2",
                    "neuron_ids": ["n2"],
                    "anchor_neuron_id": "n2",
                    "created_at": "2026-03-02",
                },
            ],
            typed_memories=[
                {"fiber_id": "f1", "memory_type": "decision"},
                {"fiber_id": "f2", "memory_type": "fact"},
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "## Facts (1)" in result
        assert "## Decisions (1)" in result
        assert "Auth bug fix" in result
        assert "PostgreSQL decision" in result

    def test_uncategorized_fibers(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "Orphan memory"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                },
            ],
            typed_memories=[],  # No typed memories
        )
        result = snapshot_to_markdown(snapshot)
        assert "## Uncategorized (1)" in result
        assert "Orphan memory" in result

    def test_tag_index(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "test"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                    "auto_tags": ["auth", "bug"],
                    "agent_tags": ["urgent"],
                },
                {
                    "id": "f2",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-02",
                    "auto_tags": ["auth"],
                    "agent_tags": [],
                },
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "## Tags Index" in result
        assert "#auth" in result
        assert "2 memories" in result

    def test_statistics_table(self) -> None:
        snapshot = _make_snapshot(
            neurons=[
                {"id": "n1", "type": "concept", "content": "a"},
                {"id": "n2", "type": "entity", "content": "b"},
            ],
            synapses=[
                {"id": "s1", "source_id": "n1", "target_id": "n2", "type": "related_to"},
            ],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1", "n2"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                },
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "## Statistics" in result
        assert "| Total neurons | 2 |" in result
        assert "| Total synapses | 1 |" in result
        assert "| Total fibers | 1 |" in result

    def test_excluded_count_shown(self) -> None:
        snapshot = _make_snapshot()
        result = snapshot_to_markdown(snapshot, excluded_count=3)
        assert "Excluded 3 neurons" in result

    def test_pinned_fibers_counted(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "kb"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                    "pinned": True,
                },
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "Pinned fibers" in result

    def test_fiber_uses_summary(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "raw content"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "created_at": "2026-03-01",
                    "summary": "Nice summary here",
                },
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "Nice summary here" in result

    def test_fiber_date_from_time_start(self) -> None:
        snapshot = _make_snapshot(
            neurons=[{"id": "n1", "type": "concept", "content": "test"}],
            fibers=[
                {
                    "id": "f1",
                    "neuron_ids": ["n1"],
                    "anchor_neuron_id": "n1",
                    "time_start": "2026-02-15T08:00:00",
                    "created_at": "2026-03-01T12:00:00",
                },
            ],
        )
        result = snapshot_to_markdown(snapshot)
        assert "[2026-02-15]" in result

    def test_custom_brain_name(self) -> None:
        snapshot = _make_snapshot()
        result = snapshot_to_markdown(snapshot, brain_name="my-custom-brain")
        assert "# Brain: my-custom-brain" in result
