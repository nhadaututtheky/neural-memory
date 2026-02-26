"""Tests for graph SVG export."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from neural_memory.cli.graph_export import (
    FiberNode,
    NeighborEdge,
    build_svg,
    layout_tree,
)

# ── Fixtures ─────────────────────────────────────────────────────


def _make_neighbor(
    content: str = "test neuron",
    synapse_type: str = "related_to",
    icon: str = "~",
    weight: float = 0.7,
) -> NeighborEdge:
    return NeighborEdge(
        content=content,
        synapse_type=synapse_type,
        icon=icon,
        weight=weight,
    )


def _make_fiber(
    label: str = "test fiber",
    neighbors: tuple[NeighborEdge, ...] = (),
) -> FiberNode:
    return FiberNode(label=label, neighbors=neighbors)


# ── build_svg tests ──────────────────────────────────────────────


class TestBuildSvg:
    def test_empty_graph_returns_valid_svg(self) -> None:
        svg = build_svg("default", [])
        assert "<svg" in svg
        assert "No memories to visualize" in svg
        # Valid XML
        ET.fromstring(svg)

    def test_single_fiber_no_neighbors(self) -> None:
        fibers = [_make_fiber("my first memory")]
        svg = build_svg("default", fibers)
        assert "<svg" in svg
        assert "my first memory" in svg
        assert "Neural Graph" in svg
        ET.fromstring(svg)

    def test_single_fiber_with_neighbors(self) -> None:
        neighbors = (
            _make_neighbor("caused by X", "caused_by", "\u2190", 0.8),
            _make_neighbor("leads to Y", "leads_to", "\u2192", 0.3),
        )
        fibers = [_make_fiber("auth system", neighbors)]
        svg = build_svg("work", fibers)
        assert "auth system" in svg
        assert "caused_by" in svg
        assert "leads_to" in svg
        ET.fromstring(svg)

    def test_multiple_fibers(self) -> None:
        fibers = [
            _make_fiber("fiber A", (_make_neighbor("neighbor A1"),)),
            _make_fiber("fiber B", (_make_neighbor("neighbor B1"),)),
            _make_fiber("fiber C"),
        ]
        svg = build_svg("multi", fibers)
        assert "fiber A" in svg
        assert "fiber B" in svg
        assert "fiber C" in svg
        ET.fromstring(svg)

    def test_brain_name_in_header(self) -> None:
        svg = build_svg("my-brain", [_make_fiber()])
        assert "my-brain" in svg

    def test_timestamp_in_header(self) -> None:
        svg = build_svg("default", [_make_fiber()], timestamp="2026-02-26 12:00")
        assert "2026-02-26 12:00" in svg

    def test_html_entities_escaped(self) -> None:
        fibers = [_make_fiber("test <script>alert(1)</script>")]
        svg = build_svg("default", fibers)
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg
        ET.fromstring(svg)

    def test_legend_present(self) -> None:
        svg = build_svg("default", [_make_fiber()])
        assert "Legend" in svg
        assert "leads_to" in svg
        assert "caused_by" in svg

    def test_svg_contains_dark_theme_colors(self) -> None:
        svg = build_svg("default", [_make_fiber()])
        assert "#0c1419" in svg  # background
        assert "#00d084" in svg  # fiber stroke / header


# ── layout_tree tests ────────────────────────────────────────────


class TestLayoutTree:
    def test_empty_fibers(self) -> None:
        layout = layout_tree("default", [])
        assert layout.rects == ()
        assert layout.lines == ()
        assert layout.width > 0
        assert layout.height > 0

    def test_single_fiber_positions(self) -> None:
        fibers = [_make_fiber("test")]
        layout = layout_tree("default", fibers)
        assert len(layout.rects) == 1
        assert len(layout.lines) == 0
        rect = layout.rects[0]
        assert rect.is_fiber is True
        assert rect.label == "test"

    def test_fiber_with_neighbors_creates_lines(self) -> None:
        neighbors = (_make_neighbor(), _make_neighbor("other"))
        fibers = [_make_fiber("parent", neighbors)]
        layout = layout_tree("default", fibers)
        # 1 fiber + 2 neighbors = 3 rects
        assert len(layout.rects) == 3
        # 2 lines connecting fiber to each neighbor
        assert len(layout.lines) == 2

    def test_no_overlapping_nodes(self) -> None:
        """Verify no two nodes share the same y position."""
        neighbors = (
            _make_neighbor("n1"),
            _make_neighbor("n2"),
            _make_neighbor("n3"),
        )
        fibers = [
            _make_fiber("f1", neighbors),
            _make_fiber("f2", (_make_neighbor("n4"),)),
        ]
        layout = layout_tree("default", fibers)

        y_positions = [(r.y, r.y + r.height) for r in layout.rects]
        for i, (y1_start, y1_end) in enumerate(y_positions):
            for j, (y2_start, y2_end) in enumerate(y_positions):
                if i != j:
                    # No overlap: one ends before other starts
                    assert y1_end <= y2_start or y2_end <= y1_start, (
                        f"Node {i} ({y1_start}-{y1_end}) overlaps node {j} ({y2_start}-{y2_end})"
                    )

    def test_edge_colors_by_weight(self) -> None:
        neighbors = (
            _make_neighbor("strong", weight=0.9),
            _make_neighbor("weak", weight=0.2),
        )
        fibers = [_make_fiber("test", neighbors)]
        layout = layout_tree("default", fibers)
        colors = [line.color for line in layout.lines]
        assert "#2196f3" in colors  # strong edge
        assert "#2a3f52" in colors  # weak edge


# ── Data model tests ─────────────────────────────────────────────


class TestDataModels:
    def test_fiber_node_frozen(self) -> None:
        fiber = _make_fiber()
        with pytest.raises(AttributeError):
            fiber.label = "mutated"  # type: ignore[misc]

    def test_neighbor_edge_frozen(self) -> None:
        edge = _make_neighbor()
        with pytest.raises(AttributeError):
            edge.weight = 0.0  # type: ignore[misc]
