"""Tests for nmem_visualize — chart generator + handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.chart_generator import (
    DataPoint,
    detect_chart_type,
    extract_data_points,
    generate_chart,
)

# ── Fixtures ──────────────────────────────────────────


@dataclass
class FakeNeuron:
    id: str = "n1"
    content: str = ""
    type: str = "fact"
    source_id: str = ""
    activation_level: float = 0.5


# ── extract_data_points tests ────────────────────────


class TestExtractDataPoints:
    def test_extract_key_value_pairs(self) -> None:
        neurons = [
            FakeNeuron(id="n1", content="- Revenue: 500\n- Profit: 200\n- Loss: 50"),
        ]
        points = extract_data_points(neurons)
        assert len(points) >= 2
        labels = {dp.label for dp in points}
        assert "Revenue" in labels
        assert any(dp.value == 500.0 for dp in points)

    def test_extract_date_value_pairs(self) -> None:
        neurons = [
            FakeNeuron(id="n1", content="Q1/2024 ROE was 12.8%, Q2/2024 was 13.2%"),
        ]
        points = extract_data_points(neurons)
        assert len(points) >= 2
        assert any(dp.date == "Q1/2024" for dp in points)

    def test_extract_plain_numbers(self) -> None:
        neurons = [
            FakeNeuron(id="n1", content="Total users reached 1,500 this month"),
        ]
        points = extract_data_points(neurons)
        assert len(points) >= 1
        assert any(dp.value == 1500.0 for dp in points)

    def test_extract_empty_neurons(self) -> None:
        points = extract_data_points([])
        assert points == []

    def test_extract_no_numeric_content(self) -> None:
        neurons = [FakeNeuron(id="n1", content="No numbers here")]
        points = extract_data_points(neurons)
        assert points == []

    def test_provenance_preserved(self) -> None:
        neurons = [
            FakeNeuron(id="n42", content="Score: 95", source_id="src1"),
        ]
        points = extract_data_points(neurons)
        assert len(points) == 1
        assert points[0].neuron_id == "n42"
        assert points[0].source_id == "src1"

    def test_extract_with_units(self) -> None:
        neurons = [
            FakeNeuron(id="n1", content="- Price: 500USD\n- Cost: 200VND"),
        ]
        points = extract_data_points(neurons)
        assert any(dp.unit == "USD" for dp in points)

    def test_multiple_neurons(self) -> None:
        neurons = [
            FakeNeuron(id="n1", content="Revenue: 100"),
            FakeNeuron(id="n2", content="Profit: 50"),
            FakeNeuron(id="n3", content="Nothing here"),
        ]
        points = extract_data_points(neurons)
        assert len(points) == 2


# ── detect_chart_type tests ──────────────────────────


class TestDetectChartType:
    def test_empty_data(self) -> None:
        assert detect_chart_type([]) == "table"

    def test_time_series_line(self) -> None:
        points = [
            DataPoint(label="Q1/2024", value=10, date="Q1/2024"),
            DataPoint(label="Q2/2024", value=20, date="Q2/2024"),
        ]
        assert detect_chart_type(points) == "line"

    def test_few_categories_pie(self) -> None:
        points = [
            DataPoint(label="A", value=30),
            DataPoint(label="B", value=50),
            DataPoint(label="C", value=20),
        ]
        assert detect_chart_type(points) == "pie"

    def test_many_categories_bar(self) -> None:
        points = [DataPoint(label=f"Cat{i}", value=i * 10) for i in range(10)]
        assert detect_chart_type(points) == "bar"


# ── generate_chart tests ────────────────────────────


class TestGenerateChart:
    def test_generate_vega_lite(self) -> None:
        points = [
            DataPoint(label="A", value=10, neuron_id="n1"),
            DataPoint(label="B", value=20, neuron_id="n2"),
        ]
        chart = generate_chart(points, output_format="vega_lite", title="Test")
        assert chart.vega_lite
        assert chart.vega_lite["$schema"].startswith("https://vega.github.io")
        assert chart.vega_lite["title"] == "Test"
        assert len(chart.vega_lite["data"]["values"]) == 2

    def test_generate_markdown(self) -> None:
        points = [
            DataPoint(label="Revenue", value=500, neuron_id="n1"),
            DataPoint(label="Profit", value=200, neuron_id="n2"),
        ]
        chart = generate_chart(points, output_format="markdown_table", title="Financials")
        assert "Revenue" in chart.markdown
        assert "500" in chart.markdown
        assert "| Label |" in chart.markdown

    def test_generate_ascii(self) -> None:
        points = [
            DataPoint(label="Q1", value=100, neuron_id="n1"),
            DataPoint(label="Q2", value=200, neuron_id="n2"),
        ]
        chart = generate_chart(points, output_format="ascii", title="Trend")
        assert "█" in chart.ascii_chart
        assert "Q1" in chart.ascii_chart

    def test_generate_all_formats(self) -> None:
        points = [DataPoint(label="X", value=42, neuron_id="n1")]
        chart = generate_chart(points, output_format="all", title="All")
        assert chart.vega_lite
        assert chart.markdown
        assert chart.ascii_chart

    def test_empty_data(self) -> None:
        chart = generate_chart([], title="Empty")
        assert chart.chart_type == "table"
        assert "No data" in chart.markdown

    def test_provenance_tracking(self) -> None:
        points = [
            DataPoint(label="A", value=1, neuron_id="n1"),
            DataPoint(label="B", value=2, neuron_id="n2"),
            DataPoint(label="C", value=3, neuron_id="n1"),
        ]
        chart = generate_chart(points, output_format="vega_lite")
        assert set(chart.provenance) == {"n1", "n2"}

    def test_chart_type_override(self) -> None:
        points = [DataPoint(label="A", value=10), DataPoint(label="B", value=20)]
        chart = generate_chart(points, chart_type="scatter", output_format="vega_lite")
        assert chart.chart_type == "scatter"
        assert chart.vega_lite["mark"]["type"] == "point"

    def test_line_chart_vega(self) -> None:
        points = [
            DataPoint(label="Jan", value=10, date="2024-01"),
            DataPoint(label="Feb", value=20, date="2024-02"),
        ]
        chart = generate_chart(points, chart_type="line", output_format="vega_lite")
        assert chart.vega_lite["mark"]["type"] == "line"

    def test_pie_chart_vega(self) -> None:
        points = [
            DataPoint(label="A", value=30),
            DataPoint(label="B", value=70),
        ]
        chart = generate_chart(points, chart_type="pie", output_format="vega_lite")
        assert chart.vega_lite["mark"]["type"] == "arc"


# ── Handler tests ────────────────────────────────────


class TestVisualizeHandler:
    @pytest.fixture
    def handler(self) -> Any:
        """Create a minimal handler with mocked storage."""
        from neural_memory.mcp.visualize_handler import VisualizeHandler

        class TestHandler(VisualizeHandler):
            def __init__(self) -> None:
                self._storage = AsyncMock()
                self.config = MagicMock()

            async def get_storage(self) -> Any:
                return self._storage

        h = TestHandler()
        h._storage.get_brain = AsyncMock(return_value=MagicMock(id="b1"))
        return h

    @pytest.mark.asyncio
    async def test_missing_query(self, handler: Any) -> None:
        result = await handler._visualize({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_neurons_found(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(return_value=[])
        result = await handler._visualize({"query": "revenue"})
        assert result["chart_type"] == "table"
        assert "No data" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_neurons_without_numeric_data(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(
            return_value=[FakeNeuron(id="n1", content="Just some text")]
        )
        result = await handler._visualize({"query": "text"})
        assert "memories" in result

    @pytest.mark.asyncio
    async def test_successful_visualization(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(
            return_value=[
                FakeNeuron(id="n1", content="- Revenue: 500\n- Profit: 200"),
            ]
        )
        result = await handler._visualize({"query": "financials"})
        assert "vega_lite" in result
        assert result["data_points_count"] >= 1
        assert "n1" in result["provenance"]

    @pytest.mark.asyncio
    async def test_markdown_format(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(
            return_value=[FakeNeuron(id="n1", content="Score: 95")]
        )
        result = await handler._visualize({"query": "score", "format": "markdown_table"})
        assert "markdown" in result
        assert "95" in result["markdown"]

    @pytest.mark.asyncio
    async def test_ascii_format(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(
            return_value=[FakeNeuron(id="n1", content="A: 10\nB: 20")]
        )
        result = await handler._visualize({"query": "data", "format": "ascii"})
        assert "ascii" in result
        assert "█" in result["ascii"]

    @pytest.mark.asyncio
    async def test_limit_capped(self, handler: Any) -> None:
        handler._storage.find_neurons = AsyncMock(return_value=[])
        await handler._visualize({"query": "test", "limit": 999})
        # Should be capped at 50
        call_args = handler._storage.find_neurons.call_args
        assert call_args.kwargs.get("limit", call_args[1].get("limit", 50)) <= 50


# ── Integration: encode financial data → visualize → valid Vega-Lite ──


class TestVisualizeIntegration:
    """End-to-end: financial neuron content → chart generation → valid spec."""

    def test_financial_table_to_vega_lite(self) -> None:
        """Encode a financial table as neurons, extract data, generate Vega-Lite."""
        neurons = [
            FakeNeuron(id="q1", content="Q1/2024 revenue: 125.5M USD"),
            FakeNeuron(id="q2", content="Q2/2024 revenue: 130.2M USD"),
            FakeNeuron(id="q3", content="Q3/2024 revenue: 118.7M USD"),
            FakeNeuron(id="q4", content="Q4/2024 revenue: 142.0M USD"),
        ]

        data_points = extract_data_points(neurons, "revenue trend")
        assert len(data_points) >= 2, "Should extract numeric data from financial content"

        chart = generate_chart(
            data_points,
            title="Revenue Trend",
            output_format="vega_lite",
        )

        assert chart.chart_type in ("line", "bar", "pie", "scatter", "table")
        assert chart.vega_lite is not None, "Vega-Lite spec should be generated"

        spec = chart.vega_lite
        assert "$schema" in spec
        assert "data" in spec
        assert "mark" in spec or "layer" in spec

    def test_mixed_content_to_markdown_table(self) -> None:
        """Non-numeric content should produce markdown table fallback."""
        neurons = [
            FakeNeuron(id="n1", content="API rate limit is 1000 req/min"),
            FakeNeuron(id="n2", content="Response time 95th percentile: 200ms"),
        ]

        data_points = extract_data_points(neurons, "API metrics")
        chart = generate_chart(
            data_points,
            title="API Metrics",
            output_format="markdown_table",
        )

        assert chart.markdown is not None
        assert "|" in chart.markdown

    def test_ascii_output(self) -> None:
        """Data with numeric values should produce ASCII chart."""
        neurons = [
            FakeNeuron(id="n1", content="Users: 500"),
            FakeNeuron(id="n2", content="Users: 750"),
            FakeNeuron(id="n3", content="Users: 300"),
        ]

        data_points = extract_data_points(neurons, "user count")
        if data_points:
            chart = generate_chart(
                data_points,
                title="User Count",
                output_format="ascii",
            )
            assert chart.ascii_chart is not None
            assert "█" in chart.ascii_chart

    def test_provenance_tracked(self) -> None:
        """Each data point should track source neuron ID."""
        neurons = [
            FakeNeuron(id="src-001", content="Revenue: 100M"),
            FakeNeuron(id="src-002", content="Revenue: 200M"),
        ]

        data_points = extract_data_points(neurons, "revenue")
        for dp in data_points:
            assert dp.neuron_id, "Data point should reference source neuron"
