from __future__ import annotations

from pathlib import Path

from wald_agent_reference.core.models import VisualArtifact
from wald_agent_reference.reasoning.visual_reasoner import VisualReasoner


def test_visual_reasoner_detects_quarterly_trend() -> None:
    visual = VisualArtifact(
        artifact_id="chart-1",
        source_path=Path("revenue_chart.svg"),
        source_type="svg",
        extracted_text="Quarterly Revenue Trend Q1 Q2 Q3 Q4 120 138 140 160",
        summary="Quarterly Revenue Trend",
        metadata={"title": "Quarterly Revenue Trend", "extraction_backend": "svg-text"},
    )

    result = VisualReasoner().answer("What does the quarterly revenue chart show?", [visual])

    assert result is not None
    assert "increasing trend" in result.answer
    assert result.chart_data is not None


def test_visual_reasoner_rejects_unsupported_metric_question() -> None:
    visual = VisualArtifact(
        artifact_id="chart-1",
        source_path=Path("revenue_chart.svg"),
        source_type="svg",
        extracted_text="Quarterly Revenue Trend Q1 Q2 Q3 Q4 120 138 140 160",
        summary="Quarterly Revenue Trend",
        metadata={"title": "Quarterly Revenue Trend", "extraction_backend": "svg-text"},
    )

    result = VisualReasoner().answer("What is our EBITDA trend across all subsidiaries?", [visual])

    assert result is None
