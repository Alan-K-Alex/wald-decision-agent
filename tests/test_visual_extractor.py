from __future__ import annotations

from pathlib import Path

from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.ingestion.visual_extractor import VisualExtractor


def test_visual_extractor_reads_svg_text() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = VisualExtractor(AppSettings()).parse_file(root / "data" / "raw" / "revenue_chart.svg")

    assert artifact is not None
    assert "Quarterly Revenue Trend" in artifact.summary
    assert "Q4" in artifact.extracted_text
    assert artifact.metadata["extraction_backend"] == "svg-text"
