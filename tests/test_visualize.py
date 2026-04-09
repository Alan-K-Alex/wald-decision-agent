from __future__ import annotations

from pathlib import Path

from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.core.models import CalculationResult
from wald_decision_agent.rendering.visualize import VisualizationEngine


def test_visualize_creates_png(tmp_path: Path) -> None:
    settings = AppSettings(plots_dir=str(tmp_path))
    engine = VisualizationEngine(settings)
    calculation = CalculationResult(
        answer="Revenue increased.",
        findings=[],
        trace=[],
        evidence_refs=[],
        chart_data={"type": "line", "labels": ["Q1", "Q2"], "values": [120, 138], "title": "Revenue trend"},
    )

    result = engine.create("What is our current revenue trend?", calculation)

    assert result is not None
    assert result.path.exists()
    assert result.path.suffix == ".png"
