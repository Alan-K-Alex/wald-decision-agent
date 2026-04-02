from __future__ import annotations

from pathlib import Path

import pandas as pd

from wald_agent_reference.core.models import StructuredTable
from wald_agent_reference.reasoning.calculator import CalculationEngine


def test_calculator_computes_trend_growth() -> None:
    frame = pd.DataFrame(
        {
            "Metric": ["Revenue"],
            "Q1 2024": [120],
            "Q2 2024": [138],
            "Q3 2024": [140],
            "Q4 2024": [160],
        }
    )
    table = StructuredTable(
        table_id="revenue",
        source_path=Path("revenue_trend.csv"),
        source_type="spreadsheet",
        dataframe=frame,
        metadata={"sheet_name": "Sheet1"},
        retrieval_text="",
    )

    result = CalculationEngine().calculate("What is our current revenue trend?", [table])

    assert result is not None
    assert "14.29%" in " ".join(result.trace)
    assert result.chart_data is not None


def test_calculator_ranks_underperforming_departments() -> None:
    frame = pd.DataFrame(
        {
            "Department": ["Sales", "Support", "Finance"],
            "Performance Score": [88, 61, 67],
            "Margin": [29, 12, 18],
        }
    )
    table = StructuredTable(
        table_id="scorecard",
        source_path=Path("department_scorecard.csv"),
        source_type="spreadsheet",
        dataframe=frame,
        metadata={"sheet_name": "Sheet1"},
        retrieval_text="",
    )

    result = CalculationEngine().calculate("Which departments are underperforming?", [table])

    assert result is not None
    assert result.answer.startswith("Support")
    assert any("Performance Score" in finding for finding in result.findings)
