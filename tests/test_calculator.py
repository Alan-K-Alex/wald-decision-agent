from __future__ import annotations

from pathlib import Path

import pandas as pd

from wald_decision_agent.core.models import StructuredTable
from wald_decision_agent.reasoning.calculator import CalculationEngine


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


def test_calculator_returns_region_breakdown_for_each_region_queries() -> None:
    frame = pd.DataFrame(
        {
            "Region": ["North America", "Europe", "APAC", "LATAM"],
            "Actual Revenue": [102, 80, 63, 35],
            "Actual Margin": [31, 24, 19, 13],
            "Actual Cost": [71, 56, 44, 22],
        }
    )
    table = StructuredTable(
        table_id="regional_actuals",
        source_path=Path("board_financial_pack.xlsx"),
        source_type="spreadsheet",
        dataframe=frame,
        metadata={"sheet_name": "Regional Actuals", "source_range": "A3:D7"},
        retrieval_text="",
    )

    result = CalculationEngine().calculate("actual margin and gained for each regions ?", [table])

    assert result is not None
    assert "Actual Margin by Region" in result.answer
    assert "North America Actual Margin = 31.00" in result.answer
    assert "Europe: Actual Margin = 24.00" in " ".join(result.findings)
    assert "do not see a grounded metric named `gained`" in result.answer
    assert result.chart_data is not None


def test_calculator_returns_temporal_lookup_for_valid_quarter_metric() -> None:
    frame = pd.DataFrame(
        {
            "Metric": ["Revenue", "Operating Margin"],
            "Q1 2024": [120, 18],
            "Q2 2024": [138, 20],
            "Q3 2024": [140, 19],
            "Q4 2024": [160, 22],
        }
    )
    table = StructuredTable(
        table_id="revenue_trend",
        source_path=Path("revenue_trend.csv"),
        source_type="spreadsheet",
        dataframe=frame,
        metadata={"sheet_name": "Sheet1"},
        retrieval_text="",
    )

    result = CalculationEngine().calculate("operating margin q2 ?", [table])

    assert result is not None
    assert result.answer == "Margin in Q2 2024 is 20.00."
    assert any("Q2 2024" in finding for finding in result.findings)
