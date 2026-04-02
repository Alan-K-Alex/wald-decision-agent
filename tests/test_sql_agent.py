from __future__ import annotations

from pathlib import Path

import pandas as pd

from wald_agent_reference.core.models import StructuredTable
from wald_agent_reference.memory.structured_store import StructuredMemoryStore
from wald_agent_reference.reasoning.sql_agent import SQLQueryAgent


def test_sql_agent_joins_actuals_and_targets_for_variance(tmp_path: Path) -> None:
    store = StructuredMemoryStore(tmp_path / "structured.db")
    actuals = StructuredTable(
        table_id="regional_actuals:Sheet1",
        source_path=Path("regional_actuals.csv"),
        source_type="spreadsheet",
        dataframe=pd.DataFrame(
            {
                "Region": ["North America", "Europe", "APAC"],
                "Actual Revenue": [92, 74, 58],
            }
        ),
        metadata={"sheet_name": "Sheet1", "table_name": "Sheet1"},
        retrieval_text="",
    )
    targets = StructuredTable(
        table_id="regional_targets:Sheet1",
        source_path=Path("regional_targets.csv"),
        source_type="spreadsheet",
        dataframe=pd.DataFrame(
            {
                "Region": ["North America", "Europe", "APAC"],
                "Revenue Target": [96, 79, 61],
            }
        ),
        metadata={"sheet_name": "Sheet1", "table_name": "Sheet1"},
        retrieval_text="",
    )

    catalog = store.persist_tables([actuals, targets])
    result = SQLQueryAgent(store).answer("Which region missed revenue plan by the largest amount?", catalog)

    assert result is not None
    assert result.answer.startswith("Europe")
    assert any("regional_actuals.csv" in step for step in result.trace)


def test_sql_agent_aggregates_open_high_risks_by_owner(tmp_path: Path) -> None:
    store = StructuredMemoryStore(tmp_path / "structured.db")
    risks = StructuredTable(
        table_id="risk_register:Sheet1",
        source_path=Path("risk_register.csv"),
        source_type="spreadsheet",
        dataframe=pd.DataFrame(
            {
                "Risk Category": ["Support Bottleneck", "Hiring Delay", "Margin Leakage"],
                "Severity": ["High", "High", "High"],
                "Status": ["Open", "Open", "Open"],
                "Owner": ["Operations", "Engineering", "Support"],
            }
        ),
        metadata={"sheet_name": "Sheet1", "table_name": "Sheet1"},
        retrieval_text="",
    )
    catalog = store.persist_tables([risks])
    result = SQLQueryAgent(store).answer("Which owner has the most open high risks?", catalog)

    assert result is not None
    assert "open high-severity risks" in result.answer.lower()
