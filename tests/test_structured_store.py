from __future__ import annotations

from pathlib import Path

import pandas as pd

from wald_decision_agent.core.models import StructuredTable
from wald_decision_agent.memory.structured_store import StructuredMemoryStore


def test_structured_store_persists_table_and_catalog(tmp_path: Path) -> None:
    store = StructuredMemoryStore(tmp_path / "structured.db")
    table = StructuredTable(
        table_id="regional_actuals:Sheet1",
        source_path=Path("regional_actuals.csv"),
        source_type="spreadsheet",
        dataframe=pd.DataFrame({"Region": ["Europe"], "Actual Revenue": [74]}),
        metadata={"sheet_name": "Sheet1", "table_name": "Sheet1"},
        retrieval_text="",
    )

    entries = store.persist_tables([table])
    columns, rows = store.execute(f"SELECT region, actual_revenue FROM {entries[0].sqlite_table}")

    assert entries[0].source_file == "regional_actuals.csv"
    assert columns == ["region", "actual_revenue"]
    assert rows == [("Europe", 74)]
