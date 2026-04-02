from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from wald_agent_reference.ingestion.spreadsheet_parser import SpreadsheetParser


def test_spreadsheet_parser_preserves_sheet_and_headers(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Quarterly KPIs"
    sheet.append(["Metric", "Q1 2024", "Q2 2024"])
    sheet.append(["Revenue", 120, 138])
    file_path = tmp_path / "kpis.xlsx"
    workbook.save(file_path)

    tables = SpreadsheetParser().parse_file(file_path)

    assert len(tables) == 1
    table = tables[0]
    assert table.metadata["sheet_name"] == "Quarterly KPIs"
    assert list(table.dataframe.columns) == ["Metric", "Q1 2024", "Q2 2024"]
    assert "Revenue" in table.retrieval_text
