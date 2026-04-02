from __future__ import annotations

from pathlib import Path

from docx import Document

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.ingestion.ingest import DocumentIngestor


def test_ingestion_supports_docx_and_csv(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Update\nRevenue expanded.\n", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text(
        "Department,Performance Score\nSales,88\nSupport,61\n",
        encoding="utf-8",
    )

    doc = Document()
    doc.add_paragraph("Support continued to underperform in the quarter.")
    table = doc.add_table(rows=3, cols=2)
    table.rows[0].cells[0].text = "Department"
    table.rows[0].cells[1].text = "Margin"
    table.rows[1].cells[0].text = "Sales"
    table.rows[1].cells[1].text = "29"
    table.rows[2].cells[0].text = "Support"
    table.rows[2].cells[1].text = "12"
    doc.save(tmp_path / "board_update.docx")

    ingestor = DocumentIngestor(AppSettings())
    corpus = ingestor.ingest_folder(tmp_path)

    assert any(chunk.source_type == "docx" for chunk in corpus.chunks)
    assert any(chunk.metadata.get("table_id") for chunk in corpus.chunks)
    assert any(table.source_type == "docx_table" for table in corpus.tables.values())
