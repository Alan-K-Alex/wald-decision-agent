from __future__ import annotations

from pathlib import Path

from wald_agent_reference import LeadershipInsightAgent
from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.ingestion.ingest import DocumentIngestor


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        enable_llm_formatting=False,
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
        log_file=str(tmp_path / "logs" / "agent.log"),
    )


def test_complex_xlsx_creates_multiple_structured_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = DocumentIngestor(_settings(tmp_path)).ingest_folder(root / "data" / "raw")

    board_tables = [table for table in corpus.tables.values() if table.source_path.name == "board_financial_pack.xlsx"]
    assert len(board_tables) == 3
    assert any(table.metadata["sheet_name"] == "Regional Actuals" for table in board_tables)
    assert any(table.metadata["sheet_name"] == "Regional Targets" for table in board_tables)


def test_pdf_generated_fixture_extracts_text_and_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = DocumentIngestor(_settings(tmp_path)).ingest_folder(root / "data" / "raw")

    pdf_chunks = [chunk for chunk in corpus.chunks if chunk.source_path.name == "strategy_performance_pack.pdf" and chunk.source_type == "pdf"]
    pdf_tables = [table for table in corpus.tables.values() if table.source_path.name == "strategy_performance_pack.pdf"]
    pdf_document = corpus.documents["strategy_performance_pack:document"]
    assert pdf_chunks
    assert pdf_tables
    assert pdf_document.metadata["page_count"] >= 2


def test_end_to_end_margin_plan_query_uses_complex_financial_pack(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    agent = LeadershipInsightAgent(_settings(tmp_path))

    response = agent.ask(
        question="Which region missed margin plan by the largest amount?",
        docs_path=root / "data" / "raw",
        generate_plot=True,
    )

    markdown = response.to_markdown()
    assert "Europe" in markdown
    assert "board_financial_pack.xlsx" in markdown
    assert response.plot_paths
    assert (tmp_path / "logs" / "agent.log").exists()


def test_end_to_end_docx_table_query(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    agent = LeadershipInsightAgent(_settings(tmp_path))

    response = agent.ask(
        question="Which department has the lowest execution score in the operational steering memo?",
        docs_path=root / "data" / "raw",
    )

    markdown = response.to_markdown()
    assert "Support" in markdown
    assert "operational_steering_memo.docx" in markdown


def test_end_to_end_visual_query_uses_visual_reasoning(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    agent = LeadershipInsightAgent(_settings(tmp_path))

    response = agent.ask(
        question="What does the quarterly revenue chart show?",
        docs_path=root / "data" / "raw",
        generate_plot=True,
    )

    markdown = response.to_markdown()
    assert "increasing trend" in markdown
    assert "revenue_chart.svg" in markdown
    assert response.plot_paths
