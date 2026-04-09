from __future__ import annotations

from pathlib import Path

from wald_decision_agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.memory.memory_backends import MemoryBackend


def test_agent_abstains_when_evidence_is_missing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
    )
    agent = LeadershipInsightAgent(settings)

    response = agent.ask(
        question="What is our EBITDA trend across all subsidiaries?",
        docs_path=root / "data" / "raw",
    )

    assert "Insufficient evidence" in response.executive_summary
    assert any("hallucination" in finding.lower() or "grounded support" in finding.lower() for finding in response.key_findings)


def test_agent_uses_explicit_abstention_language_for_out_of_context_question(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
    )
    agent = LeadershipInsightAgent(settings)

    response = agent.ask(
        question="What is the leadership stance on quantum computing investment?",
        docs_path=root / "data" / "raw",
    )

    assert "do not have enough grounded context" in response.executive_summary.lower()
    assert len(response.evidence) == 0
    assert len(response.source_references) == 0
    assert any("abstaining" in finding.lower() for finding in response.key_findings)


def test_agent_includes_plot_markdown_when_generated(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
    )
    agent = LeadershipInsightAgent(settings)

    response = agent.ask(
        question="Which region missed revenue plan by the largest amount?",
        docs_path=root / "data" / "raw",
        generate_plot=True,
    )

    markdown = response.to_markdown()
    assert "![plot](" in markdown
    assert "[" in markdown and "](" in markdown


def test_agent_can_use_supermemory_for_narrative_retrieval(monkeypatch, tmp_path: Path) -> None:
    class StubMemoryBackend(MemoryBackend):
        def sync_document(self, document):
            return None

        def sync_table(self, table):
            return None

        def sync_visual(self, visual):
            return None

        def sync_chunk(self, chunk):
            return None

        def search(self, query: str, limit: int) -> list[dict[str, object]]:
            return [
                {
                    "content": "Leadership noted that enterprise demand strengthened over the year.",
                    "score": 0.95,
                    "metadata": {"chunk_id": "annual_report_2024:1", "source_file": "annual_report_2024.md"},
                }
            ]

    monkeypatch.setattr("wald_decision_agent.core.tools.build_memory_backend", lambda settings: StubMemoryBackend())

    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="supermemory",
        memory_backend="supermemory",
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
    )
    agent = LeadershipInsightAgent(settings)

    response = agent.ask(
        question="What does leadership say about enterprise demand?",
        docs_path=root / "data" / "raw",
    )

    assert "enterprise demand" in response.executive_summary.lower()


def test_agent_combines_supermemory_evidence_with_sql_answer(monkeypatch, tmp_path: Path) -> None:
    class StubMemoryBackend(MemoryBackend):
        def sync_document(self, document):
            return None

        def sync_table(self, table):
            return None

        def sync_visual(self, visual):
            return None

        def sync_chunk(self, chunk):
            return None

        def search(self, query: str, limit: int) -> list[dict[str, object]]:
            return [
                {
                    "content": "Leadership highlighted continued weakness in Europe.",
                    "score": 0.95,
                    "metadata": {"chunk_id": "annual_report_2024:1", "source_file": "annual_report_2024.md"},
                }
            ]

    monkeypatch.setattr("wald_decision_agent.core.tools.build_memory_backend", lambda settings: StubMemoryBackend())

    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        retrieval_backend="supermemory",
        memory_backend="supermemory",
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
    )
    agent = LeadershipInsightAgent(settings)

    response = agent.ask(
        question="Which region missed revenue plan by the largest amount?",
        docs_path=root / "data" / "raw",
    )

    assert "Europe has the largest negative variance" in response.executive_summary
    assert any("annual_report_2024.md" in evidence for evidence in response.evidence)
    assert any("board_financial_pack.xlsx" in reference for reference in response.source_references)
