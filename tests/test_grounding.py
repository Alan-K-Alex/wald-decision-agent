from __future__ import annotations

from pathlib import Path

from wald_agent_reference import LeadershipInsightAgent
from wald_agent_reference.core.config import AppSettings


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
