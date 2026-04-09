from __future__ import annotations

from pathlib import Path

from wald_decision_agent import LeadershipInsightAgent
from wald_decision_agent.core.config import AppSettings


def test_agent_answers_cross_table_variance_question(tmp_path: Path) -> None:
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
    assert "Europe has the largest negative variance" in markdown
    assert "row" in markdown
    assert "columns Region, Actual Revenue" in markdown or "columns Region, Revenue" in markdown
    assert response.plot_paths
