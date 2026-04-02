from __future__ import annotations

from pathlib import Path

from wald_agent_reference import LeadershipInsightAgent
from wald_agent_reference.core.config import AppSettings


def test_agent_writes_execution_log(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        enable_llm_formatting=False,
        vector_backend="hash",
        vector_store_dir=str(tmp_path / "vector_store"),
        structured_store_path=str(tmp_path / "structured.db"),
        plots_dir=str(tmp_path / "plots"),
        reports_dir=str(tmp_path / "reports"),
        log_file=str(tmp_path / "logs" / "agent.log"),
    )
    agent = LeadershipInsightAgent(settings)

    agent.ask(
        question="Which region missed revenue plan by the largest amount?",
        docs_path=root / "data" / "raw",
        generate_plot=True,
    )

    log_path = tmp_path / "logs" / "agent.log"
    log_text = log_path.read_text(encoding="utf-8")
    assert "Planner route sequence" in log_text
    assert "Ingestion completed" in log_text
    assert "Generated plot at" in log_text
