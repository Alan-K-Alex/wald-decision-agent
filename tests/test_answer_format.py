from __future__ import annotations

from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.core.models import CalculationResult, DocumentChunk, QueryPlan, RetrievedChunk
from wald_agent_reference.reasoning.answer import AnswerComposer


def test_answer_format_contains_required_sections() -> None:
    composer = AnswerComposer(AppSettings(enable_llm_formatting=False))
    chunk = DocumentChunk(
        chunk_id="1",
        source_path=Path("annual_report_2024.md"),
        content="Revenue expanded steadily over the year.",
        source_type="text",
    )
    calculation = CalculationResult(
        answer="Revenue is increasing.",
        findings=["Latest revenue is higher than the previous quarter."],
        trace=["Growth formula executed deterministically."],
        evidence_refs=["revenue_trend.csv [Sheet1]"],
    )

    response = composer.compose(
        question="What is our current revenue trend?",
        plan=QueryPlan(
            primary_route="calculator",
            route_sequence=["calculator", "retrieval"],
            reasoning=["Use deterministic calculation first."],
            should_visualize=True,
        ),
        retrieved=[RetrievedChunk(chunk=chunk, score=0.9)],
        calculation=calculation,
        visualization=None,
    )
    markdown = response.to_markdown()

    assert "Planned Approach" in markdown
    assert "Executive Summary" in markdown
    assert "Calculations Performed" in markdown
    assert "Source References" in markdown
    assert "[annual_report_2024.md](" in markdown
