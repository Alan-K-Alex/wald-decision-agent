from __future__ import annotations

from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.core.models import AgentResponse, CalculationResult, DocumentChunk, QueryPlan, RetrievedChunk
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
        visualizations=[],
    )
    markdown = response.to_markdown()

    assert "Planned Approach" in markdown
    assert "Executive Summary" in markdown
    assert "Calculations Performed" in markdown
    assert "Source References" in markdown
    assert "revenue_trend.csv [Sheet1]" in markdown


def test_formatted_response_preserves_grounded_evidence_and_links() -> None:
    composer = AnswerComposer(AppSettings(enable_llm_formatting=False))
    original = AgentResponse(
        question="Why did Europe miss the plan?",
        planned_approach=["Use structured data and retrieval."],
        executive_summary="Europe missed plan.",
        key_findings=["Europe variance was -7.00."],
        calculations=["Variance formula executed."],
        evidence=["[leadership_brief_long.txt](leadership_brief_long.txt): Europe missed plan because of slower enterprise conversions."],
        caveats=[],
        source_references=["[leadership_brief_long.txt](leadership_brief_long.txt)"],
        visual_insights=[],
    )

    formatted = composer._finalize_formatted_response(
        original,
        {
            "executive_summary": "Europe underperformed against plan.",
            "key_findings": ["Europe revenue trailed target."],
            "evidence": ["plain text evidence that should be ignored"],
            "source_references": ["plain text source that should be ignored"],
        },
    )

    assert formatted.executive_summary == "Europe underperformed against plan."
    assert formatted.evidence == original.evidence
    assert formatted.source_references == original.source_references


def test_formatted_response_preserves_raw_explanatory_summary_when_rewrite_drops_cause() -> None:
    composer = AnswerComposer(AppSettings(enable_llm_formatting=False))
    original = AgentResponse(
        question="Why did Europe miss the plan and how has the quarterly revenue trend?",
        planned_approach=["Use structured data and retrieval."],
        executive_summary=(
            "Europe has the largest negative variance at -7.00 (actual 80.00 vs target 87.00), "
            "and the supporting narrative attributes the miss to slower mid-market conversion and weaker channel execution. "
            "Quarterly Revenue Trend increased from 120.00 in Q1 to 160.00 in Q4."
        ),
        key_findings=[
            "Europe's variance was -7.00.",
            "Supporting narrative attributes the miss to slower mid-market conversion and weaker channel execution.",
        ],
        calculations=["Variance formula executed."],
        evidence=["[strategy_performance_pack.pdf p.1](strategy_performance_pack.pdf): Europe missed plan because of slower mid-market conversion and weaker channel execution."],
        caveats=[],
        source_references=["[strategy_performance_pack.pdf p.1](strategy_performance_pack.pdf)"],
        visual_insights=["Quarterly Revenue Trend increased from Q1 to Q4."],
    )

    formatted = composer._finalize_formatted_response(
        original,
        {
            "executive_summary": "Europe missed its revenue plan with a variance of -7.00.",
            "key_findings": ["Europe variance was -7.00.", "Quarterly revenue values were 120.00, 138.00, 140.00, and 160.00."],
        },
    )

    assert formatted.executive_summary == original.executive_summary
    assert any("slower mid-market conversion" in finding.lower() for finding in formatted.key_findings)
