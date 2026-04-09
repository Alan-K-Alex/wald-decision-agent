from __future__ import annotations

from wald_decision_agent.core.config import AppSettings
from wald_decision_agent.core.models import AgentResponse
from wald_decision_agent.reasoning.answer import AnswerComposer


def test_llm_rewrite_prompt_forbids_unit_assumptions() -> None:
    composer = AnswerComposer(AppSettings(enable_llm_formatting=False))
    response = AgentResponse(
        question="Why did Europe miss the plan and how has the quarterly revenue trend?",
        planned_approach=["Use structured evidence and retrieval."],
        executive_summary="Europe missed plan and the quarterly trend increased.",
        key_findings=["Europe: actual 80 vs target 87."],
        calculations=["Variance formula executed."],
        evidence=["[strategy_performance_pack.pdf](strategy_performance_pack.pdf): Europe missed plan."],
        caveats=[],
        source_references=["[strategy_performance_pack.pdf](strategy_performance_pack.pdf)"],
        visual_insights=[],
    )

    prompt = composer._build_json_rewrite_prompt(response)

    assert "NEVER add or normalize units" in prompt
    assert "If a value appears without an explicit unit, keep it unitless" in prompt
    assert "80M" not in prompt
    assert "7 million" not in prompt
