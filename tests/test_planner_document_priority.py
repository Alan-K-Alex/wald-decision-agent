"""Test that planner correctly prioritizes document retrieval when explicitly requested."""

from wald_agent_reference.reasoning.planner import PlannerAgent


def test_planner_respects_explicit_document_request_over_sql_keywords() -> None:
    """
    When user asks "strategy plans as per documents?", should route to retrieval
    even though "plan" would normally trigger SQL-first routing.
    """
    planner = PlannerAgent()
    plan = planner.plan("strategy plans as per documents?")
    
    # Should prioritize retrieval, not SQL
    assert plan.primary_route == "retrieval", f"Expected retrieval but got {plan.primary_route}"
    assert plan.route_sequence[0] == "retrieval", f"Expected first route to be retrieval, got {plan.route_sequence}"
    assert "document" in " ".join(plan.reasoning).lower(), "Reasoning should mention document-explicit preference"


def test_planner_routes_document_request_to_retrieval() -> None:
    """Test various document-explicit phrasings."""
    planner = PlannerAgent()
    document_requests = [
        "strategic priorities from documents?",
        "what are the key risks per document?",
        "summarize the strategic plan from the narrative",
        "what priorities are written in the brief?",
    ]
    
    for question in document_requests:
        plan = planner.plan(question)
        assert plan.primary_route == "retrieval", f"Question '{question}' should route to retrieval but got {plan.primary_route}"


def test_planner_still_routes_finance_to_sql() -> None:
    """Test that pure financial questions still route to SQL."""
    planner = PlannerAgent()
    plan = planner.plan("What was the revenue variance by region?")
    
    # Should still route to SQL when not explicitly requesting documents
    assert plan.primary_route == "sql", f"Expected sql but got {plan.primary_route}"
    assert plan.route_sequence[0] == "sql"


def test_planner_routes_mixed_query_to_sql_first() -> None:
    """Test that queries asking for explanation still route SQL-first."""
    planner = PlannerAgent()
    plan = planner.plan("Why did Europe miss the revenue plan?")
    
    # Mixed explanatory + structured should route to SQL first
    assert plan.primary_route == "sql", f"Mixed query should route to SQL but got {plan.primary_route}"
    assert plan.route_sequence[0] == "sql"
