"""Integration test showing the bug fix in action."""

from pathlib import Path
from wald_agent_reference.reasoning.planner import PlannerAgent
from wald_agent_reference.core.agent import LeadershipInsightAgent
from wald_agent_reference.core.config import load_settings


def test_strategy_plans_document_query_routes_to_retrieval():
    """
    Integration test: Verify that "strategy plans as per documents?"
    now routes to retrieval first instead of SQL.
    """
    planner = PlannerAgent()
    
    # Original problem query
    plan = planner.plan("strategy plans as per documents?")
    
    # Verify routing decision
    print(f"✓ Question: 'strategy plans as per documents?'")
    print(f"✓ Primary route: {plan.primary_route}")
    print(f"✓ Route sequence: {plan.route_sequence}")
    print(f"✓ Should visualize: {plan.should_visualize}")
    print(f"✓ Reasoning: {plan.reasoning}")
    
    # Assert correct routing
    assert plan.primary_route == "retrieval", (
        f"Expected 'retrieval' but got '{plan.primary_route}'. "
        "Bug fix may not be applied correctly."
    )
    assert plan.route_sequence[0] == "retrieval", (
        f"Expected first route to be 'retrieval' but got '{plan.route_sequence[0]}'"
    )
    
    print("\n✅ BUG FIX VERIFIED: Document-explicit requests now route to retrieval first!")


def test_financial_questions_still_use_sql():
    """Ensure we didn't break normal SQL routing for financial questions."""
    planner = PlannerAgent()
    
    # Pure financial question (no document mention)
    plan = planner.plan("Which region has the largest revenue variance?")
    
    print(f"\n✓ Question: 'Which region has the largest revenue variance?'")
    print(f"✓ Primary route: {plan.primary_route}")
    print(f"✓ Route sequence: {plan.route_sequence}")
    
    # Should still use SQL for pure financial questions
    assert plan.primary_route == "sql", (
        f"Financial questions should route to SQL but got '{plan.primary_route}'"
    )
    
    print("✅ Financial questions still correctly route to SQL")


if __name__ == "__main__":
    test_strategy_plans_document_query_routes_to_retrieval()
    test_financial_questions_still_use_sql()
    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS PASSED ✅")
    print("="*60)
