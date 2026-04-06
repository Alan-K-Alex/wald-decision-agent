from __future__ import annotations

from wald_agent_reference.reasoning.planner import PlannerAgent


def test_planner_routes_variance_queries_to_sql() -> None:
    plan = PlannerAgent().plan("Which region missed revenue plan by the largest amount?")
    assert plan.primary_route == "sql"
    assert plan.route_sequence[0] == "sql"


def test_planner_routes_visual_queries_to_visual_reasoner() -> None:
    plan = PlannerAgent().plan("What does the quarterly revenue chart show?")
    assert plan.primary_route == "visual"
    assert plan.should_visualize is True


def test_planner_routes_mixed_explanatory_queries_to_structured_then_retrieval() -> None:
    plan = PlannerAgent().plan("Why did Europe miss revenue plan by the largest amount?")
    assert plan.primary_route == "sql"
    assert plan.route_sequence[:3] == ["sql", "calculator", "retrieval"]
    assert "Compute the grounded numeric result first from tables." in plan.reasoning
