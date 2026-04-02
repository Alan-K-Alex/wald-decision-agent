from __future__ import annotations

from ..core.logging import get_logger
from ..core.models import QueryPlan


class PlannerAgent:
    def __init__(self) -> None:
        self.logger = get_logger("reasoning.planner")

    def plan(self, question: str) -> QueryPlan:
        lowered = question.lower()
        should_visualize = any(term in lowered for term in ["plot", "chart", "graph", "visual", "trend", "compare", "comparison"])

        if any(term in lowered for term in ["plan", "target", "variance", "missed", "beat"]):
            plan = QueryPlan(
                primary_route="sql",
                route_sequence=["sql", "calculator", "retrieval"],
                reasoning=[
                    "Query implies comparison across structured tables.",
                    "Attempt SQL join/aggregation first.",
                    "Fall back to calculator only if SQL route cannot resolve the question.",
                ],
                should_visualize=should_visualize,
            )
            self.logger.debug("Planner selected SQL route for question: %s", question)
            return plan

        if any(term in lowered for term in ["chart", "graph", "visual", "figure"]):
            plan = QueryPlan(
                primary_route="visual",
                route_sequence=["visual", "calculator", "retrieval"],
                reasoning=[
                    "Query explicitly references a visual artifact.",
                    "Attempt visual extraction and visual reasoning first.",
                    "Use numeric/table tools only if the visual route is insufficient.",
                ],
                should_visualize=True,
            )
            self.logger.debug("Planner selected visual route for question: %s", question)
            return plan

        if any(term in lowered for term in ["revenue", "growth", "margin", "underperform", "lowest", "highest", "count", "sum", "average", "trend"]):
            plan = QueryPlan(
                primary_route="calculator",
                route_sequence=["calculator", "sql", "retrieval"],
                reasoning=[
                    "Query is numeric or ranking-oriented.",
                    "Use deterministic computation before narrative summarization.",
                    "Only answer with values supported by tables or extracted evidence.",
                ],
                should_visualize=should_visualize,
            )
            self.logger.debug("Planner selected calculator route for question: %s", question)
            return plan

        plan = QueryPlan(
            primary_route="retrieval",
            route_sequence=["retrieval"],
            reasoning=[
                "Query is primarily narrative.",
                "Use grounded retrieval and sentence-level synthesis only.",
                "Abstain if evidence is weak or incomplete.",
            ],
            should_visualize=should_visualize,
        )
        self.logger.debug("Planner selected retrieval route for question: %s", question)
        return plan
