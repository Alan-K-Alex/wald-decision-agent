from __future__ import annotations

from ..core.logging import get_logger
from ..core.models import QueryPlan


class PlannerAgent:
    def __init__(self) -> None:
        self.logger = get_logger("reasoning.planner")

    def plan(self, question: str) -> QueryPlan:
        lowered = question.lower()
        should_visualize = any(term in lowered for term in ["plot", "chart", "graph", "visual", "trend", "compare", "comparison"])
        needs_explanation = any(term in lowered for term in ["why", "explain", "driver", "drivers", "because", "contributed", "context"])
        # Detect questions about narrative/qualitative topics (leadership priorities, risks, improvements, challenges, opinions)
        is_narrative_question = any(term in lowered for term in [
            "risk", "risks", "challenge", "challenges", "improvement", "improve", "priority", "priorities",
            "opinion", "perspective", "leadership", "board", "executive", "strategy", "strategic",
            "initiative", "initiatives", "focus", "outlook", "direction", "guidance", "concern", "concerns",
            "pressure", "pressured", "bottleneck", "barrier", "blocker"
        ])
        
        # If user explicitly asks for document/narrative content, prioritize retrieval over SQL
        # even if question contains keywords like "plan" that might suggest financial metrics
        requests_document_data = any(term in lowered for term in [
            "document", "as per document", "from document", "per document", "narrative", 
            "text", "written", "brief", "report", "strategic", "priorities", "according to",
            "in the text", "from the text", "per the report"
        ])
        
        has_structured_need = any(term in lowered for term in ["plan", "target", "variance", "missed", "beat", "revenue", "growth", "margin", "underperform", "lowest", "highest", "count", "sum", "average", "trend", "risk", "risks"])

        # NEW: Find explicit filename mentions (e.g. "In strategy_performance_pack.pdf...")
        # Looks for words with common document extensions
        import re
        filename_matches = re.findall(r'[\w\-_]+\.(?:pdf|docx?|xlsx?|csv|md|txt|svg|png)', lowered)
        filename_filter = filename_matches[0] if filename_matches else None
        
        if filename_filter:
            self.logger.info("Detected filename constraint in query: %s", filename_filter)
        
        if needs_explanation and has_structured_need:
            plan = QueryPlan(
                primary_route="sql" if any(term in lowered for term in ["plan", "target", "variance", "missed", "beat"]) else "calculator",
                route_sequence=["sql", "calculator", "retrieval"],  # Fetch from all sources
                reasoning=[
                    "Query mixes structured analysis with explanatory context.",
                    "Fetch numeric results from tables and calculator.",
                    "Also retrieve narrative evidence to provide context.",
                    "Answer composer will intelligently blend the sources.",
                ],
                should_visualize=should_visualize,
                filename_filter=filename_filter,
            )
            self.logger.debug("Planner selected mixed structured+narrative route for question: %s", question)
            return plan

        # If question is narrative in nature (about risks, priorities, improvements, etc.),
        # STILL fetch from SQL/calculator to have both sources available
        if is_narrative_question and not has_structured_need and (requests_document_data or filename_filter):
            plan = QueryPlan(
                primary_route="retrieval",
                route_sequence=["retrieval", "sql", "calculator"],  # Try retrieval first but fetch all
                reasoning=[
                    "Question asks about qualitative/narrative topics (risks, priorities, improvements, challenges).",
                    "Prioritize document retrieval for narrative information.",
                    "Also attempt structured queries to provide numeric context if relevant.",
                    "Answer composer will select the most appropriate source(s).",
                ],
                should_visualize=should_visualize,
                filename_filter=filename_filter,
            )
            self.logger.debug("Planner selected narrative-first route for question: %s", question)
            return plan

        if any(term in lowered for term in ["plan", "target", "variance", "missed", "beat"]):
            # But if user explicitly requested documents, prioritize retrieval
            if requests_document_data or filename_filter:
                plan = QueryPlan(
                    primary_route="retrieval",
                    route_sequence=["retrieval", "sql", "calculator"],  # Added sql to sequence
                    reasoning=[
                        "User explicitly requested document-based information or provided a document filter.",
                        "Prioritize narrative retrieval.",
                        "Also fetch structured data to support or validate retrieved insights.",
                    ],
                    should_visualize=should_visualize,
                    filename_filter=filename_filter,
                )
                self.logger.debug("Planner selected retrieval route (document-explicit) for question: %s", question)
                return plan
            
            plan = QueryPlan(
                primary_route="sql",
                route_sequence=["sql", "calculator", "retrieval"],  # Always include retrieval
                reasoning=[
                    "Query implies structured table analysis.",
                    "Fetch SQL results and calculations.",
                    "Also retrieve context from documents.",
                    "Answer composer will blend numeric and narrative appropriately.",
                ],
                should_visualize=should_visualize,
                filename_filter=filename_filter,
            )
            self.logger.debug("Planner selected SQL route for question: %s", question)
            return plan

        if any(term in lowered for term in ["chart", "graph", "visual", "figure"]):
            plan = QueryPlan(
                primary_route="visual",
                route_sequence=["visual", "calculator", "sql", "retrieval"],  # Full sequence
                reasoning=[
                    "Query explicitly references visual artifacts.",
                    "Attempt visual extraction first.",
                    "Support with numeric/table tools and narrative context.",
                ],
                should_visualize=True,
                filename_filter=filename_filter,
            )
            self.logger.debug("Planner selected visual route for question: %s", question)
            return plan

        if any(term in lowered for term in ["revenue", "growth", "margin", "underperform", "lowest", "highest", "count", "sum", "average", "trend", "risk", "risks"]):
            # Unless it's asking about narrative aspects of these topics
            # But keep 'risk' queries structured as they often map to a risk register
            is_metric_query = any(term in lowered for term in ["revenue", "growth", "margin", "cost", "sum", "average", "total"])
            if is_narrative_question and not any(term in lowered for term in ["risk", "risks"]) and not is_metric_query:
                plan = QueryPlan(
                    primary_route="retrieval",
                    route_sequence=["retrieval", "calculator", "sql"],  # Full sequence
                    reasoning=[
                        "Question asks about narrative aspects of numeric topics.",
                        "Use retrieval first for business context.",
                        "Support with numeric validation from tables.",
                    ],
                    should_visualize=should_visualize,
                    filename_filter=filename_filter,
                )
                self.logger.debug("Planner selected narrative-first (hybrid numeric) route for question: %s", question)
                return plan
            
            plan = QueryPlan(
                primary_route="calculator",
                route_sequence=["calculator", "sql", "retrieval"],  # Always include retrieval
                reasoning=[
                    "Query is numeric, ranking-oriented, or a qualitative list (risks).",
                    "Use deterministic computation or lookup with table data.",
                    "Also retrieve narrative explanations.",
                    "Answer composer selects the most relevant source(s).",
                ],
                should_visualize=should_visualize,
                filename_filter=filename_filter,
            )
            self.logger.debug("Planner selected calculator route for question: %s", question)
            return plan

        # Default: prioritize retrieval but still try to fetch from other sources
        plan = QueryPlan(
            primary_route="retrieval",
            route_sequence=["retrieval", "sql", "calculator"],  # Always include all sources
            reasoning=[
                "Query is primarily narrative.",
                "Use grounded retrieval for primary answer.",
                "Support with structured data if relevant.",
            ],
            should_visualize=should_visualize,
        )
        self.logger.debug("Planner selected retrieval route for question: %s", question)
        return plan
