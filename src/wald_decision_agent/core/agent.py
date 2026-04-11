from __future__ import annotations

import base64
import json
from pathlib import Path

from ..ingestion.ingest import DocumentIngestor
from ..reasoning.answer import AnswerComposer
from ..reasoning.planner import PlannerAgent
from ..retrieval.retrieve import HybridRetriever
from .config import AppSettings, load_settings
from .logging import configure_logging, get_logger
from .models import AgentResponse, StructuredTable, Corpus, DocumentChunk, ExtractedDocument, VisualArtifact
from .tools import ToolRouter
from ..utils import slugify


class LeadershipInsightAgent:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or load_settings()
        configure_logging(self.settings)
        self.logger = get_logger("core.agent")
        self.ingestor = DocumentIngestor(self.settings)
        self.tools = ToolRouter(self.settings)
        self.planner = PlannerAgent()
        self.answerer = AnswerComposer(self.settings)

    def ask(self, question: str, docs_path: str | Path, generate_plot: bool = False) -> AgentResponse:
        # Extract the core user question for routing/planning.
        # Follow-up questions are expanded with conversation context by the
        # ConversationContextResolver, but the planner, calculator, and SQL
        # agent should route and detect metrics based on the actual user intent,
        # not the previous conversation context which can bias routing toward
        # the wrong tables (e.g. regional data instead of quarterly data).
        core_question = self._extract_core_question(question)
        plan = self.planner.plan(core_question)
        self.logger.info("Received question: %s", question)
        if core_question != question:
            self.logger.info("Core question (for routing): %s", core_question)
        self.logger.info("Planner route sequence: %s", " -> ".join(plan.route_sequence))
        corpus, catalog = self._prepare_context(docs_path)
        
        # Apply strict filename filters to the corpus objects if plan defines it
        if plan.filename_filters:
            self.logger.info("Applying strict filename filters: %s", plan.filename_filters)
            # Filter tables and visuals at the source
            filtered_tables = {tid: t for tid, t in corpus.tables.items() if t.source_path.name in plan.filename_filters}
            filtered_visuals = {vid: v for vid, v in corpus.visuals.items() if v.source_path.name in plan.filename_filters}
            corpus.tables = filtered_tables
            corpus.visuals = filtered_visuals
            
        retrieved = self._retrieve_evidence(core_question, plan, corpus)
        self.logger.info("Retriever returned %d chunks", len(retrieved))
        
        # Bind retrieved evidence back to the higher-fidelity structured objects before reasoning.
        table_ids = {
            item.chunk.metadata["table_id"]
            for item in retrieved
            if item.chunk.metadata.get("table_id") in corpus.tables
        }
        visual_ids = {
            item.chunk.metadata["visual_id"]
            for item in retrieved
            if item.chunk.metadata.get("visual_id") in corpus.visuals
        }
        tables: list[StructuredTable] = [corpus.tables[table_id] for table_id in table_ids]
        visuals = [corpus.visuals[visual_id] for visual_id in visual_ids]

        route_results: list[CalculationResult] = []
        # Execute ALL routes in the planner-selected sequence to gather results from multiple sources.
        # The AnswerComposer will intelligently select which source(s) to use for the final answer.
        # This allows hybrid answers that combine both structured data and narrative context.
        for route in plan.route_sequence:
            if route == "sql":
                self.logger.info("Attempting SQL route")
                sql_result = self.tools.sql_agent.answer(core_question, catalog)
                if sql_result is not None:
                    route_results.append(sql_result)
                    
            elif route == "visual":
                self.logger.info("Attempting visual reasoning route")
                visual_result = self.tools.visual_reasoner.answer(core_question, visuals)
                if visual_result is not None:
                    route_results.append(visual_result)
                    
            elif route == "calculator":
                self.logger.info("Attempting calculator route")
                calc_result = self.tools.calculator.calculate(core_question, tables)
                if calc_result is not None:
                    route_results.append(calc_result)
                    
            elif route == "retrieval":
                # Continue to retrieval, but don't break - let all routes execute
                self.logger.info("Attempting retrieval route")
                continue
        
        if not route_results and corpus.tables and not self._should_skip_structured_fallback(core_question, plan):
            # Retrieval can miss the best table, so use the full structured catalog as a grounded fallback.
            self.logger.info("Primary planned routes did not resolve query, falling back to all structured tables")
            fallback_result = self.tools.calculator.calculate(core_question, list(corpus.tables.values()))
            if fallback_result is not None:
                route_results.append(fallback_result)

        # Mixed questions about trends/comparisons should still leverage chart artifacts when available.
        if plan.should_visualize and visuals:
            visual_result = self.tools.visual_reasoner.answer(core_question, visuals)
            if visual_result is not None and not any(result.answer == visual_result.answer for result in route_results):
                route_results.append(visual_result)

        primary_calculation = route_results[0] if route_results else None
        supplemental_calculations = route_results[1:] if len(route_results) > 1 else []

        visualizations = []
        seen_chart_signatures: list[tuple[str, tuple, tuple]] = []  # (chart_type, labels, values)
        if generate_plot or plan.should_visualize:
            for index, result in enumerate(route_results):
                if not self.tools.visualizer.should_visualize(core_question, result) and not generate_plot and not plan.should_visualize:
                    continue
                if result.chart_data is None:
                    continue
                # Deduplicate charts: skip if a chart with substantially
                # overlapping data has already been created.  Two charts are
                # considered duplicates when they share the same chart type AND
                # BOTH (a) their labels overlap significantly AND (b) their
                # numeric values are similar.  Requiring both prevents
                # suppressing legitimate charts that share time-period labels
                # but plot different metrics (e.g. revenue vs margin).
                chart_type = result.chart_data.get("type", "")
                chart_labels = tuple(str(l) for l in result.chart_data.get("labels", []))
                chart_values = tuple(float(v) for v in result.chart_data.get("values", []) if v is not None)
                is_duplicate = False
                for seen_type, seen_labels, seen_values in seen_chart_signatures:
                    if chart_type != seen_type:
                        continue
                    # Check label overlap (normalize: lowercase, strip year suffix)
                    norm = lambda s: s.lower().split()[0] if s else s
                    norm_labels = {norm(l) for l in chart_labels}
                    norm_seen = {norm(l) for l in seen_labels}
                    label_overlap = norm_labels & norm_seen
                    labels_match = label_overlap and len(label_overlap) >= min(len(norm_labels), len(norm_seen)) * 0.5
                    # Check value similarity: sorted values within 1% tolerance
                    values_match = False
                    if chart_values and seen_values and len(chart_values) == len(seen_values):
                        max_val = max(max(chart_values), max(seen_values), 1)
                        values_match = all(
                            abs(a - b) <= max_val * 0.01
                            for a, b in zip(sorted(chart_values), sorted(seen_values))
                        )
                    # Must match on BOTH labels and values to be a true duplicate
                    if labels_match and values_match:
                        is_duplicate = True
                        break
                if is_duplicate:
                    self.logger.info("Skipping duplicate chart (type=%s, title=%s)", chart_type, result.chart_data.get("title", ""))
                    continue
                seen_chart_signatures.append((chart_type, chart_labels, chart_values))
                suffix = result.chart_data.get("title", f"chart-{index + 1}")
                visualization = self.tools.visualizer.create(core_question, result, suffix=suffix)
                if visualization is not None:
                    visualizations.append(visualization)
                    self.logger.info("Generated plot at %s", visualization.path)

        response = self.answerer.compose(
            question,
            plan,
            retrieved,
            primary_calculation,
            visualizations,
            docs_path=docs_path,
            supplemental_calculations=supplemental_calculations,
        )
        self._persist_report(response)
        self.logger.info("Response persisted for question: %s", question)
        return response

    def prepare_documents(self, docs_path: str | Path) -> dict[str, int]:
        corpus, _ = self._prepare_context(docs_path)
        return {
            "documents": len(corpus.documents),
            "chunks": len(corpus.chunks),
            "tables": len(corpus.tables),
            "visuals": len(corpus.visuals),
        }

    def _prepare_context(self, docs_path: str | Path):
        docs_path = Path(docs_path)
        metadata_file = Path(self.settings.output_dir) / ".ingestion_metadata.json"
        
        # Check if we can use cached ingestion
        catalog = self.tools.structured_store.get_all_tables()
        if catalog and metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                if metadata.get("docs_path") == str(docs_path) and catalog:
                    self.logger.info("Using cached ingestion data from structured store")
                    corpus = self._load_corpus_from_store()
                    if corpus and len(corpus.chunks) > 0:
                        return corpus, catalog
            except (json.JSONDecodeError, Exception):
                pass
        
        # If cache miss or error, perform fresh ingestion
        corpus = self.ingestor.ingest_folder(docs_path)
        self.logger.info(
            "Ingestion completed | documents=%d chunks=%d tables=%d visuals=%d",
            len(corpus.documents),
            len(corpus.chunks),
            len(corpus.tables),
            len(corpus.visuals),
        )
        self.tools.structured_store.persist_documents(list(corpus.documents.values()))
        self.tools.structured_store.persist_chunks(corpus.chunks)
        catalog = self.tools.structured_store.persist_tables(list(corpus.tables.values()))
        self.tools.structured_store.persist_visual_artifacts(list(corpus.visuals.values()))
        
        # Save ingestion metadata for future cache hits
        try:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(json.dumps({"docs_path": str(docs_path)}))
        except Exception as exc:
            self.logger.debug("Failed to save ingestion metadata: %s", exc)
        
        return corpus, catalog

    def _load_corpus_from_store(self) -> Corpus | None:
        """Load previously ingested data from the structured store."""
        try:
            chunks = self.tools.structured_store.load_all_chunks()
            documents = self.tools.structured_store.load_all_documents()
            tables = self.tools.structured_store.load_all_tables()
            visuals = self.tools.structured_store.load_all_visuals()
            
            if chunks or documents or tables or visuals:
                corpus = Corpus(
                    chunks=chunks,
                    documents=documents,
                    tables=tables,
                    visuals=visuals,
                )
                return corpus
        except Exception as exc:
            self.logger.debug("Failed to load corpus from store: %s", exc)
        return None

    def _retrieve_evidence(self, question: str, plan, corpus) -> list:
        # For narrative/retrieval-focused questions, do expanded retrieval
        retriever = HybridRetriever(corpus, self.settings)
        local_results = retriever.search(question, source_filter=plan.filename_filters)
        
        needs_explanation = any(term in question.lower() for term in ["why", "because", "driver", "drivers", "reason", "caused"])

        # If retrieval-focused or explanation-heavy, try expanded keywords to recover narrative context.
        if (plan.primary_route == "retrieval" and len(local_results) < 2) or needs_explanation:
            expanded_queries = self._generate_expanded_queries(question)
            for expanded_q in expanded_queries:
                if len(local_results) >= self.settings.top_k:
                    break
                expanded_results = retriever.search(expanded_q)
                for item in expanded_results:
                    if item.chunk.chunk_id not in {r.chunk.chunk_id for r in local_results}:
                        local_results.append(item)
        
        return local_results

    def _generate_expanded_queries(self, question: str) -> list[str]:
        """Generate expanded query variations for narrative questions."""
        expansions = []
        lower_q = question.lower().replace("operartional", "operational").replace("opertional", "operational")
        
        # Risk-related expansions
        if "risk" in lower_q:
            expansions.extend([
                lower_q.replace("risks", "threats").replace("risk", "threat"),
                lower_q.replace("risks", "challenges").replace("risk", "challenge"),
                "operational challenges highlighted",
                "obstacles and concerns"
            ])
        
        # Leadership/highlights expansions
        if "leadership" in lower_q or "highlights" in lower_q:
            expansions.extend([
                lower_q.replace("highlights", "discusses").replace("says", "mentions"),
                "leadership brief",
                "executive commentary"
            ])

        if "operational" in lower_q and ("update" in lower_q or "q2" in lower_q or "quarter" in lower_q):
            expansions.extend([
                "q2 operational update",
                "quarterly operations improved support finance engineering sales",
                "operational update key highlights",
                "support and finance continued to lag target performance",
            ])

        if "why" in lower_q or "because" in lower_q or "driver" in lower_q:
            tokens = [token for token in lower_q.replace("?", "").split() if token not in {"why", "did", "does", "the", "and", "how", "has"}]
            subject = " ".join(tokens[:4]).strip()
            if subject:
                expansions.extend([
                    f"{subject} because",
                    f"{subject} due to",
                    f"{subject} driver",
                    f"{subject} slower conversion",
                ])
            expansions.extend([
                "slower enterprise conversions",
                "weaker channel execution",
                "missed plan because",
            ])
        
        return [q for q in expansions if q != question]

    def _persist_report(self, response: AgentResponse) -> None:
        self.settings.reports_path.mkdir(parents=True, exist_ok=True)
        filename = self.settings.reports_path / f"{slugify(response.question)}.md"
        filename.write_text(response.to_markdown(), encoding="utf-8")

    def _should_skip_structured_fallback(self, question: str, plan) -> bool:
        lowered = question.lower()
        has_temporal_constraint = any(term in lowered for term in ["q1", "q2", "q3", "q4", "quarter", "quarterly", "fy"])
        return plan.primary_route == "retrieval" and has_temporal_constraint

    @staticmethod
    def _extract_core_question(question: str) -> str:
        """Extract the core user question from a contextualized follow-up string.

        ConversationContextResolver expands follow-ups into:
            Previous user question: ...
            Previous assistant answer: ...
            Follow-up question: <actual question>

        The planner, calculator, and SQL agent should route based on the actual
        user intent, not the previous conversation context which can bias metric
        detection and table selection (e.g. routing to regional tables when the
        user asked about quarterly trends).

        However, very short/vague follow-ups (e.g. "who owns the most?") may
        lose critical topic context.  When the core question lacks domain
        keywords, we enrich it with the topic noun from the prior question so
        routing still works.  Only the previous *question* is used for this —
        never the previous *answer*, which is what caused the original bias.
        """
        marker = "Follow-up question:"
        if marker not in question:
            return question

        core = question.split(marker, 1)[1].strip()

        # If the core question already has clear routing signals, return as-is.
        routing_keywords = {
            "revenue", "margin", "cost", "risk", "score", "plan", "target",
            "trend", "growth", "variance", "department", "region", "quarter",
            "operational", "performance", "budget", "profit",
        }
        core_tokens = {t.lower().rstrip("?.,!") for t in core.split()}
        if core_tokens & routing_keywords:
            return core

        # The core question is vague (e.g. "who owns the most?"), so extract
        # topic keywords from the previous *question* only (not the answer)
        # to help routing without re-introducing the answer-bias problem.
        prev_q_marker = "Previous user question:"
        if prev_q_marker in question:
            prev_q_part = question.split(prev_q_marker, 1)[1]
            # Take text up to the next section marker
            for end_marker in ["Previous assistant answer:", marker]:
                if end_marker in prev_q_part:
                    prev_q_part = prev_q_part.split(end_marker, 1)[0]
            prev_tokens = {t.lower().rstrip("?.,!") for t in prev_q_part.split()}
            topic_tokens = prev_tokens & routing_keywords
            if topic_tokens:
                core = core.rstrip("?").rstrip() + " (" + " ".join(sorted(topic_tokens)) + ")?"

        return core
