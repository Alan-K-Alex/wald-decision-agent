from __future__ import annotations

from pathlib import Path

from ..ingestion.ingest import DocumentIngestor
from ..reasoning.answer import AnswerComposer
from ..reasoning.planner import PlannerAgent
from ..retrieval.retrieve import HybridRetriever
from ..retrieval.supermemory_retrieve import SupermemoryRetriever
from .config import AppSettings, load_settings
from .logging import configure_logging, get_logger
from .models import AgentResponse, StructuredTable
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
        plan = self.planner.plan(question)
        self.logger.info("Received question: %s", question)
        self.logger.info("Planner route sequence: %s", " -> ".join(plan.route_sequence))
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
        for document in corpus.documents.values():
            self.tools.memory_backend.sync_document(document)
        for chunk in corpus.chunks:
            self.tools.memory_backend.sync_chunk(chunk)
        for table in corpus.tables.values():
            self.tools.memory_backend.sync_table(table)
        for visual in corpus.visuals.values():
            self.tools.memory_backend.sync_visual(visual)
        retrieved = self._retrieve_evidence(question, plan, corpus)
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

        calculation = None
        # Execute the planner-selected routes in order and stop once one route resolves the query.
        for route in plan.route_sequence:
            if route == "sql" and calculation is None:
                self.logger.info("Attempting SQL route")
                calculation = self.tools.sql_agent.answer(question, catalog)
            elif route == "visual" and calculation is None:
                self.logger.info("Attempting visual reasoning route")
                calculation = self.tools.visual_reasoner.answer(question, visuals)
            elif route == "calculator" and calculation is None:
                self.logger.info("Attempting calculator route")
                calculation = self.tools.calculator.calculate(question, tables)
            elif route == "retrieval":
                break
        if calculation is None and corpus.tables:
            # Retrieval can miss the best table, so use the full structured catalog as a grounded fallback.
            self.logger.info("Primary planned routes did not resolve query, falling back to all structured tables")
            calculation = self.tools.calculator.calculate(question, list(corpus.tables.values()))
        visualization = None
        if generate_plot or plan.should_visualize or self.tools.visualizer.should_visualize(question, calculation):
            if calculation is not None:
                visualization = self.tools.visualizer.create(question, calculation)
                if visualization is not None:
                    self.logger.info("Generated plot at %s", visualization.path)

        response = self.answerer.compose(question, plan, retrieved, calculation, visualization)
        self._persist_report(response)
        self.logger.info("Response persisted for question: %s", question)
        return response

    def _retrieve_evidence(self, question: str, plan, corpus) -> list:
        local_results = HybridRetriever(corpus, self.settings).search(question)
        if self.settings.retrieval_backend not in {"auto", "supermemory"}:
            return local_results

        memory_results = SupermemoryRetriever(corpus, self.tools.memory_backend).search(question, self.settings.top_k)
        if not memory_results:
            self.logger.info("Supermemory retrieval unavailable or empty, falling back to local hybrid retrieval")
            return local_results

        merged = []
        seen: set[str] = set()
        for item in memory_results + local_results:
            if item.chunk.chunk_id in seen:
                continue
            seen.add(item.chunk.chunk_id)
            merged.append(item)
            if len(merged) >= self.settings.top_k:
                break
        self.logger.info("Using Supermemory-assisted retrieval for %s route", plan.primary_route)
        return merged

    def _persist_report(self, response: AgentResponse) -> None:
        self.settings.reports_path.mkdir(parents=True, exist_ok=True)
        filename = self.settings.reports_path / f"{slugify(response.question)}.md"
        filename.write_text(response.to_markdown(), encoding="utf-8")
