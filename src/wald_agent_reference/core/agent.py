from __future__ import annotations

import base64
import json
from pathlib import Path

from ..ingestion.ingest import DocumentIngestor
from ..reasoning.answer import AnswerComposer
from ..reasoning.planner import PlannerAgent
from ..retrieval.retrieve import HybridRetriever
from ..retrieval.supermemory_retrieve import SupermemoryRetriever
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
        plan = self.planner.plan(question)
        self.logger.info("Received question: %s", question)
        self.logger.info("Planner route sequence: %s", " -> ".join(plan.route_sequence))
        corpus, catalog = self._prepare_context(docs_path)
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
        # Execute ALL routes in the planner-selected sequence to gather results from multiple sources.
        # The AnswerComposer will intelligently select which source(s) to use for the final answer.
        # This allows hybrid answers that combine both structured data and narrative context.
        for route in plan.route_sequence:
            if route == "sql":
                self.logger.info("Attempting SQL route")
                sql_result = self.tools.sql_agent.answer(question, catalog)
                if sql_result is not None and calculation is None:
                    calculation = sql_result  # Use first successful result as primary
                    
            elif route == "visual":
                self.logger.info("Attempting visual reasoning route")
                visual_result = self.tools.visual_reasoner.answer(question, visuals)
                if visual_result is not None and calculation is None:
                    calculation = visual_result
                    
            elif route == "calculator":
                self.logger.info("Attempting calculator route")
                calc_result = self.tools.calculator.calculate(question, tables)
                if calc_result is not None and calculation is None:
                    calculation = calc_result
                    
            elif route == "retrieval":
                # Continue to retrieval, but don't break - let all routes execute
                self.logger.info("Attempting retrieval route")
                continue
        
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

        response = self.answerer.compose(question, plan, retrieved, calculation, visualization, docs_path=docs_path)
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
        
        # Sync to memory backend with graceful error handling
        # DISABLED: Supermemory API is rate-limiting (HTTP 429), blocking requests for 25-30s
        # Local retrieval fallback is working correctly, so sync is not critical
        # Uncomment below if needed in the future with async processing
        # try:
        #     for document in corpus.documents.values():
        #         self.tools.memory_backend.sync_document(document)
        #     for chunk in corpus.chunks:
        #         self.tools.memory_backend.sync_chunk(chunk)
        #     for table in corpus.tables.values():
        #         self.tools.memory_backend.sync_table(table)
        #     for visual in corpus.visuals.values():
        #         self.tools.memory_backend.sync_visual(visual)
        # except Exception as exc:
        #     self.logger.warning("Memory backend sync encountered an error: %s. Continuing with query processing.", exc)
        
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
        local_results = retriever.search(question)
        
        # If retrieval-focused and few results, try expanded keywords
        if plan.primary_route == "retrieval" and len(local_results) < 2:
            expanded_queries = self._generate_expanded_queries(question)
            for expanded_q in expanded_queries:
                if len(local_results) >= self.settings.top_k:
                    break
                expanded_results = retriever.search(expanded_q)
                for item in expanded_results:
                    if item.chunk.chunk_id not in {r.chunk.chunk_id for r in local_results}:
                        local_results.append(item)
        
        if self.settings.retrieval_backend not in {"auto", "supermemory"}:
            return local_results

        try:
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
        except Exception as exc:
            self.logger.debug("Supermemory retrieval failed: %s. Using local results.", exc)
            return local_results

    def _generate_expanded_queries(self, question: str) -> list[str]:
        """Generate expanded query variations for narrative questions."""
        expansions = []
        lower_q = question.lower()
        
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
        
        return [q for q in expansions if q != question]

    def _persist_report(self, response: AgentResponse) -> None:
        self.settings.reports_path.mkdir(parents=True, exist_ok=True)
        filename = self.settings.reports_path / f"{slugify(response.question)}.md"
        filename.write_text(response.to_markdown(), encoding="utf-8")
