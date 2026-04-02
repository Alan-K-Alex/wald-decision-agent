from __future__ import annotations

from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.ingestion.ingest import DocumentIngestor
from wald_agent_reference.retrieval.retrieve import HybridRetriever


def test_retrieval_finds_revenue_trend_table() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_path = root / "data" / "raw"
    settings = AppSettings(vector_backend="hash")
    corpus = DocumentIngestor(settings).ingest_folder(docs_path)
    retriever = HybridRetriever(corpus, settings)

    results = retriever.search("What is our current revenue trend?")

    assert results
    top_sources = [item.chunk.source_path.name for item in results[:3]]
    assert "revenue_trend.csv" in top_sources


def test_retrieval_persists_vector_index(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs_path = root / "data" / "raw"
    settings = AppSettings(vector_backend="hash", vector_store_dir=str(tmp_path / "vector_store"))
    corpus = DocumentIngestor(settings).ingest_folder(docs_path)

    retriever = HybridRetriever(corpus, settings)
    results = retriever.search("Which departments are underperforming?")

    assert results
    assert (tmp_path / "vector_store" / "index_metadata.json").exists()
    assert (tmp_path / "vector_store" / "chunk_vectors.npy").exists()
