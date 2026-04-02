from __future__ import annotations

import json
from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.ingestion.ingest import DocumentIngestor
from wald_agent_reference.memory.structured_store import StructuredMemoryStore


def test_long_document_is_chunked_with_offsets_and_persisted(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = AppSettings(chunk_size=220, chunk_overlap=40)
    corpus = DocumentIngestor(settings).ingest_folder(root / "data" / "raw")

    document = corpus.documents["leadership_brief_long:document"]
    long_chunks = [chunk for chunk in corpus.chunks if chunk.source_path.name == "leadership_brief_long.txt"]
    assert len(long_chunks) > 1
    assert all("start_offset" in chunk.metadata for chunk in long_chunks)
    assert document.metadata["text_hash"]

    store = StructuredMemoryStore(tmp_path / "memory.db")
    store.persist_documents(list(corpus.documents.values()))
    store.persist_chunks(corpus.chunks)
    columns, rows = store.execute(
        "SELECT source_file, json_extract(metadata_json, '$.text_hash') FROM documents WHERE document_id = 'leadership_brief_long:document'"
    )
    assert columns == ["source_file", "json_extract(metadata_json, '$.text_hash')"]
    assert rows and rows[0][0] == "leadership_brief_long.txt"


def test_visual_artifact_is_persisted_to_sqlite(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = DocumentIngestor(AppSettings()).ingest_folder(root / "data" / "raw")
    store = StructuredMemoryStore(tmp_path / "memory.db")
    store.persist_visual_artifacts(list(corpus.visuals.values()))

    columns, rows = store.execute("SELECT source_file, summary, metadata_json FROM visual_artifacts")
    assert columns == ["source_file", "summary", "metadata_json"]
    assert any(row[0] == "revenue_chart.svg" for row in rows)
    assert any(json.loads(row[2]).get("extraction_backend") == "svg-text" for row in rows)
