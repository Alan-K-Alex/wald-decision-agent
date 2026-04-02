from __future__ import annotations

import io
import json
from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.core.models import Corpus, DocumentChunk, ExtractedDocument
from wald_agent_reference.memory.memory_backends import SupermemoryBackend
from wald_agent_reference.retrieval.supermemory_retrieve import SupermemoryRetriever


class DummyResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_supermemory_backend_posts_expected_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    settings = AppSettings(memory_backend="supermemory", supermemory_container_tag="wald-tests")
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    backend = SupermemoryBackend(settings)
    document = ExtractedDocument(
        document_id="doc-1",
        source_path=Path("brief.txt"),
        source_type="text",
        raw_text="Leadership brief content",
        metadata={"text_hash": "abc123"},
    )

    backend.sync_document(document)

    assert captured["url"] == "https://api.supermemory.ai/v3/documents"
    assert captured["body"]["content"] == "Leadership brief content"
    assert captured["body"]["containerTag"] == "wald-tests"
    assert captured["body"]["customId"] == "doc-1"


def test_supermemory_backend_search_posts_expected_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse(json.dumps({"results": [{"content": "Leadership trend", "score": 0.91, "metadata": {"chunk_id": "doc:1"}}]}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    settings = AppSettings(memory_backend="supermemory", supermemory_container_tag="wald-tests")
    backend = SupermemoryBackend(settings)

    results = backend.search("What is the trend?", limit=3)

    assert captured["url"] == "https://api.supermemory.ai/v4/search"
    assert captured["body"]["q"] == "What is the trend?"
    assert captured["body"]["containerTags"] == ["wald-tests"]
    assert results[0]["metadata"]["chunk_id"] == "doc:1"


def test_supermemory_retriever_resolves_chunk_hits() -> None:
    class StubBackend:
        def search(self, query: str, limit: int) -> list[dict[str, object]]:
            return [
                {
                    "content": "Revenue expanded steadily over the year.",
                    "score": 0.9,
                    "metadata": {"chunk_id": "annual:1"},
                }
            ]

    chunk = DocumentChunk(
        chunk_id="annual:1",
        source_path=Path("annual_report_2024.md"),
        source_type="text",
        content="Revenue expanded steadily over the year.",
        metadata={},
    )
    corpus = Corpus(chunks=[chunk])

    results = SupermemoryRetriever(corpus, StubBackend()).search("What is the revenue trend?", top_k=3)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "annual:1"
