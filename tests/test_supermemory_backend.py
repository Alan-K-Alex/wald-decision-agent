from __future__ import annotations

import io
import json
from pathlib import Path

from wald_agent_reference.core.config import AppSettings
from wald_agent_reference.core.models import ExtractedDocument
from wald_agent_reference.memory.memory_backends import SupermemoryBackend


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

    settings = AppSettings(memory_backend="supermemory", supermemory_container_tag="adobe-tests")
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
    assert captured["body"]["containerTag"] == "adobe-tests"
    assert captured["body"]["customId"] == "doc-1"
