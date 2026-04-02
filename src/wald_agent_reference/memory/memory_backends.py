from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from ..core.config import AppSettings
from ..core.logging import get_logger
from ..core.models import DocumentChunk, ExtractedDocument, StructuredTable, VisualArtifact


class MemoryBackend:
    def sync_document(self, document: ExtractedDocument) -> None:
        raise NotImplementedError

    def sync_table(self, table: StructuredTable) -> None:
        raise NotImplementedError

    def sync_visual(self, visual: VisualArtifact) -> None:
        raise NotImplementedError

    def sync_chunk(self, chunk: DocumentChunk) -> None:
        raise NotImplementedError

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        raise NotImplementedError


class NullMemoryBackend(MemoryBackend):
    def sync_document(self, document: ExtractedDocument) -> None:
        return None

    def sync_table(self, table: StructuredTable) -> None:
        return None

    def sync_visual(self, visual: VisualArtifact) -> None:
        return None

    def sync_chunk(self, chunk: DocumentChunk) -> None:
        return None

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        return []


@dataclass
class SupermemoryBackend(MemoryBackend):
    settings: AppSettings

    def __post_init__(self) -> None:
        self.logger = get_logger("memory.supermemory")

    def _post_document(self, content: str, custom_id: str, metadata: dict[str, object]) -> None:
        if not self.settings.supermemory_api_key:
            return
        payload = json.dumps(
            {
                "content": content,
                "customId": custom_id,
                "containerTag": self.settings.supermemory_container_tag,
                "metadata": self._sanitize_metadata(metadata),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.supermemory.ai/v3/documents",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.supermemory_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30):
            self.logger.debug("Synced object to Supermemory with custom_id=%s", custom_id)
            return None

    def search(self, query: str, limit: int) -> list[dict[str, object]]:
        if not self.settings.supermemory_api_key:
            return []
        payload_dict = {
            "q": query,
            "limit": limit,
            "searchMode": self.settings.supermemory_search_mode,
            "containerTags": [self.settings.supermemory_container_tag],
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.supermemory.ai/v4/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.supermemory_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except Exception as exc:
            self.logger.warning("Supermemory search failed for query '%s': %s", query, exc)
            return []
        results = body.get("results") or body.get("items") or body.get("data", {}).get("results") or []
        return [result for result in results if isinstance(result, dict)]

    def sync_document(self, document: ExtractedDocument) -> None:
        self._post_document(
            content=document.raw_text,
            custom_id=document.document_id,
            metadata={
                "source_type": document.source_type,
                "source_file": document.source_path.name,
                "document_id": document.document_id,
                **document.metadata,
            },
        )

    def sync_table(self, table: StructuredTable) -> None:
        self._post_document(
            content=table.retrieval_text,
            custom_id=f"table:{table.table_id}",
            metadata={
                "source_type": table.source_type,
                "source_file": table.source_path.name,
                "table_id": table.table_id,
                **table.metadata,
            },
        )

    def sync_visual(self, visual: VisualArtifact) -> None:
        self._post_document(
            content=f"{visual.summary}\n{visual.extracted_text}",
            custom_id=f"visual:{visual.artifact_id}",
            metadata={
                "source_type": visual.source_type,
                "source_file": visual.source_path.name,
                "artifact_id": visual.artifact_id,
                **visual.metadata,
            },
        )

    def sync_chunk(self, chunk: DocumentChunk) -> None:
        self._post_document(
            content=chunk.content,
            custom_id=f"chunk:{chunk.chunk_id}",
            metadata={
                "source_type": chunk.source_type,
                "source_file": chunk.source_path.name,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata,
            },
        )

    def _sanitize_metadata(self, metadata: dict[str, object]) -> dict[str, object]:
        sanitized: dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value if value is not None else ""
            else:
                sanitized[key] = json.dumps(value, sort_keys=True)
        return sanitized


def build_memory_backend(settings: AppSettings) -> MemoryBackend:
    if settings.memory_backend == "supermemory" and settings.supermemory_api_key:
        return SupermemoryBackend(settings)
    return NullMemoryBackend()
