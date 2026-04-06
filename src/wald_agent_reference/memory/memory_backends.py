from __future__ import annotations

import json
import re
import urllib.error
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

    def delete_container(self) -> None:
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

    def delete_container(self) -> None:
        return None


@dataclass
class SupermemoryBackend(MemoryBackend):
    settings: AppSettings

    def __post_init__(self) -> None:
        self.logger = get_logger("memory.supermemory")

    def _post_document(self, content: str, custom_id: str, metadata: dict[str, object]) -> None:
        if not self.settings.supermemory_api_key:
            return
        if not self.settings.supermemory_container_tag:
            self.logger.debug("Skipping Supermemory sync - container tag not configured")
            return
        normalized_content = content.strip()
        if not normalized_content:
            self.logger.info("Skipping Supermemory sync for %s because content is empty", custom_id)
            return
        
        try:
            payload = json.dumps(
                {
                    "content": normalized_content,
                    "customId": custom_id,
                    "containerTag": self.settings.supermemory_container_tag,
                    "metadata": self._sanitize_metadata(metadata),
                }
            ).encode("utf-8")
        except Exception as exc:
            self.logger.warning("Failed to serialize payload for Supermemory sync of %s: %s", custom_id, exc)
            return None
        
        try:
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
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            self.logger.warning(
                "Supermemory sync failed for %s with HTTP %s. Detail: %s",
                custom_id,
                exc.code,
                detail or exc.reason,
            )
            return None
        except Exception as exc:
            self.logger.warning("Supermemory sync failed for %s: %s", custom_id, exc)
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

    def delete_container(self) -> None:
        if not self.settings.supermemory_api_key:
            return
        payload = json.dumps({"containerTags": [self.settings.supermemory_container_tag]}).encode("utf-8")
        request = urllib.request.Request(
            url="https://api.supermemory.ai/v3/documents/bulk",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.supermemory_api_key}",
            },
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(request, timeout=30):
                self.logger.debug("Deleted Supermemory documents for container=%s", self.settings.supermemory_container_tag)
        except Exception as exc:
            self.logger.warning("Failed to delete Supermemory container %s: %s", self.settings.supermemory_container_tag, exc)

    def _sanitize_custom_id(self, custom_id: str) -> str:
        """Sanitize custom_id to contain only alphanumeric characters, hyphens, underscores, and colons."""
        # Replace spaces and other special characters with underscores
        sanitized = re.sub(r'[^\w\-:]', '_', custom_id)
        return sanitized

    def sync_document(self, document: ExtractedDocument) -> None:
        self._post_document(
            content=document.raw_text,
            custom_id=self._sanitize_custom_id(document.document_id),
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
            custom_id=self._sanitize_custom_id(f"table:{table.table_id}"),
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
            custom_id=self._sanitize_custom_id(f"visual:{visual.artifact_id}"),
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
            custom_id=self._sanitize_custom_id(f"chunk:{chunk.chunk_id}"),
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
            try:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    sanitized[key] = value if value is not None else ""
                else:
                    # Try to convert to JSON string
                    try:
                        sanitized[key] = json.dumps(value, sort_keys=True)
                    except (TypeError, ValueError):
                        # Fall back to string representation if JSON serialization fails
                        sanitized[key] = str(value)
            except Exception as exc:
                self.logger.debug("Failed to sanitize metadata key %s: %s", key, exc)
                sanitized[key] = str(value) if value is not None else ""
        return sanitized


def build_memory_backend(settings: AppSettings) -> MemoryBackend:
    if settings.memory_backend == "supermemory" and settings.supermemory_api_key:
        return SupermemoryBackend(settings)
    return NullMemoryBackend()
