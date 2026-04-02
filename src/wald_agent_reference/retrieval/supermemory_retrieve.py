from __future__ import annotations

from pathlib import Path

from ..core.models import Corpus, DocumentChunk, RetrievedChunk
from ..memory.memory_backends import MemoryBackend


class SupermemoryRetriever:
    def __init__(self, corpus: Corpus, memory_backend: MemoryBackend) -> None:
        self.corpus = corpus
        self.memory_backend = memory_backend
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in corpus.chunks}
        self.table_chunk_lookup = {
            chunk.metadata["table_id"]: chunk
            for chunk in corpus.chunks
            if chunk.metadata.get("table_id")
        }
        self.visual_chunk_lookup = {
            chunk.metadata["visual_id"]: chunk
            for chunk in corpus.chunks
            if chunk.metadata.get("visual_id")
        }
        self.document_lookup = corpus.documents
        self.path_lookup = self._build_path_lookup()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        hits = self.memory_backend.search(query, top_k)
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for hit in hits:
            resolved = self._resolve_hit(hit)
            if resolved is None or resolved.chunk.chunk_id in seen:
                continue
            seen.add(resolved.chunk.chunk_id)
            results.append(resolved)
        return results[:top_k]

    def _resolve_hit(self, hit: dict[str, object]) -> RetrievedChunk | None:
        metadata = self._extract_metadata(hit)
        score = self._extract_score(hit)
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in self.chunk_lookup:
            return RetrievedChunk(chunk=self.chunk_lookup[chunk_id], score=score)

        table_id = str(metadata.get("table_id", "")).strip()
        if table_id and table_id in self.table_chunk_lookup:
            return RetrievedChunk(chunk=self.table_chunk_lookup[table_id], score=score)

        artifact_id = str(metadata.get("artifact_id", "")).strip()
        if artifact_id and artifact_id in self.visual_chunk_lookup:
            return RetrievedChunk(chunk=self.visual_chunk_lookup[artifact_id], score=score)

        document_id = str(metadata.get("document_id", "")).strip()
        if document_id and document_id in self.document_lookup:
            document = self.document_lookup[document_id]
            return RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"supermemory:{document_id}",
                    source_path=document.source_path,
                    source_type=document.source_type,
                    content=self._extract_text(hit) or document.raw_text[:400],
                    metadata={"document_id": document_id},
                ),
                score=score,
            )

        source_file = str(metadata.get("source_file", "")).strip()
        if source_file and source_file in self.path_lookup:
            return RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"supermemory:{source_file}:{abs(hash(self._extract_text(hit)))}",
                    source_path=self.path_lookup[source_file],
                    source_type=str(metadata.get("source_type", "memory")),
                    content=self._extract_text(hit),
                    metadata=metadata,
                ),
                score=score,
            )
        return None

    def _build_path_lookup(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for document in self.corpus.documents.values():
            paths[document.source_path.name] = document.source_path
        for table in self.corpus.tables.values():
            paths[table.source_path.name] = table.source_path
        for visual in self.corpus.visuals.values():
            paths[visual.source_path.name] = visual.source_path
        for chunk in self.corpus.chunks:
            paths[chunk.source_path.name] = chunk.source_path
        return paths

    @staticmethod
    def _extract_metadata(hit: dict[str, object]) -> dict[str, object]:
        metadata = hit.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        if isinstance(hit.get("chunk"), dict):
            nested = hit["chunk"].get("metadata")
            if isinstance(nested, dict):
                return nested
        if isinstance(hit.get("memory"), dict):
            nested = hit["memory"].get("metadata")
            if isinstance(nested, dict):
                return nested
        return {}

    @staticmethod
    def _extract_text(hit: dict[str, object]) -> str:
        for key in ("content", "text", "summary"):
            value = hit.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for nested_key in ("chunk", "memory"):
            nested = hit.get(nested_key)
            if isinstance(nested, dict):
                for key in ("content", "text", "summary"):
                    value = nested.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        return ""

    @staticmethod
    def _extract_score(hit: dict[str, object]) -> float:
        for key in ("score", "similarity", "relevanceScore"):
            value = hit.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 1.0
