from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.config import AppSettings
from ..core.logging import get_logger
from ..core.models import Corpus, DocumentChunk
from ..utils import tokenize


class BaseEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @property
    def backend_name(self) -> str:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def backend_name(self) -> str:
        return "hash"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dim
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vectors[row_idx, bucket] += sign
            norm = np.linalg.norm(vectors[row_idx])
            if norm > 0:
                vectors[row_idx] /= norm
        return vectors


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: AppSettings) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    @property
    def backend_name(self) -> str:
        return "openai"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [record.embedding for record in response.data]
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, settings: AppSettings) -> None:
        from google import genai

        self.client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else genai.Client()
        self.model = settings.embedding_model

    @property
    def backend_name(self) -> str:
        return "gemini"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            response = self.client.models.embed_content(model=self.model, contents=text)
            raw_embedding = getattr(response, "embeddings", None) or getattr(response, "embedding", None) or response
            if isinstance(raw_embedding, list):
                first = raw_embedding[0]
                values = getattr(first, "values", None) or getattr(first, "embedding", None) or first
            else:
                values = getattr(raw_embedding, "values", None) or getattr(raw_embedding, "embedding", None) or raw_embedding
            vectors.append([float(value) for value in values])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@dataclass
class VectorSearchResult:
    chunk_id: str
    score: float


class VectorIndex:
    def __init__(self, settings: AppSettings, embedder: BaseEmbedder, chunks: dict[str, DocumentChunk]) -> None:
        import chromadb
        self.settings = settings
        self.embedder = embedder
        self.chunks = chunks
        self.logger = get_logger("retrieval.vector_index")
        
        # Initialize Persistent Chroma Client
        store_path = str(settings.vector_store_path.absolute())
        self.client = chromadb.PersistentClient(path=store_path)
        
        # Collection name is scoped to the embedder type to avoid conflicts
        collection_name = f"chunks_{self.embedder.backend_name}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def build(cls, corpus: Corpus, settings: AppSettings) -> "VectorIndex":
        """Build or load vector index using ChromaDB."""
        embedder = cls._select_embedder(settings)
        chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
        index = cls(settings=settings, embedder=embedder, chunks=chunks)
        
        # Check if we need to ingest data
        if index.collection.count() != len(corpus.chunks):
            index.logger.info("Ingesting %d chunks into ChromaDB", len(corpus.chunks))
            index._ingest_corpus(corpus)
        else:
            index.logger.info("ChromaDB index loaded with %d chunks", index.collection.count())
            
        return index

    def _ingest_corpus(self, corpus: Corpus) -> None:
        """Embed and add corpus to ChromaDB."""
        if not corpus.chunks:
            return

        texts = [chunk.content for chunk in corpus.chunks]
        ids = [chunk.chunk_id for chunk in corpus.chunks]
        metadatas = [{"source": chunk.source_path.name} for chunk in corpus.chunks]
        
        # Embed in batches to avoid API or memory limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            
            embeddings = self.embedder.embed_texts(batch_texts)
            # Chroma expects a list of lists for embeddings
            embeddings_list = embeddings.tolist()
            
            self.collection.upsert(
                ids=batch_ids,
                embeddings=embeddings_list,
                metadatas=batch_metadatas,
                documents=batch_texts
            )

    @classmethod
    def load(cls, corpus: Corpus, settings: AppSettings) -> "VectorIndex | None":
        """ChromaDB handled persistence automatically, so we just 'build' (load)."""
        return cls.build(corpus, settings)

    @classmethod
    def _select_embedder(cls, settings: AppSettings) -> BaseEmbedder:
        if settings.vector_backend == "hash":
            return HashingEmbedder(dim=settings.vector_dim)
        if settings.vector_backend == "gemini":
            return GeminiEmbedder(settings)
        if settings.vector_backend == "openai":
            return OpenAIEmbedder(settings)
        if settings.gemini_api_key:
            try:
                return GeminiEmbedder(settings)
            except Exception:
                return HashingEmbedder(dim=settings.vector_dim)
        if settings.openai_api_key:
            try:
                return OpenAIEmbedder(settings)
            except Exception:
                return HashingEmbedder(dim=settings.vector_dim)
        return HashingEmbedder(dim=settings.vector_dim)

    def persist(self) -> None:
        """ChromaDB PersistentClient persists automatically on write/upsert."""
        pass

    def search(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.collection.count() == 0:
            return []
            
        query_vector = self.embedder.embed_query(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        search_results: list[VectorSearchResult] = []
        for chunk_id, dist in zip(ids, distances):
            # Chroma distances for cosine are 1 - similarity
            score = 1.0 - float(dist)
            if score <= 0.05: # Low similarity threshold
                continue
            search_results.append(VectorSearchResult(chunk_id=chunk_id, score=score))
            
        return search_results
