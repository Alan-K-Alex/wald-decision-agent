from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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
    def __init__(self, settings: AppSettings, embedder: BaseEmbedder, chunks: dict[str, DocumentChunk], matrix: np.ndarray) -> None:
        self.settings = settings
        self.embedder = embedder
        self.chunks = chunks
        self.logger = get_logger("retrieval.vector_index")
        self.matrix = self._sanitize(matrix)
        self.chunk_ids = list(chunks.keys())

    @classmethod
    def build(cls, corpus: Corpus, settings: AppSettings) -> "VectorIndex":
        embedder = cls._select_embedder(settings)
        chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
        texts = [chunk.content for chunk in corpus.chunks]
        matrix = embedder.embed_texts(texts) if texts else np.zeros((0, settings.vector_dim), dtype=np.float32)
        index = cls(settings=settings, embedder=embedder, chunks=chunks, matrix=matrix)
        index.persist()
        return index

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
        store_dir = self.settings.vector_store_path
        store_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = store_dir / "index_metadata.json"
        matrix_path = store_dir / "chunk_vectors.npy"
        np.save(matrix_path, self.matrix)
        metadata = {
            "backend": self.embedder.backend_name,
            "chunk_ids": self.chunk_ids,
            "dim": int(self.matrix.shape[1]) if self.matrix.ndim == 2 and self.matrix.size else self.settings.vector_dim,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.matrix.size == 0:
            return []
        query_vector = self._sanitize(self.embedder.embed_query(query))
        scores = (self.matrix * query_vector).sum(axis=1, dtype=np.float32)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[VectorSearchResult] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(VectorSearchResult(chunk_id=self.chunk_ids[idx], score=score))
        return results

    def _sanitize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = np.clip(array, -1.0e3, 1.0e3)
        if array.ndim == 2:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = array / norms
            return np.clip(normalized, -1.0, 1.0).astype(np.float32)
        if array.ndim == 1:
            norm = np.linalg.norm(array)
            norm = norm if norm != 0 else 1.0
            normalized = array / norm
            return np.clip(normalized, -1.0, 1.0).astype(np.float32)
        self.logger.warning("Unexpected embedding shape encountered: %s", array.shape)
        return array.astype(np.float32)
