from __future__ import annotations

import math
from collections import Counter

from ..core.config import AppSettings
from ..core.logging import get_logger
from ..core.models import Corpus, RetrievedChunk
from ..utils import tokenize
from .vector_index import VectorIndex


class HybridRetriever:
    def __init__(self, corpus: Corpus, settings: AppSettings) -> None:
        self.corpus = corpus
        self.settings = settings
        self.logger = get_logger("retrieval.hybrid")
        self.df = self._document_frequencies()
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in corpus.chunks}
        try:
            self.vector_index = VectorIndex.build(corpus, settings)
        except Exception as exc:
            self.logger.warning("Failed to build vector index: %s. Falling back to lexical-only retrieval.", exc)
            self.vector_index = None

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        k = top_k or self.settings.top_k
        lexical_scores: dict[str, float] = {}
        for chunk in self.corpus.chunks:
            chunk_tokens = tokenize(chunk.content)
            score = self._score_tokens(query_counter, chunk_tokens)
            if chunk.metadata.get("sheet_name") and chunk.metadata["sheet_name"].lower() in query.lower():
                score += 0.2
            lexical_scores[chunk.chunk_id] = score

        vector_scores: dict[str, float] = {}
        if self.vector_index is not None:
            try:
                vector_hits = self.vector_index.search(query, top_k=max(k * 3, k))
                vector_scores = {item.chunk_id: item.score for item in vector_hits}
            except Exception as exc:
                self.logger.debug("Vector search failed, using lexical-only scores: %s", exc)

        combined_scores: list[RetrievedChunk] = []
        lexical_max = max(lexical_scores.values(), default=1.0) or 1.0
        vector_max = max(vector_scores.values(), default=1.0) or 1.0
        candidate_ids = {
            chunk_id
            for chunk_id, score in lexical_scores.items()
            if score > 0
        } | set(vector_scores)
        for chunk_id in candidate_ids:
            # Verify chunk exists in lookup before processing
            if chunk_id not in self.chunk_lookup:
                self.logger.debug("Chunk %s not found in lookup, skipping", chunk_id)
                continue
            lexical_score = lexical_scores.get(chunk_id, 0.0) / lexical_max
            vector_score = vector_scores.get(chunk_id, 0.0) / vector_max
            combined = self.settings.lexical_weight * lexical_score + self.settings.vector_weight * vector_score
            if combined <= 0:
                continue
            combined_scores.append(RetrievedChunk(chunk=self.chunk_lookup[chunk_id], score=combined))
        combined_scores.sort(key=lambda item: item.score, reverse=True)
        self.logger.debug("Top retrieval scores for '%s': %s", query, [round(item.score, 4) for item in combined_scores[:k]])
        return combined_scores[:k]

    def _document_frequencies(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for chunk in self.corpus.chunks:
            for token in set(tokenize(chunk.content)):
                df[token] = df.get(token, 0) + 1
        return df

    def _score_tokens(self, query_counter: Counter[str], chunk_tokens: list[str]) -> float:
        if not chunk_tokens:
            return 0.0
        chunk_counter = Counter(chunk_tokens)
        chunk_len = len(chunk_tokens)
        score = 0.0
        corpus_size = max(1, len(self.corpus.chunks))
        for token, q_tf in query_counter.items():
            tf = chunk_counter.get(token, 0)
            if tf == 0:
                continue
            idf = math.log((corpus_size + 1) / (1 + self.df.get(token, 0))) + 1
            score += (tf / chunk_len) * idf * q_tf
        return score
