# src/rag/vector_store/in_memory.py

import math
import re
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.document_processing.models import DocumentChunk, RetrievalResult


class InMemoryVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.chunk_map: Dict[str, DocumentChunk] = {}
        self.embeddings: List[np.ndarray] = []

        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids_in_order: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _normalize_scores_minmax(self, scores: List[float]) -> List[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if math.isclose(min_score, max_score):
            if max_score == 0:
                return [0.0 for _ in scores]
            return [1.0 for _ in scores]

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _rebuild_bm25_index(self) -> None:
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in self.chunks]
        self.chunk_ids_in_order = [chunk.chunk_id for chunk in self.chunks]

        if self.tokenized_corpus and any(len(tokens) > 0 for tokens in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add chunks to the in-memory store.
        Expects chunk.embedding to already be present.
        """
        for chunk in chunks:
            if chunk.chunk_id in self.chunk_map:
                continue

            self.chunks.append(chunk)
            self.chunk_map[chunk.chunk_id] = chunk

            if getattr(chunk, "embedding", None) is not None:
                self.embeddings.append(np.array(chunk.embedding, dtype=np.float32))
            else:
                # keep index aligned even if missing embedding
                self.embeddings.append(None)

        self._rebuild_bm25_index()

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunk_map.get(chunk_id)

    def delete_chunk(self, chunk_id: str) -> bool:
        if chunk_id not in self.chunk_map:
            return False

        new_chunks = []
        for chunk in self.chunks:
            if chunk.chunk_id != chunk_id:
                new_chunks.append(chunk)

        self.chunks = new_chunks
        self.chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}

        self.embeddings = []
        for chunk in self.chunks:
            if getattr(chunk, "embedding", None) is not None:
                self.embeddings.append(np.array(chunk.embedding, dtype=np.float32))
            else:
                self.embeddings.append(None)

        self._rebuild_bm25_index()
        return True

    def get_stats(self) -> Dict:
        total_embeddings = sum(1 for e in self.embeddings if e is not None)
        return {
            "total_chunks": len(self.chunks),
            "total_embeddings": total_embeddings,
            "bm25_indexed": self.bm25 is not None,
        }

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """
        Dense cosine similarity search.
        Returns normalized dense scores in [0, 1].
        """
        if not self.chunks or not self.embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        raw_scores = []
        valid_chunks = []

        for chunk, emb in zip(self.chunks, self.embeddings):
            if emb is None:
                continue

            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0 or query_norm == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = float(np.dot(query_vec, emb) / (query_norm * emb_norm))

            raw_scores.append(cosine_sim)
            valid_chunks.append(chunk)

        if not raw_scores:
            return []

        normalized_scores = self._normalize_scores_minmax(raw_scores)

        scored_results = []
        for chunk, score in zip(valid_chunks, normalized_scores):
            scored_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    sparse_score=0.0,
                )
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        BM25 keyword search.
        Returns normalized sparse scores in [0, 1].
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query).tolist()
        normalized_scores = self._normalize_scores_minmax(raw_scores)

        scored_results = []
        for chunk, score in zip(self.chunks, normalized_scores):
            scored_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dense_score=0.0,
                    sparse_score=score,
                )
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]