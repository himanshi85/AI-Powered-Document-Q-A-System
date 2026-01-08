"""In-memory vector store implementation with BM25 keyword search"""

from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.rag.vector_store.base import VectorStore
from src.rag.document_processing.models import DocumentChunk, RetrievalResult


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store with:
    - Dense vector similarity search using cosine distance
    - BM25 keyword search
    """
    
    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the store"""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            # Store embedding if provided
            if chunk.embedding:
                self.embeddings[chunk.chunk_id] = np.array(chunk.embedding)
        
        # Rebuild BM25 index for keyword search
        self._rebuild_bm25_index()
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from all chunks"""
        self.tokenized_corpus = [
            chunk.content.lower().split()
            for chunk in self.chunks.values()
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Dense vector similarity search.
        Uses cosine similarity.
        """
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        results = []
        
        for chunk_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_vec, embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(embedding) + 1e-10
            )
            # Normalize to 0-1 range
            score = (similarity + 1) / 2
            
            chunk = self.chunks[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    source_doc=chunk.source_doc,
                    score=float(score),
                    search_type="dense_vector",
                )
            )
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        BM25 keyword search.
        Good for exact terms and product codes.
        """
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Create results
        chunk_ids = list(self.chunks.keys())
        results = []
        
        for chunk_id, score in zip(chunk_ids, scores):
            if score > 0:  # Only include positive scores
                chunk = self.chunks[chunk_id]
                # Normalize BM25 score
                normalized_score = min(float(score) / 100.0, 1.0)
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        content=chunk.content,
                        source_doc=chunk.source_doc,
                        score=normalized_score,
                        search_type="bm25_keyword",
                    )
                )
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID"""
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
            self.embeddings.pop(chunk_id, None)
        self._rebuild_bm25_index()
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk"""
        return self.chunks.get(chunk_id)
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            "total_chunks": len(self.chunks),
            "chunks_with_embeddings": len(self.embeddings),
            "total_tokens": sum(c.token_count for c in self.chunks.values()),
        }
