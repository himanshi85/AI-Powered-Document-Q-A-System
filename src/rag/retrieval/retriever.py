"""Hybrid retrieval combining dense and sparse search"""

from typing import List, Optional, Callable
import numpy as np
from src.rag.vector_store.base import VectorStore
from src.rag.document_processing.models import RetrievalResult


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Dense vector search (semantic similarity)
    2. Sparse BM25 search (keyword matching)
    
    Uses weighted combination for final ranking.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store backend
            embedding_fn: Function to embed text (required for dense search)
            dense_weight: Weight for dense search results (0-1)
            sparse_weight: Weight for sparse search results (0-1)
        """
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Normalize weights
        total = dense_weight + sparse_weight
        self.dense_weight /= total
        self.sparse_weight /= total
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            use_dense: Include dense vector search
            use_sparse: Include sparse BM25 search
            
        Returns:
            List of RetrievalResult objects ranked by combined score
        """
        if not use_dense and not use_sparse:
            raise ValueError("At least one retrieval method must be enabled")
        
        combined_results = {}
        
        # Dense search
        if use_dense and self.embedding_fn:
            try:
                query_embedding = self.embedding_fn(query)
                dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
                
                for result in dense_results:
                    if result.chunk_id not in combined_results:
                        combined_results[result.chunk_id] = {
                            "chunk_id": result.chunk_id,
                            "content": result.content,
                            "source_doc": result.source_doc,
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                        }
                    combined_results[result.chunk_id]["dense_score"] = result.score
            except Exception as e:
                print(f"Dense search failed: {e}")
        
        # Sparse search
        if use_sparse:
            try:
                sparse_results = self.vector_store.keyword_search(query, top_k=top_k * 2)
                
                for result in sparse_results:
                    if result.chunk_id not in combined_results:
                        combined_results[result.chunk_id] = {
                            "chunk_id": result.chunk_id,
                            "content": result.content,
                            "source_doc": result.source_doc,
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                        }
                    combined_results[result.chunk_id]["sparse_score"] = result.score
            except Exception as e:
                print(f"Sparse search failed: {e}")
        
        # Calculate combined scores and rank
        final_results = []
        for chunk_id, data in combined_results.items():
            combined_score = (
                data["dense_score"] * self.dense_weight +
                data["sparse_score"] * self.sparse_weight
            )
            
            final_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=data["content"],
                    source_doc=data["source_doc"],
                    score=combined_score,
                    search_type="hybrid",
                )
            )
        
        # Sort by combined score
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def retrieve_with_reasoning(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[List[RetrievalResult], dict]:
        """
        Retrieve with detailed reasoning about scores.
        
        Returns:
            (results, reasoning_dict)
        """
        results = self.retrieve(query, top_k=top_k)
        
        reasoning = {
            "query": query,
            "retrieval_method": "hybrid",
            "top_k_requested": top_k,
            "results_count": len(results),
            "weights": {
                "dense": self.dense_weight,
                "sparse": self.sparse_weight,
            },
        }
        
        return results, reasoning
