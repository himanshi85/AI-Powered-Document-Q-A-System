"""Base vector store interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.rag.document_processing.models import DocumentChunk, RetrievalResult


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Search for similar chunks using embeddings"""
        pass
    
    @abstractmethod
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Keyword-based search (BM25 style)"""
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID"""
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID"""
        pass
