"""Vector store factory for different backends"""

from typing import Optional
from src.rag.vector_store.base import VectorStore
from src.rag.vector_store.in_memory import InMemoryVectorStore


class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def create(
        store_type: str = "in_memory",
        **kwargs
    ) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: Type of store ("in_memory", "pinecone", "weaviate", "milvus")
            **kwargs: Store-specific configuration
            
        Returns:
            VectorStore instance
        """
        if store_type == "in_memory":
            return InMemoryVectorStore()
        
        elif store_type == "pinecone":
            try:
                from src.rag.vector_store.pinecone_store import PineconeVectorStore
                return PineconeVectorStore(**kwargs)
            except ImportError:
                raise ImportError(
                    "Pinecone store requires: pip install '[pinecone]'"
                )
        
        elif store_type == "weaviate":
            try:
                from src.rag.vector_store.weaviate_store import WeaviateVectorStore
                return WeaviateVectorStore(**kwargs)
            except ImportError:
                raise ImportError(
                    "Weaviate store requires: pip install '[weaviate]'"
                )
        
        elif store_type == "milvus":
            try:
                from src.rag.vector_store.milvus_store import MilvusVectorStore
                return MilvusVectorStore(**kwargs)
            except ImportError:
                raise ImportError(
                    "Milvus store requires: pip install '[milvus]'"
                )
        
        else:
            raise ValueError(f"Unknown store type: {store_type}")
