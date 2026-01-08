"""Data models for document processing"""

from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentChunk(BaseModel):
    """Represents a single document chunk with metadata"""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The actual text content")
    source_doc: str = Field(..., description="Original document source/filename")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    start_char: int = Field(..., description="Starting character position in original document")
    end_char: int = Field(..., description="Ending character position in original document")
    token_count: int = Field(..., description="Number of tokens in this chunk")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding (optional)")


class Document(BaseModel):
    """Represents a source document"""
    
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Source filename")
    content: str = Field(..., description="Full document content")
    doc_type: str = Field(default="product_manual", description="Type of document")
    metadata: dict = Field(default_factory=dict, description="Document-level metadata")


class RetrievalResult(BaseModel):
    """Result from retrieval"""
    
    chunk_id: str
    content: str
    source_doc: str
    score: float = Field(..., description="Relevance score (0-1)")
    search_type: str = Field(..., description="Type of search that returned this result")
