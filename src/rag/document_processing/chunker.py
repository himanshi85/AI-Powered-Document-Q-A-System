"""Semantic chunking strategies for documents"""

import re
from typing import List, Optional
from src.rag.document_processing.models import DocumentChunk


class SemanticChunker:
    """
    Chunks documents into semantically coherent units.
    Supports both fixed-size and semantic-aware chunking.
    """
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target tokens per chunk (approximate)
            chunk_overlap: Tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _count_tokens_approx(self, text: str) -> int:
        """Approximate token count (simple word-based estimate)"""
        return len(text.split())
    
    def _split_on_delimiters(self, text: str) -> List[str]:
        """Split text on semantic boundaries (sentences, paragraphs)"""
        # Split on double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        segments = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            # Further split on sentences
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            segments.extend(sentences)
        
        return [s.strip() for s in segments if s.strip()]
    
    def chunk(
        self,
        text: str,
        doc_id: str,
        source_doc: str,
        metadata: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a document into semantic units.
        
        Args:
            text: Document content to chunk
            doc_id: Document ID
            source_doc: Source filename
            metadata: Optional document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Split into segments
        segments = self._split_on_delimiters(text)
        
        chunks = []
        current_chunk = []
        current_char_pos = 0
        chunk_index = 0
        
        for segment in segments:
            current_chunk.append(segment)
            current_tokens = self._count_tokens_approx(' '.join(current_chunk))
            
            # Create chunk if we exceed size or this is the last segment
            if current_tokens >= self.chunk_size or segment == segments[-1]:
                chunk_text = ' '.join(current_chunk)
                
                if self._count_tokens_approx(chunk_text) >= self.min_chunk_size:
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    start_char = text.find(chunk_text)
                    end_char = start_char + len(chunk_text)
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        source_doc=source_doc,
                        chunk_index=chunk_index,
                        start_char=start_char if start_char >= 0 else current_char_pos,
                        end_char=end_char if end_char >= 0 else current_char_pos + len(chunk_text),
                        token_count=self._count_tokens_approx(chunk_text),
                        metadata=metadata.copy(),
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_char_pos += len(chunk_text) + 1
                
                # Reset for next chunk, keeping overlap
                if current_tokens >= self.chunk_size:
                    overlap_segments = []
                    remaining_tokens = 0
                    for seg in reversed(current_chunk):
                        overlap_segments.insert(0, seg)
                        remaining_tokens += self._count_tokens_approx(seg)
                        if remaining_tokens >= self.chunk_overlap:
                            break
                    current_chunk = overlap_segments
                else:
                    current_chunk = []
        
        return chunks
