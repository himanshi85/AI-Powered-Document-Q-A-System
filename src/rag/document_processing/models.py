from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    document_id: str
    title: str
    text: str
    source_path: str
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Backward-compatible alias."""
        return self.text


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source_path: str
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def content(self) -> str:
        """Backward-compatible alias."""
        return self.text


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank: int = 0