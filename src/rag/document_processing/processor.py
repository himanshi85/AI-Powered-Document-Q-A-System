"""Main document processor orchestrating the pipeline"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from src.rag.document_processing.models import Document, DocumentChunk
from src.rag.document_processing.chunker import SemanticChunker


class DocumentProcessor:
    """
    Orchestrates document loading, cleaning, and chunking.
    """
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        """Initialize document processor"""
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.documents: Dict[str, Document] = {}
        self.chunks: List[DocumentChunk] = []
    
    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from a directory.
        
        Supports: .txt, .md, .json files
        """
        documents = []
        path = Path(directory)
        
        for file_path in path.glob('**/*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json','.csv']:
                try:
                    doc = self._load_single_file(file_path)
                    if doc:
                        documents.append(doc)
                        self.documents[doc.doc_id] = doc
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """Load and parse a single file"""
        doc_id = file_path.stem
        filename = file_path.name
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    content = data.get('content', '') or str(data)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Clean content
            content = self._clean_text(content)
            
            if not content.strip():
                return None
            
            return Document(
                doc_id=doc_id,
                filename=filename,
                content=content,
                doc_type=self._infer_doc_type(filename),
            )
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = text.replace('\x00', '')
        return text.strip()
    
    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename"""
        lower_name = filename.lower()
        if 'faq' in lower_name:
            return 'faq'
        elif 'manual' in lower_name:
            return 'product_manual'
        elif 'api' in lower_name:
            return 'api_docs'
        elif 'guide' in lower_name:
            return 'user_guide'
        else:
            return 'general_document'
    
    def process(self) -> List[DocumentChunk]:
        """
        Process all loaded documents into chunks.
        """
        self.chunks = []
        
        for doc_id, doc in self.documents.items():
            metadata = {
                'doc_type': doc.doc_type,
                'filename': doc.filename,
                **doc.metadata
            }
            
            chunks = self.chunker.chunk(
                text=doc.content,
                doc_id=doc_id,
                source_doc=doc.filename,
                metadata=metadata,
            )
            self.chunks.extend(chunks)
        
        return self.chunks
    
    def get_chunks_for_doc(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific document"""
        return [c for c in self.chunks if c.source_doc == self.documents.get(doc_id, {}).filename]
    
    def export_chunks(self, output_path: str) -> None:
        """Export chunks to JSON for inspection"""
        data = [
            {
                'chunk_id': c.chunk_id,
                'content': c.content[:100] + '...' if len(c.content) > 100 else c.content,
                'source_doc': c.source_doc,
                'token_count': c.token_count,
            }
            for c in self.chunks
        ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
