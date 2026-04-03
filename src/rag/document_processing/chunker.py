import re
from typing import List

from .models import Document, DocumentChunk

try:
    import tiktoken
except ImportError:
    tiktoken = None


class SemanticChunker:
    """
    Metadata-aware chunker with approximate token control and overlap.
    It first splits into paragraphs, then sentences, then groups them
    into chunks close to target_tokens.
    """

    def __init__(self, target_tokens: int = 350, overlap_tokens: int = 60):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = None

        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return max(1, len(text.split()))

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        if self._count_tokens(paragraph) <= self.target_tokens:
            return [paragraph]

        sentences = self._split_into_sentences(paragraph)
        if not sentences:
            return [paragraph]

        groups = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)

            if current and current_tokens + sent_tokens > self.target_tokens:
                groups.append(" ".join(current).strip())
                current = [sentence]
                current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            groups.append(" ".join(current).strip())

        return groups

    def _create_overlap_text(self, chunk_text: str) -> str:
        words = chunk_text.split()
        if not words:
            return ""
        overlap_word_count = min(len(words), self.overlap_tokens)
        return " ".join(words[-overlap_word_count:])

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        text = self._normalize_text(document.text)
        paragraphs = self._split_into_paragraphs(text)

        units = []
        for paragraph in paragraphs:
            units.extend(self._split_large_paragraph(paragraph))

        chunks: List[DocumentChunk] = []
        current_parts = []
        current_tokens = 0
        current_start = 0
        running_char_pointer = 0
        chunk_index = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)

            if current_parts and current_tokens + unit_tokens > self.target_tokens:
                chunk_text = "\n\n".join(current_parts).strip()
                end_char = current_start + len(chunk_text)

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                        document_id=document.document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=end_char,
                        source_path=document.source_path,
                        doc_type=document.doc_type,
                        metadata={
                            **document.metadata,
                            "title": document.title,
                            "chunk_index": chunk_index,
                        },
                    )
                )

                overlap_text = self._create_overlap_text(chunk_text)
                current_parts = [overlap_text, unit] if overlap_text else [unit]
                current_tokens = self._count_tokens(" ".join(current_parts))
                current_start = max(0, end_char - len(overlap_text))
                chunk_index += 1
            else:
                if not current_parts:
                    current_start = running_char_pointer
                current_parts.append(unit)
                current_tokens += unit_tokens

            running_char_pointer += len(unit) + 2

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            end_char = current_start + len(chunk_text)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                    document_id=document.document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=end_char,
                    source_path=document.source_path,
                    doc_type=document.doc_type,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "chunk_index": chunk_index,
                    },
                )
            )

        return chunks