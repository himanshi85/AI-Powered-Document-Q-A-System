from typing import List

from src.rag.document_processing.models import RetrievalResult


class GroundingPrompts:
    """
    Prompt builder for grounded RAG answering.
    """

    @staticmethod
    def system_prompt() -> str:
        return (
            "You are a retrieval-grounded assistant. "
            "Answer the user's question using ONLY the provided context chunks. "
            "Do not invent facts, steps, policies, numbers, or explanations that are not supported by the context. "
            "If the context is insufficient, clearly say that the answer is not available in the provided documents. "
            "When possible, synthesize information across multiple chunks, but stay faithful to the text. "
            "Prefer concise, precise, and factual answers. "
            "Do not claim certainty when the context is partial or ambiguous."
        )

    @staticmethod
    def format_context(results: List[RetrievalResult]) -> str:
        """
        Build a structured context block for the LLM.
        """
        if not results:
            return "No context available."

        context_parts = []

        for i, result in enumerate(results, start=1):
            chunk = result.chunk

            source_name = None
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                source_name = chunk.metadata.get("title")

            if not source_name:
                source_name = getattr(chunk, "source_path", "Unknown Source")

            chunk_index = getattr(chunk, "chunk_index", i - 1)

            context_parts.append(
                f"[CHUNK {i}]\n"
                f"Source: {source_name}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Original Chunk Index: {chunk_index}\n"
                f"Retrieval Score: {result.score:.4f}\n"
                f"Dense Score: {result.dense_score:.4f}\n"
                f"Sparse Score: {result.sparse_score:.4f}\n"
                f"Content:\n{chunk.content}\n"
            )

        return "\n" + ("\n" + "-" * 80 + "\n").join(context_parts)

    @staticmethod
    def user_prompt(query: str, results: List[RetrievalResult]) -> str:
        context_block = GroundingPrompts.format_context(results)

        return (
            f"User Question:\n{query}\n\n"
            f"Retrieved Context:\n{context_block}\n\n"
            "Instructions:\n"
            "1. Answer ONLY from the retrieved context above.\n"
            "2. If the answer is fully supported, provide a direct answer.\n"
            "3. If the answer is partially supported, clearly say what is supported and what is missing.\n"
            "4. If the answer is not present, say that the provided documents do not contain enough information.\n"
            "5. After the answer, include a short section called 'Evidence Used' listing the chunk numbers you relied on.\n"
            "6. Do not mention any knowledge outside the provided context.\n"
            "7. Do not fabricate citations.\n\n"
            "Response format:\n"
            "Answer: <your grounded answer>\n"
            "Evidence Used: <example: CHUNK 1, CHUNK 3>\n"
        )