from typing import Dict, List

from src.rag.document_processing.models import RetrievalResult
from src.rag.generation.prompts import GroundingPrompts


class RAGGenerator:
    """
    Retrieval-augmented answer generator with:
    - grounded prompting
    - context filtering
    - source tracking
    - confidence estimation
    - evidence-aware outputs
    """

    def __init__(
        self,
        retriever,
        llm_fn,
        min_context_score: float = 0.2,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.min_context_score = min_context_score
        self.top_k = top_k

    def _filter_relevant_results(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Filter out weak retrieval results before sending to the LLM.
        """
        filtered = [r for r in results if r.score >= self.min_context_score]
        return filtered

    def _extract_sources(self, results: List[RetrievalResult]) -> List[str]:
        """
        Return unique source names from the retrieved chunks.
        """
        seen = set()
        sources = []

        for result in results:
            chunk = result.chunk

            source_name = None
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                source_name = chunk.metadata.get("title")

            if not source_name:
                source_name = getattr(chunk, "source_path", "Unknown Source")

            if source_name not in seen:
                seen.add(source_name)
                sources.append(source_name)

        return sources

    def _build_chunk_scores(self, results: List[RetrievalResult]) -> List[Dict]:
        """
        Return per-chunk score details for UI/debugging.
        """
        items = []

        for i, result in enumerate(results, start=1):
            items.append(
                {
                    "rank": i,
                    "chunk_number": i,
                    "chunk_id": result.chunk.chunk_id,
                    "source": (
                        result.chunk.metadata.get("title")
                        if hasattr(result.chunk, "metadata") and isinstance(result.chunk.metadata, dict)
                        else getattr(result.chunk, "source_path", "Unknown Source")
                    ),
                    "score": float(result.score),
                    "dense_score": float(result.dense_score),
                    "sparse_score": float(result.sparse_score),
                }
            )

        return items

    def _estimate_confidence(
        self,
        retrieved_results: List[RetrievalResult],
        filtered_results: List[RetrievalResult],
        answer_text: str,
    ) -> float:
        """
        Heuristic confidence score.

        Factors:
        - average score of filtered chunks
        - best chunk score
        - number of supporting chunks
        - penalty when no/weak context
        - penalty if answer explicitly says information is missing
        """
        if not retrieved_results or not filtered_results:
            return 0.0

        scores = [r.score for r in filtered_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        support_factor = min(len(filtered_results) / max(self.top_k, 1), 1.0)

        confidence = (
            0.45 * avg_score +
            0.35 * max_score +
            0.20 * support_factor
        )

        lower_answer = (answer_text or "").lower()
        uncertainty_markers = [
            "not enough information",
            "insufficient information",
            "not available in the provided documents",
            "cannot determine from the provided context",
            "partially supported",
            "missing from the provided documents",
        ]

        if any(marker in lower_answer for marker in uncertainty_markers):
            confidence *= 0.65

        confidence = max(0.0, min(confidence, 1.0))
        return confidence

    def _build_retrieval_reasoning(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        filtered_results: List[RetrievalResult],
    ) -> str:
        if not retrieved_results:
            return (
                f"No chunks were retrieved for the query: '{query}'. "
                "The answer was generated without usable context."
            )

        removed_count = len(retrieved_results) - len(filtered_results)

        reasoning = (
            f"Retrieved {len(retrieved_results)} chunks for query '{query}'. "
            f"Filtered to {len(filtered_results)} chunks using min_context_score={self.min_context_score:.2f}. "
        )

        if filtered_results:
            top = filtered_results[0]
            reasoning += (
                f"Top supporting chunk was '{top.chunk.chunk_id}' "
                f"with hybrid score {top.score:.3f} "
                f"(dense={top.dense_score:.3f}, sparse={top.sparse_score:.3f}). "
            )

        if removed_count > 0:
            reasoning += f"Excluded {removed_count} low-scoring chunk(s) before generation."

        return reasoning

    def _safe_fallback_answer(self) -> str:
        return (
            "Answer: The provided documents do not contain enough information to answer this question reliably.\n"
            "Evidence Used: None"
        )

    def generate(
        self,
        query: str,
        use_verification: bool = False,
    ) -> Dict:
        """
        Generate an answer from retrieved context.
        """
        retrieved_results = self.retriever.retrieve(query=query, top_k=self.top_k)
        filtered_results = self._filter_relevant_results(retrieved_results)

        if not filtered_results:
            answer = self._safe_fallback_answer()
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.0,
                "num_context_chunks": 0,
                "chunk_scores": [],
                "retrieval_reasoning": self._build_retrieval_reasoning(
                    query=query,
                    retrieved_results=retrieved_results,
                    filtered_results=filtered_results,
                ),
            }

        prompt = GroundingPrompts.user_prompt(query, filtered_results)
        raw_answer = self.llm_fn(prompt)

        if not raw_answer or not raw_answer.strip():
            raw_answer = self._safe_fallback_answer()

        sources = self._extract_sources(filtered_results)
        chunk_scores = self._build_chunk_scores(filtered_results)
        confidence = self._estimate_confidence(
            retrieved_results=retrieved_results,
            filtered_results=filtered_results,
            answer_text=raw_answer,
        )
        reasoning = self._build_retrieval_reasoning(
            query=query,
            retrieved_results=retrieved_results,
            filtered_results=filtered_results,
        )

        return {
            "answer": raw_answer,
            "sources": sources,
            "confidence": confidence,
            "num_context_chunks": len(filtered_results),
            "chunk_scores": chunk_scores,
            "retrieval_reasoning": reasoning,
        }