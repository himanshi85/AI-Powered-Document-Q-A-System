"""Main RAG generator combining retrieval and generation"""

from typing import Optional, Callable, List, Dict
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.prompts import GroundingPrompts, ResponseBuilder
from src.rag.document_processing.models import RetrievalResult


class RAGGenerator:
    """
    Complete RAG pipeline: retrieves context and generates grounded responses.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_fn: Callable[[str], str],
        min_context_score: float = 0.3,
        top_k: int = 5,
    ):
        """
        Initialize RAG generator.
        
        Args:
            retriever: HybridRetriever instance
            llm_fn: Function to call LLM (takes prompt, returns response)
            min_context_score: Minimum relevance score for context
            top_k: Number of context chunks to retrieve
        """
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.min_context_score = min_context_score
        self.top_k = top_k
    
    def generate(
        self,
        query: str,
        use_verification: bool = False,
    ) -> Dict:
        """
        Generate a grounded response to a query.
        
        Args:
            query: User question
            use_verification: Verify response grounding (requires extra LLM call)
            
        Returns:
            Response dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        retrieved_chunks, retrieval_reasoning = self.retriever.retrieve_with_reasoning(
            query=query,
            top_k=self.top_k,
        )
        
        # Filter by minimum score
        relevant_chunks = [
            c for c in retrieved_chunks
            if c.score >= self.min_context_score
        ]
        
        if not relevant_chunks:
            return {
                **ResponseBuilder.build_fallback_response(
                    query,
                    reason="Could not find relevant documentation.",
                ),
                "retrieval_reasoning": retrieval_reasoning,
            }
        
        # Step 2: Build RAG prompt with context
        context_texts = [chunk.content for chunk in relevant_chunks]
        sources = [chunk.source_doc for chunk in relevant_chunks]
        
        rag_prompt = GroundingPrompts.build_rag_prompt(
            query=query,
            context_chunks=context_texts,
            sources=sources,
        )
        
        # Step 3: Generate response
        try:
            answer = self.llm_fn(rag_prompt)
        except Exception as e:
            return {
                **ResponseBuilder.build_fallback_response(
                    query,
                    reason=f"LLM error: {str(e)}",
                ),
                "retrieval_reasoning": retrieval_reasoning,
            }
        
        # Step 4: Optional verification
        verification = None
        if use_verification:
            verification_prompt = GroundingPrompts.build_verification_prompt(
                original_query=query,
                retrieved_chunks=context_texts,
                model_response=answer,
            )
            try:
                verification = self.llm_fn(verification_prompt)
            except Exception as e:
                print(f"Verification failed: {e}")
        
        # Step 5: Build final response
        response = ResponseBuilder.build_response(
            answer=answer,
            sources=sources,
            confidence=min(1.0, sum(c.score for c in relevant_chunks) / len(relevant_chunks)),
        )
        
        # Add metadata
        response["retrieval_reasoning"] = retrieval_reasoning
        response["num_context_chunks"] = len(relevant_chunks)
        response["verification"] = verification
        response["chunk_scores"] = [
            {"chunk_id": c.chunk_id, "score": c.score}
            for c in relevant_chunks
        ]
        
        return response
    
    def generate_batch(
        self,
        queries: List[str],
        use_verification: bool = False,
    ) -> List[Dict]:
        """Generate responses for multiple queries"""
        return [
            self.generate(query, use_verification=use_verification)
            for query in queries
        ]
    
    def generate_with_followup(
        self,
        query: str,
        followup_questions: List[str],
    ) -> List[Dict]:
        """
        Generate response and handle follow-up questions,
        maintaining context relevance.
        """
        results = [self.generate(query)]
        
        # For follow-ups, use same context
        first_response = results[0]
        sources = first_response.get("sources", [])
        
        for followup in followup_questions:
            followup_result = self.generate(followup)
            results.append(followup_result)
        
        return results
