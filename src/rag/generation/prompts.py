"""Grounding prompts for strict context adherence"""


class GroundingPrompts:
    """Structured prompts designed for strict context grounding"""
    
    @staticmethod
    def system_prompt() -> str:
        """System prompt enforcing strict context adherence"""
        return """You are a helpful technical support assistant for product documentation.

CRITICAL INSTRUCTION: You MUST only answer questions using the provided context documents.

Rules:
1. ONLY use information from the provided context chunks.
2. If the context doesn't contain the answer, respond: "I don't have information about that in the documentation."
3. Do NOT use external knowledge or make assumptions.
4. Always cite which document(s) your answer comes from.
5. If you're uncertain, ask for clarification or say you need more information.
6. Be concise and clear in your responses.
7. Format citations as: [Source: document_name]"""
    
    @staticmethod
    def build_rag_prompt(
        query: str,
        context_chunks: list[str],
        sources: list[str],
    ) -> str:
        """
        Build a complete RAG prompt with context.
        
        Args:
            query: User question
            context_chunks: List of relevant document chunks
            sources: List of source document names
            
        Returns:
            Complete prompt with context
        """
        context_text = "\n\n".join([
            f"[Source: {source}]\n{chunk}"
            for chunk, source in zip(context_chunks, sources)
        ])
        
        prompt = f"""Based on the following documentation excerpts, answer the question.

DOCUMENTATION CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt
    
    @staticmethod
    def build_verification_prompt(
        original_query: str,
        retrieved_chunks: list[str],
        model_response: str,
    ) -> str:
        """
        Build a prompt for verifying if response is grounded in context.
        """
        context_text = "\n\n".join(retrieved_chunks)
        
        prompt = f"""Verify if the following response is grounded in the provided documentation.

DOCUMENTATION:
{context_text}

ORIGINAL QUESTION: {original_query}

MODEL RESPONSE: {model_response}

VERIFICATION QUESTIONS:
1. Does the response only use information from the documentation?
2. Are all claims properly sourced?
3. Is there any information that seems to come from outside the documentation?

Provide a brief verification result."""
        return prompt


class ResponseBuilder:
    """Build structured responses with citations"""
    
    @staticmethod
    def build_response(
        answer: str,
        sources: list[str],
        confidence: float = 1.0,
    ) -> dict:
        """
        Build a structured response with metadata.
        
        Args:
            answer: The generated answer
            sources: List of source documents
            confidence: Confidence score (0-1)
            
        Returns:
            Structured response dictionary
        """
        return {
            "answer": answer,
            "sources": list(set(sources)),  # Deduplicate sources
            "confidence": confidence,
            "grounded": True,
        }
    
    @staticmethod
    def build_fallback_response(
        query: str,
        reason: str = "No relevant information found",
    ) -> dict:
        """Build a response when no context is available"""
        return {
            "answer": f"I don't have information about that in the documentation. {reason}",
            "sources": [],
            "confidence": 0.0,
            "grounded": False,
        }
