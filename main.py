"""Main RAG pipeline example with Gemini LLM"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

from src.rag.document_processing import DocumentProcessor
from src.rag.vector_store import VectorStoreFactory
from src.rag.retrieval import HybridRetriever
from src.rag.generation import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts


def setup_embedding_fn():
    """Setup embedding function using Gemini"""
    def embed_text(text: str):
        """Embed text using Gemini's embedding model"""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding failed: {e}")
            # Return a dummy embedding on failure
            return [0.0] * 768
    
    return embed_text


def setup_llm_fn():
    """Setup LLM function using Gemini"""
    system_instruction = GroundingPrompts.system_prompt()
    model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=system_instruction
    )
    
    def generate_response(prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
    
    return generate_response


def main():
    """Main RAG pipeline demonstration"""
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Please configure .env file.")
    
    genai.configure(api_key=api_key)
    
    print("=" * 80)
    print("RAG System for Product Documentation QA")
    print("=" * 80)
    
    # Step 1: Load and process documents
    print("\n[1] DOCUMENT PROCESSING")
    print("-" * 80)
    processor = DocumentProcessor(chunk_size=400, chunk_overlap=100)
    
    docs = processor.load_documents("data/")
    print(f"✓ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.filename} ({len(doc.content)} chars)")
    
    chunks = processor.process()
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Total tokens: {sum(c.token_count for c in chunks)}")
    
    # Step 2: Setup vector store
    print("\n[2] VECTOR STORE SETUP")
    print("-" * 80)
    vector_store = VectorStoreFactory.create("in_memory")
    print("✓ Created in-memory vector store")
    
    # Setup embedding function
    embedding_fn = setup_embedding_fn()
    print("✓ Setup Gemini embedding function")
    
    # Embed and store chunks
    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        if i % max(1, len(chunks) // 4) == 0:
            print(f"  Progress: {i}/{len(chunks)}")
        try:
            embedding = embedding_fn(chunk.content)
            chunk.embedding = embedding
        except Exception as e:
            print(f"Embedding failed for chunk {chunk.chunk_id}: {e}")
    
    vector_store.add_chunks(chunks)
    stats = vector_store.get_stats()
    print(f"✓ Vector store ready: {stats}")
    
    # Step 3: Setup retrieval
    print("\n[3] RETRIEVAL SETUP")
    print("-" * 80)
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        dense_weight=0.6,
        sparse_weight=0.4,
    )
    print("✓ Hybrid retriever configured (60% dense, 40% sparse)")
    
    # Step 4: Setup generation
    print("\n[4] GENERATION SETUP")
    print("-" * 80)
    llm_fn = setup_llm_fn()
    print("✓ Gemini LLM configured")
    
    generator = RAGGenerator(
        retriever=retriever,
        llm_fn=llm_fn,
        min_context_score=0.2,
        top_k=5,
    )
    print("✓ RAG generator ready")
    
    # Step 5: Run example queries
    print("\n[5] EXAMPLE QUERIES")
    print("=" * 80)
    
    queries = [
        "How do I troubleshoot WiFi connection issues?",
        # "What is the recommended water temperature for brewing?",
        # "Can multiple phones control the same coffee maker?",
        # "How often should I clean the machine?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        
        response = generator.generate(query, use_verification=False)
        
        print(f"Answer: {response['answer'][:500]}...")
        print(f"Sources: {', '.join(response['sources'])}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Retrieved {response['num_context_chunks']} context chunks")
        
        if response['chunk_scores']:
            print("Chunk scores:")
            for score in response['chunk_scores'][:3]:
                print(f"  - {score['chunk_id']}: {score['score']:.3f}")
    
    print("\n" + "=" * 80)
    print("RAG Pipeline demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
