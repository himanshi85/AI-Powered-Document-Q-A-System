import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

from src.rag.document_processing.processor import DocumentProcessor
from src.rag.vector_store import VectorStoreFactory
from src.rag.retrieval import HybridRetriever, CrossEncoderReranker
from src.rag.generation import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts


# -----------------------------------------------------------------------------
# ENV + CONFIG
# -----------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in .env")

genai.configure(api_key=GOOGLE_API_KEY)

CONFIG = {
    "target_tokens": 350,
    "overlap_tokens": 60,
    "dense_weight": 0.6,
    "sparse_weight": 0.4,
    "min_context_score": 0.2,
    "top_k": 5,
    "max_chunks_per_document": 3,
    "embedding_model": "models/gemini-embedding-001",
    "generation_model": "models/gemini-2.5-flash",
}

def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def make_chunk_backward_compatible(chunk) -> None:
    if not hasattr(chunk, "token_count"):
        chunk.token_count = estimate_token_count(chunk.content)

    if not hasattr(chunk, "source_doc"):
        title = None
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            title = chunk.metadata.get("title")
        chunk.source_doc = title or getattr(chunk, "source_path", "Unknown Source")


def make_document_display_compatible(doc) -> None:
    if not hasattr(doc, "filename"):
        source_name = Path(
            getattr(doc, "source_path", getattr(doc, "title", "document"))
        ).name
        doc.filename = source_name


def setup_embedding_fn():
    def embed_text(text: str):
        result = genai.embed_content(
            model=CONFIG["embedding_model"],
            content=text,
        )

        if isinstance(result, dict):
            return result["embedding"]

        if hasattr(result, "embedding"):
            return result.embedding

        raise ValueError("Unexpected embedding response format.")

    return embed_text


def setup_llm_fn():
    system_instruction = GroundingPrompts.system_prompt()
    model = genai.GenerativeModel(
        CONFIG["generation_model"],
        system_instruction=system_instruction,
    )

    def generate_response(prompt: str) -> str:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Empty text response from generation model.")
        return text

    return generate_response


def print_pipeline_summary(documents, chunks):
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Documents loaded : {len(documents)}")
    print(f"Chunks created   : {len(chunks)}")
    print(f"Estimated tokens : {sum(getattr(c, 'token_count', 0) for c in chunks)}")

    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(
            f"  {i}. {getattr(doc, 'filename', doc.title)} "
            f"(type={doc.doc_type}, chars={len(doc.content)})"
        )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    data_dir = input("Enter folder path containing documents: ").strip()

    if not data_dir:
        raise ValueError("Folder path cannot be empty.")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    print("Initializing RAG pipeline...")

    processor = DocumentProcessor(
        target_tokens=CONFIG["target_tokens"],
        overlap_tokens=CONFIG["overlap_tokens"],
    )

    print("\n[1/5] Loading documents...")
    documents = processor.load_documents_from_directory(data_dir)

    if not documents:
        raise ValueError(
            "No supported documents found in the folder. "
            "Add TXT, MD, JSON, CSV, PDF, or DOCX files."
        )

    for doc in documents:
        make_document_display_compatible(doc)

    print("[2/5] Creating chunks...")
    chunks = processor.process_documents(documents)

    for chunk in chunks:
        make_chunk_backward_compatible(chunk)

    print_pipeline_summary(documents, chunks)

    print("\n[3/5] Creating embeddings...")
    embedding_fn = setup_embedding_fn()

    for i, chunk in enumerate(chunks, 1):
        chunk.embedding = embedding_fn(chunk.content)
        if i % 10 == 0 or i == len(chunks):
            print(f"  Embedded {i}/{len(chunks)} chunks")

    print("\n[4/5] Building vector store...")
    vector_store = VectorStoreFactory.create("in_memory")
    vector_store.add_chunks(chunks)

    print("[5/5] Initializing retriever and generator...")
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        enabled=True,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        dense_weight=CONFIG["dense_weight"],
        sparse_weight=CONFIG["sparse_weight"],
        max_chunks_per_document=CONFIG["max_chunks_per_document"],
        reranker=reranker,
        rerank_top_n=15,
        rerank_weight=0.35,
    )

    if reranker.available():
        print("Reranker enabled: cross-encoder/ms-marco-MiniLM-L6-v2")
    else:
        print("Reranker unavailable. Continuing with hybrid retrieval only.")

    llm_fn = setup_llm_fn()
    generator = RAGGenerator(
        retriever=retriever,
        llm_fn=llm_fn,
        min_context_score=CONFIG["min_context_score"],
        top_k=CONFIG["top_k"],
    )

    print("\nSystem ready. Type 'exit' to quit.\n")

    while True:
        q = input("Query: ").strip()
        if q.lower() == "exit":
            print("Exiting.")
            break

        if not q:
            continue

        try:
            res = generator.generate(q)

            print("\nAnswer:")
            print(res["answer"])
            print(f"\nConfidence: {res['confidence']:.2f}")

            if res.get("sources"):
                print("Sources:")
                for source in res["sources"]:
                    print(f"  - {source}")

            print("\n" + "-" * 80)

        except Exception as e:
            print(f"Error: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()