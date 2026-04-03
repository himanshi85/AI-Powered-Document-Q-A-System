import os
import tempfile
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from src.rag.document_processing.processor import DocumentProcessor
from src.rag.vector_store import VectorStoreFactory
from src.rag.retrieval import HybridRetriever, CrossEncoderReranker
from src.rag.generation import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts

load_dotenv()

APP_CONFIG = {
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file. "
        "Create a .env file in the project root and add:\n"
        "GOOGLE_API_KEY=your_actual_key"
    )

genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Intelligent Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 0.5rem;
    }
    .debug-box {
        padding: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: rgba(128,128,128,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_session() -> None:
    defaults = {
        "vector_store": None,
        "retriever": None,
        "generator": None,
        "documents": [],
        "chunks": [],
        "processed": False,
        "query_history": [],
        "last_uploaded_names": [],
        "last_response": None,
        "last_query": None,
        "staged_files": {},          
        "processed_file_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()

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


def safe_preview(text: Optional[str], limit: int = 800) -> str:
    if not text:
        return ""
    return text[:limit] + ("..." if len(text) > limit else "")


def stage_uploaded_files(uploaded_files: List) -> int:
    """
    Store uploaded files in session state so multiple upload rounds
    can be combined before processing.
    """
    if not uploaded_files:
        return 0

    added_count = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        existing = st.session_state.staged_files.get(uploaded_file.name)

        # Replace if same file name is uploaded again with new content
        if existing != file_bytes:
            st.session_state.staged_files[uploaded_file.name] = file_bytes
            added_count += 1

    return added_count


def reset_pipeline(clear_staged: bool = False) -> None:
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.processed = False
    st.session_state.last_response = None
    st.session_state.last_query = None
    st.session_state.processed_file_names = []

    if clear_staged:
        st.session_state.staged_files = {}
        st.session_state.last_uploaded_names = []


def get_chunk_by_id(chunk_id: str):
    for chunk in st.session_state.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    return None

@st.cache_resource(show_spinner=False)
def setup_embedding_fn():
    def embed_text(text: str):
        try:
            result = genai.embed_content(
                model=APP_CONFIG["embedding_model"],
                content=text,
            )

            if isinstance(result, dict):
                embedding = result.get("embedding")
                if embedding is None:
                    raise ValueError("Embedding response did not contain 'embedding'.")
                return embedding

            if hasattr(result, "embedding"):
                return result.embedding

            raise ValueError("Unexpected embedding response format.")

        except Exception as e:
            raise RuntimeError(
                f"Embedding failed using model '{APP_CONFIG['embedding_model']}': {e}"
            ) from e

    return embed_text


@st.cache_resource(show_spinner=False)
def setup_llm_fn():
    system_instruction = GroundingPrompts.system_prompt()
    model = genai.GenerativeModel(
        APP_CONFIG["generation_model"],
        system_instruction=system_instruction,
    )

    def generate_response(prompt: str) -> str:
        try:
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("Empty text response from generation model.")
            return text
        except Exception as e:
            raise RuntimeError(
                f"Generation failed using model '{APP_CONFIG['generation_model']}': {e}"
            ) from e

    return generate_response

def process_uploaded_documents() -> bool:
    """
    Process all staged files, not just the most recent uploader selection.
    """
    if not st.session_state.staged_files:
        st.error("Please upload and stage at least one document.")
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for file_name, file_bytes in st.session_state.staged_files.items():
                file_path = temp_path / file_name
                file_path.write_bytes(file_bytes)

            st.info("Step 1/4: Loading documents...")
            processor = DocumentProcessor(
                target_tokens=APP_CONFIG["target_tokens"],
                overlap_tokens=APP_CONFIG["overlap_tokens"],
            )

            docs = processor.load_documents_from_directory(str(temp_path))
            if not docs:
                st.error("No valid documents found.")
                return False

            for doc in docs:
                make_document_display_compatible(doc)

            st.session_state.documents = docs

            st.info("Step 2/4: Chunking documents...")
            chunks = processor.process_documents(docs)

            for chunk in chunks:
                make_chunk_backward_compatible(chunk)

            if not chunks:
                st.error("Documents were loaded, but no chunks were created.")
                return False

            st.session_state.chunks = chunks

            st.info("Step 3/4: Building vector store...")
            vector_store = VectorStoreFactory.create("in_memory")
            embedding_fn = setup_embedding_fn()

            progress_bar = st.progress(0.0)
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                chunk.embedding = embedding_fn(chunk.content)
                progress_bar.progress((i + 1) / total_chunks)

            vector_store.add_chunks(chunks)
            st.session_state.vector_store = vector_store

            st.info("Step 4/4: Initializing retriever and generator...")
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
                enabled=True,
            )

            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_fn=embedding_fn,
                dense_weight=APP_CONFIG["dense_weight"],
                sparse_weight=APP_CONFIG["sparse_weight"],
                max_chunks_per_document=APP_CONFIG["max_chunks_per_document"],
                reranker=reranker,
                rerank_top_n=15,
                rerank_weight=0.35,
            )
            st.session_state.retriever = retriever

            if reranker.available():
                st.info("Reranker enabled: cross-encoder/ms-marco-MiniLM-L6-v2")
            else:
                load_error = reranker.get_load_error()
                if load_error:
                    st.warning(f"Reranker could not be loaded. Falling back to hybrid retrieval only. Error: {load_error}")
                else:
                    st.warning("Reranker unavailable. Falling back to hybrid retrieval only.")

            llm_fn = setup_llm_fn()
            generator = RAGGenerator(
                retriever=retriever,
                llm_fn=llm_fn,
                min_context_score=APP_CONFIG["min_context_score"],
                top_k=APP_CONFIG["top_k"],
            )
            st.session_state.generator = generator

            st.session_state.processed = True
            st.session_state.processed_file_names = list(st.session_state.staged_files.keys())
            st.session_state.last_uploaded_names = list(st.session_state.staged_files.keys())

            st.success(f"RAG system is ready with {len(st.session_state.staged_files)} file(s).")
            return True

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return False


def run_query(query: str):
    if not st.session_state.generator or not st.session_state.retriever:
        raise RuntimeError("RAG pipeline is not initialized.")

    response = st.session_state.generator.generate(
        query=query,
        use_verification=False,
    )

    st.session_state.last_query = query
    st.session_state.last_response = response
    return response

def render_pipeline_stats():
    if not st.session_state.processed:
        return

    st.sidebar.markdown("### Pipeline Stats")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Documents", len(st.session_state.documents))
    with col2:
        st.metric("Chunks", len(st.session_state.chunks))

    total_tokens = sum(getattr(c, "token_count", 0) for c in st.session_state.chunks)
    st.sidebar.metric("Estimated Tokens", total_tokens)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Processed Files")
    for file_name in st.session_state.processed_file_names:
        st.sidebar.caption(f"• {file_name}")

st.title(" Intelligent Document Q&A System Using RAG")
st.caption("Upload documents, retrieve grounded evidence, and generate reliable answers.")

with st.sidebar:
    if st.button("Reset Pipeline", use_container_width=True):
        reset_pipeline(clear_staged=False)
        st.success("Pipeline reset. Staged files kept.")

    if st.button("Clear All Files + Pipeline", use_container_width=True):
        reset_pipeline(clear_staged=True)
        st.success("Cleared staged files and pipeline.")

    st.sidebar.markdown("---")
    render_pipeline_stats()


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Upload & Process", "❓ Ask Questions", "🔎 Retrieval Debug", "🕘 History"]
)


# -----------------------------------------------------------------------------
# TAB 1: UPLOAD & PROCESS
# -----------------------------------------------------------------------------
with tab1:
    st.header("Upload Documents")
    st.write("Supported formats: **TXT, MD, JSON, CSV, PDF, DOCX**")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "md", "json", "csv", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more documents",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Add Uploaded Files", use_container_width=True):
            if uploaded_files:
                added = stage_uploaded_files(uploaded_files)
                st.success(
                    f"Added/updated {added} file(s). "
                    f"Total staged: {len(st.session_state.staged_files)}"
                )
            else:
                st.warning("Please choose files first.")

    with col2:
        if st.button("Process All Staged Files", use_container_width=True):
            auto_added = stage_uploaded_files(uploaded_files) if uploaded_files else 0

            if st.session_state.staged_files:
                if auto_added > 0:
                    st.info(
                        f"Auto-added {auto_added} currently selected file(s) before processing."
                    )

                with st.spinner("Processing all staged documents..."):
                    success = process_uploaded_documents()
            else:
                st.warning("No files available. Upload files first.")

    with col3:
        if st.button("Clear Staged Files", use_container_width=True):
            reset_pipeline(clear_staged=True)
            st.success("Cleared staged files and pipeline.")

    st.markdown("---")

    st.subheader("Staged Files")
    if uploaded_files:
        st.caption(f"Currently selected in uploader: {len(uploaded_files)} file(s)")
        for f in uploaded_files:
            st.caption(f"Selected now → {f.name}")

    if st.session_state.staged_files:
        st.info(f"{len(st.session_state.staged_files)} file(s) currently staged")
        for i, file_name in enumerate(st.session_state.staged_files.keys(), start=1):
            st.caption(f"{i}. {file_name}")
    else:
        st.caption("No staged files yet.")

    if st.session_state.processed:
        st.markdown("---")
        st.subheader("Processed Document Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Loaded", len(st.session_state.documents))
        with col2:
            st.metric("Text Chunks", len(st.session_state.chunks))
        with col3:
            total_tokens = sum(getattr(c, "token_count", 0) for c in st.session_state.chunks)
            st.metric("Estimated Tokens", total_tokens)

        with st.expander("Document Details", expanded=False):
            for doc in st.session_state.documents:
                st.markdown(f"**{doc.filename}**")
                st.caption(
                    f"Type: {getattr(doc, 'doc_type', 'unknown')} | "
                    f"Chars: {len(getattr(doc, 'content', '') or '')}"
                )

        with st.expander("Chunk Preview", expanded=False):
            preview_count = min(5, len(st.session_state.chunks))
            for i in range(preview_count):
                chunk = st.session_state.chunks[i]
                st.markdown(f"**Chunk {i + 1}: {chunk.chunk_id}**")
                st.caption(
                    f"Source: {getattr(chunk, 'source_doc', 'Unknown')} | "
                    f"Tokens: {getattr(chunk, 'token_count', 0)}"
                )
                st.code(safe_preview(chunk.content, limit=1200))


# -----------------------------------------------------------------------------
# TAB 2: ASK QUESTIONS
# -----------------------------------------------------------------------------
with tab2:
    st.header("Ask Questions")

    if not st.session_state.processed:
        st.warning("Please upload, stage, and process documents first.")
    else:
        query = st.text_input(
            "Your question",
            placeholder="e.g. How do I troubleshoot WiFi issues?",
        )

        ask_button = st.button("Get Answer", use_container_width=True)

        if ask_button and query:
            with st.spinner("Searching and generating grounded answer..."):
                try:
                    response = run_query(query)

                    st.session_state.query_history.append(
                        {
                            "query": query,
                            "response": response,
                        }
                    )

                    st.markdown("---")
                    st.subheader("Answer")
                    st.markdown(response.get("answer", "No answer returned."))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{response.get('confidence', 0.0):.1%}")
                    with col2:
                        st.metric("Context Chunks", response.get("num_context_chunks", 0))
                    with col3:
                        st.metric("Sources", len(response.get("sources", [])))

                    if response.get("sources"):
                        st.subheader("Sources")
                        for source in response["sources"]:
                            st.caption(f"• {source}")

                    if response.get("retrieval_reasoning"):
                        with st.expander("Retrieval Reasoning", expanded=False):
                            st.write(response["retrieval_reasoning"])

                except Exception as e:
                    st.error(f"Error while answering query: {e}")

        elif ask_button:
            st.warning("Please enter a question.")


# -----------------------------------------------------------------------------
# TAB 3: RETRIEVAL DEBUG
# -----------------------------------------------------------------------------
with tab3:
    st.header("Retrieval Debug")

    if not st.session_state.last_response:
        st.info("Ask a question first to inspect retrieved evidence.")
    else:
        response = st.session_state.last_response
        chunk_scores = response.get("chunk_scores", [])

        st.markdown(f"**Last Query:** {st.session_state.last_query}")

        if chunk_scores:
            st.subheader("Retrieved Chunks and Scores")

            for item in chunk_scores:
                chunk = get_chunk_by_id(item["chunk_id"])

                st.markdown(
                    f"""
<div class="debug-box">
<b>Rank:</b> {item.get("rank")} <br>
<b>Chunk Number:</b> {item.get("chunk_number")} <br>
<b>Chunk ID:</b> {item.get("chunk_id")} <br>
<b>Source:</b> {item.get("source")} <br>
<b>Hybrid Score:</b> {item.get("score"):.4f} <br>
<b>Dense Score:</b> {item.get("dense_score"):.4f} <br>
<b>Sparse Score:</b> {item.get("sparse_score"):.4f}
</div>
""",
                    unsafe_allow_html=True,
                )

                if chunk:
                    with st.expander(f"View chunk text: {item.get('chunk_id')}", expanded=False):
                        st.code(safe_preview(chunk.content, limit=2500))
        else:
            st.warning("No chunk scores available for the last query.")


# -----------------------------------------------------------------------------
# TAB 4: HISTORY
# -----------------------------------------------------------------------------
with tab4:
    st.header("Query History")

    if not st.session_state.query_history:
        st.info("No queries yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.query_history), 1):
            q = item.get("query", "")
            r = item.get("response", {})

            with st.expander(
                f"Query {len(st.session_state.query_history) - i + 1}: {q[:70]}{'...' if len(q) > 70 else ''}"
            ):
                st.markdown("**Question:**")
                st.write(q)

                st.markdown("**Answer:**")
                st.write(r.get("answer", ""))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Confidence: {r.get('confidence', 0.0):.1%}")
                with col2:
                    st.caption(f"Chunks: {r.get('num_context_chunks', 0)}")
                with col3:
                    st.caption(f"Sources: {len(r.get('sources', []))}")

        history_text = ""
        for i, item in enumerate(st.session_state.query_history, 1):
            response = item.get("response", {})
            history_text += f"\n{'=' * 80}\n"
            history_text += f"Query {i}: {item.get('query', '')}\n"
            history_text += f"{'-' * 80}\n"
            history_text += f"Answer: {response.get('answer', '')}\n"
            history_text += f"Confidence: {response.get('confidence', 0.0):.1%}\n"
            history_text += f"Sources: {', '.join(response.get('sources', []))}\n"

        st.download_button(
            label="Download History",
            data=history_text,
            file_name="query_history.txt",
            mime="text/plain",
        )


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("RAG System")