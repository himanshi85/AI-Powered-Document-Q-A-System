import os
import re
import json
import math
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

from src.rag.document_processing.processor import DocumentProcessor
from src.rag.vector_store import VectorStoreFactory
from src.rag.retrieval import HybridRetriever
from src.rag.generation import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts


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

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}

    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_counts:
        if t in gold_counts:
            common += min(pred_counts[t], gold_counts[t])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def jaccard_similarity(pred: str, gold: str) -> float:
    pred_set = set(tokenize(pred))
    gold_set = set(tokenize(gold))

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return intersection / union if union else 0.0


def sequence_similarity(pred: str, gold: str) -> float:
    return SequenceMatcher(None, normalize_text(pred), normalize_text(gold)).ratio()


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts: Dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    match = 0
    used: Dict[str, int] = {}
    for token in pred_tokens:
        used[token] = used.get(token, 0) + 1
        if used[token] <= gold_counts.get(token, 0):
            match += 1

    precision = match / len(pred_tokens)
    if precision == 0:
        return 0.0

    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = math.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1))

    return bp * precision


def rouge_l(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gold_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def strip_answer_prefix(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    answer_lines = []
    in_answer = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("answer:"):
            in_answer = True
            answer_lines.append(stripped[len("answer:"):].strip())
            continue

        if stripped.lower().startswith("evidence used:"):
            break

        if in_answer:
            answer_lines.append(line)

    cleaned = "\n".join(answer_lines).strip()
    return cleaned if cleaned else text.strip()


def contains_insufficient_marker(text: str) -> int:
    lower_text = (text or "").lower()
    markers = [
        "not enough information",
        "insufficient information",
        "not available in the provided documents",
        "cannot determine from the provided context",
        "missing from the provided documents",
        "provided documents do not contain enough information",
    ]
    return int(any(marker in lower_text for marker in markers))


def pass_at_threshold(value: float, threshold: float) -> int:
    return int(value >= threshold)

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

def build_rag_pipeline(docs_dir: str):
    processor = DocumentProcessor(
        target_tokens=CONFIG["target_tokens"],
        overlap_tokens=CONFIG["overlap_tokens"],
    )

    print("[1/4] Loading documents...")
    docs = processor.load_documents_from_directory(docs_dir)
    if not docs:
        raise ValueError(f"No valid documents found in: {docs_dir}")

    for doc in docs:
        make_document_display_compatible(doc)

    print(f"Loaded {len(docs)} documents")

    print("[2/4] Chunking documents...")
    chunks = processor.process_documents(docs)
    if not chunks:
        raise ValueError("Documents loaded, but no chunks were created.")

    for chunk in chunks:
        make_chunk_backward_compatible(chunk)

    print(f"Created {len(chunks)} chunks")

    print("[3/4] Embedding chunks...")
    embedding_fn = setup_embedding_fn()
    for i, chunk in enumerate(chunks, start=1):
        chunk.embedding = embedding_fn(chunk.content)
        if i % 10 == 0 or i == len(chunks):
            print(f"Embedded {i}/{len(chunks)} chunks")

    vector_store = VectorStoreFactory.create("in_memory")
    vector_store.add_chunks(chunks)

    print("[4/4] Initializing retriever and generator...")
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
        dense_weight=CONFIG["dense_weight"],
        sparse_weight=CONFIG["sparse_weight"],
        max_chunks_per_document=CONFIG["max_chunks_per_document"],
    )

    llm_fn = setup_llm_fn()
    generator = RAGGenerator(
        retriever=retriever,
        llm_fn=llm_fn,
        min_context_score=CONFIG["min_context_score"],
        top_k=CONFIG["top_k"],
    )

    return {
        "documents": docs,
        "chunks": chunks,
        "vector_store": vector_store,
        "retriever": retriever,
        "generator": generator,
    }

def load_qa_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"question", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"QA CSV missing required columns: {missing}")

    return df

def evaluate(generator, qa_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    records = []

    total = len(qa_df)
    print(f"Evaluating {total} questions...")

    for i, row in qa_df.iterrows():
        question = str(row["question"])
        gold_answer = str(row["answer"])

        result = generator.generate(question, use_verification=False)
        raw_generated = result.get("answer", "")
        generated_answer = strip_answer_prefix(raw_generated)

        em = exact_match(generated_answer, gold_answer)
        f1 = token_f1(generated_answer, gold_answer)
        jaccard = jaccard_similarity(generated_answer, gold_answer)
        seq_sim = sequence_similarity(generated_answer, gold_answer)
        bleu_1 = bleu1(generated_answer, gold_answer)
        rouge_l_score = rouge_l(generated_answer, gold_answer)

        insufficient_flag = contains_insufficient_marker(raw_generated)
        gold_len = len(tokenize(gold_answer))
        pred_len = len(tokenize(generated_answer))
        source_list = result.get("sources", [])
        source_count = len(source_list)
        context_chunks = result.get("num_context_chunks", 0)
        confidence = result.get("confidence", 0.0)

        records.append(
            {
                "row_id": i,
                "question": question,
                "gold_answer": gold_answer,
                "generated_raw": raw_generated,
                "generated_answer": generated_answer,
                "exact_match": em,
                "token_f1": f1,
                "jaccard": jaccard,
                "sequence_similarity": seq_sim,
                "bleu1": bleu_1,
                "rouge_l": rouge_l_score,
                "pass_f1_0_25": pass_at_threshold(f1, 0.25),
                "pass_f1_0_50": pass_at_threshold(f1, 0.50),
                "pass_f1_0_75": pass_at_threshold(f1, 0.75),
                "insufficient_flag": insufficient_flag,
                "confidence": confidence,
                "num_context_chunks": context_chunks,
                "source_count": source_count,
                "gold_answer_tokens": gold_len,
                "generated_answer_tokens": pred_len,
                "length_ratio": (pred_len / gold_len) if gold_len > 0 else None,
                "sources": json.dumps(source_list, ensure_ascii=False),
                "retrieval_reasoning": result.get("retrieval_reasoning", ""),
                "status": (
                    "good"
                    if f1 >= 0.75
                    else "partial"
                    if f1 >= 0.25
                    else "poor"
                ),
            }
        )

        print(
            f"[{i + 1}/{total}] "
            f"EM={em} F1={f1:.3f} Jaccard={jaccard:.3f} "
            f"BLEU1={bleu_1:.3f} ROUGE-L={rouge_l_score:.3f}"
        )

    results_df = pd.DataFrame(records)

    summary = {
        "num_questions": int(len(results_df)),
        "exact_match_count": int(results_df["exact_match"].sum()),
        "avg_exact_match": float(results_df["exact_match"].mean()),
        "avg_token_f1": float(results_df["token_f1"].mean()),
        "median_token_f1": float(results_df["token_f1"].median()),
        "min_token_f1": float(results_df["token_f1"].min()),
        "max_token_f1": float(results_df["token_f1"].max()),
        "avg_jaccard": float(results_df["jaccard"].mean()),
        "avg_sequence_similarity": float(results_df["sequence_similarity"].mean()),
        "avg_bleu1": float(results_df["bleu1"].mean()),
        "avg_rouge_l": float(results_df["rouge_l"].mean()),
        "pass_f1_0_25_rate": float(results_df["pass_f1_0_25"].mean()),
        "pass_f1_0_50_rate": float(results_df["pass_f1_0_50"].mean()),
        "pass_f1_0_75_rate": float(results_df["pass_f1_0_75"].mean()),
        "avg_confidence": float(results_df["confidence"].mean()),
        "median_confidence": float(results_df["confidence"].median()),
        "avg_num_context_chunks": float(results_df["num_context_chunks"].mean()),
        "avg_source_count": float(results_df["source_count"].mean()),
        "insufficient_rate": float(results_df["insufficient_flag"].mean()),
        "avg_gold_answer_tokens": float(results_df["gold_answer_tokens"].mean()),
        "avg_generated_answer_tokens": float(results_df["generated_answer_tokens"].mean()),
        "good_count": int((results_df["status"] == "good").sum()),
        "partial_count": int((results_df["status"] == "partial").sum()),
        "poor_count": int((results_df["status"] == "poor").sum()),
    }

    return results_df, summary


def print_summary(summary: Dict):
    print("\nEvaluation complete")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, required=True, help="Folder containing source documents")
    parser.add_argument("--qa_csv", type=str, required=True, help="CSV file with question,answer columns")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Where to save detailed results")
    parser.add_argument("--summary_json", type=str, default="evaluation_summary.json", help="Where to save summary metrics")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    genai.configure(api_key=api_key)

    pipeline = build_rag_pipeline(args.docs_dir)
    qa_df = load_qa_dataset(args.qa_csv)

    results_df, summary = evaluate(pipeline["generator"], qa_df)

    results_df.to_csv(args.output_csv, index=False)

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved detailed results to: {args.output_csv}")
    print(f"Saved summary to: {args.summary_json}")
    print_summary(summary)


if __name__ == "__main__":
    main()