# 🤖 AI-Powered Document Q&A System (RAG)

A complete **Retrieval-Augmented Generation (RAG)** system built using **Streamlit + Google Gemini AI**.

Upload documents → Process → Ask questions → Get grounded answers with sources + evaluation metrics.

---

## 🎥 Demo



---

## 🚀 Features

* 📄 Multi-document upload (TXT, PDF, DOCX, CSV, JSON, MD)
* 🔍 Hybrid Retrieval (Semantic + Keyword search)
* 🤖 Gemini-based grounded answer generation
* 📊 Confidence score + sources
* 🧠 Retrieval Debug tab (inspect chunks + scores)
* 🕘 Query History tracking
* 🧪 Full RAG Evaluation pipeline with advanced metrics

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add API Key

Create `.env` file:

```bash
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## 📚 How to Use

### 1. Upload & Process

* Upload one or more documents
* Click **Process All Staged Files**

System performs:

* document loading
* chunking
* embedding
* vector indexing

---

### 2. Ask Questions

* Enter your query
* System retrieves relevant context
* Generates answer using LLM

Output includes:

* Answer
* Confidence score
* Sources used

---

### 3. Retrieval Debug

* View retrieved chunks
* Inspect:

  * hybrid score
  * dense score
  * sparse score
* Understand model reasoning

---

### 4. History

* View previous queries
* Expand answers
* Download history

---

## 📁 Project Structure

```
├── app.py                  # Streamlit UI (tabs: Upload, Ask, Debug, History)
├── main.py                 # CLI pipeline
├── scripts/
│   └── evaluate_rag.py     # RAG evaluation (advanced metrics)
├── src/rag/
│   ├── document_processing/
│   ├── vector_store/
│   ├── retrieval/
│   └── generation/
├── data/                   # Documents + QA dataset
├── .env.example
└── requirements.txt
```

---

## 🧠 RAG Pipeline

```
Documents → Chunking → Embedding → Vector Store
                                ↓
Query → Hybrid Retrieval → Context → LLM → Answer
```

---

## 🧪 Evaluation

Evaluate your RAG system using a QA dataset.

### 📌 Dataset Format

Create CSV:

```csv
question,answer
What is the capacity?,12 cups
What WiFi band is supported?,2.4GHz
```

---

### ▶️ Run Evaluation

```bash
python -m scripts.evaluate_rag \
  --docs_dir data \
  --qa_csv data/evaluation_questions.csv
```

---

### 📊 Output Files

* `evaluation_results.csv` → per-question results
* `evaluation_summary.json` → overall metrics

---

## 📈 Evaluation Metrics

The system computes:

### Core Metrics

* Exact Match (EM)
* Token F1 Score
* Jaccard Similarity

### Advanced Metrics

* BLEU-1
* ROUGE-L
* Sequence Similarity

### Performance Metrics

* Pass@F1 (0.25 / 0.50 / 0.75)
* Confidence Score
* Context Chunk Usage
* Source Count

### Analysis Metrics

* Answer Length Ratio
* Insufficient Information Rate
* Good / Partial / Poor classification

---

## 📌 Notes

* Works best with **text-based PDFs**
* Scanned PDFs require OCR (not included)
* Internet required for Gemini API
* Evaluation requires QA dataset aligned with documents

---

## 🎯 Future Improvements

* Local embeddings (offline support)
* Vector DB integration (Milvus / FAISS)
* Chat-style UI (LLM interface)
* Advanced evaluation (faithfulness, retrieval recall)
* Multi-modal support (images + text)

---

## ⭐ Summary

This project demonstrates a full **end-to-end RAG system**:

* Document ingestion
* Hybrid retrieval
* Grounded generation
* Debugging tools
* Evaluation framework

Suitable for:

* AI/ML research projects
* RAG system experimentation
* Production-ready prototypes

---

## 📜 License

MIT License
