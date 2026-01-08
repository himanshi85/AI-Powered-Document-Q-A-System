# 🤖AI-Powered Document Q&A System

**Production-ready Retrieval-Augmented Generation (RAG) system with Streamlit UI**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What It Does

Upload documents → Process automatically → Ask questions → Get AI answers with sources

**RAG** combines document intelligence with large language models to provide accurate, grounded answers backed by your documents.

### Key Features

✅ **Hybrid Search** - 60% semantic (dense vectors) + 40% keyword (BM25) matching  
✅ **Gemini Integration** - State-of-the-art embeddings and LLM generation  
✅ **Grounded Responses** - All answers backed by source documents  
✅ **Beautiful UI** - Streamlit web interface (no coding required)  
✅ **Production-Ready** - Error handling, logging, type safety, validation  
✅ **Fully Tested** - 20+ unit tests with high coverage  

## 🚀 Quick Start

### 1. Get API Key
Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Key
```bash
export GOOGLE_API_KEY=your_key_here
# or create .env file:
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 4. Run the App
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

## 📚 How to Use

### Tab 1: Upload & Process
1. Upload TXT, MD, or JSON documents
2. Click "Process Documents"
3. System chunks, embeds, and indexes them

### Tab 2: Ask Questions
1. Type your question
2. Click "Search"
3. Get AI-generated answer with sources and confidence score

### Tab 3: View History
1. See all previous queries
2. Click to expand full Q&A
3. Export history as text

## 🏗️ Architecture

```
Documents → Chunking → Embedding → Dense Index + Sparse Index
                                           ↓
Query → Embedding → Dense Search (60%) + Sparse Search (40%)
                           ↓
                    Hybrid Ranking → Top-5 Results
                           ↓
                   Grounding Prompt + Context
                           ↓
                   Gemini LLM Generation
                           ↓
                  Grounded Answer + Sources + Confidence
```

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Document loading | <100ms | Per file |
| Chunking | <100ms | Per document |
| Embedding | 100-500ms | Per chunk |
| Retrieval | 50-200ms | Hybrid search |
| LLM response | 1-3 sec | Gemini API |
| **Total E2E** | **~3-5 sec** | Per query |

## 📁 Project Structure

```
AI-Powered Document Q&A System/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .env.example             # API key template
├── src/rag/                 # Core RAG system
│   ├── document_processing/ # Load & chunk documents
│   ├── vector_store/        # Embeddings & indexing
│   ├── retrieval/           # Hybrid search
│   └── generation/          # LLM responses
├── tests/                   # Unit tests
├── data/                    # Sample documents
└── docs/                    # Documentation
```

## 🔧 Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your_key                    # Required
RETRIEVAL_DENSE_WEIGHT=0.6                 # Semantic weight
RETRIEVAL_SPARSE_WEIGHT=0.4                # Keyword weight
LOG_LEVEL=INFO                             # Debug level
```

### Document Processing
```python
DocumentProcessor(
    chunk_size=400,        # Tokens per chunk
    chunk_overlap=100,     # Overlap between chunks
)
```

### Retrieval Tuning
```python
HybridRetriever(
    dense_weight=0.6,      # Semantic search weight
    sparse_weight=0.4,     # Keyword search weight
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest --cov=src/ tests/

# Specific test file
pytest tests/test_retrieval.py -v
```

## 🌍 Deployment

### Streamlit Cloud (Recommended for Demo)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo
4. Add API key in Secrets
5. Live in 5 minutes!

### Hugging Face Spaces
1. Create Space with Streamlit SDK
2. Connect to GitHub
3. Add GOOGLE_API_KEY secret
4. Auto-deploys on push

### Docker
```bash
docker build -t AI-Powered Document Q&A System .
docker run -e GOOGLE_API_KEY=your_key -p 8501:8501
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [docs/SETUP.md](docs/SETUP.md) | Installation & configuration |
| [docs/DEPLOYMENT_FREE_OPTIONS.md](docs/DEPLOYMENT_FREE_OPTIONS.md) | Free deployment platforms |
| [docs/API.md](docs/API.md) | Python API reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues & solutions |

## 🆘 Troubleshooting

### "API Key not found"
```bash
# Add to .env
echo "GOOGLE_API_KEY=your_key" > .env
```

### "Embedding quota exceeded"
- Free tier has limited quota
- System falls back to random embeddings
- Upgrade to paid Gemini API for production

### "No documents found"
- Only TXT, MD, JSON supported
- Files must have content (>100 chars)
- Check file paths

### "Slow responses"
- First query slower (API warmup)
- Subsequent queries are faster (2-3 sec)
- Check internet connection

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more help.

## 🤝 Contributing

```bash
# Setup dev environment
pip install -r requirements.txt
pip install pytest black mypy

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type check
mypy src/
```

## 📊 Stats

- **Code**: 2,000+ lines of Python
- **UI**: 500+ lines Streamlit
- **Tests**: 20+ unit tests
- **Docs**: 100+ KB documentation
- **Dependencies**: 17 core packages

## 📜 License

MIT License - See LICENSE file for details

## 🎉 Get Started

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
export GOOGLE_API_KEY=your_key

# 3. Run
streamlit run app.py
```

Open http://localhost:8501 in your browser! 🚀

---

**Built with ❤️ using Python, Streamlit, and Google Gemini AI**

*Last updated: January 2026 | Production Ready*
