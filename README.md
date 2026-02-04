# Advanced Hybrid RAG Application (v2)

A comprehensive RAG application with hybrid search, document summarization, metadata enrichment, and local LLM support.

## Features

- **✨ Hybrid Search**: Combines sparse (BM25-style) and dense (semantic) embeddings for optimal retrieval
- **📝 Document Summarization**: Automatic generation of document summaries for quick insights
- **🏷️ Metadata Enrichment**: Automated extraction of entities, topics, and structured metadata
- **🤖 Local LLM**: Uses Ollama for private, offline inference
- **☁️ Qdrant Storage**: Persistent cloud-based vector storage with hybrid capabilities
- **📊 Rich Statistics**: Document analytics and search quality metrics

## Architecture

### Indexing Pipeline
```
Documents → Splitter → Metadata Enricher → Summarizer → Dense Embedder → Sparse Embedder → Qdrant
```

### Query Pipeline
```
Query → Dense Embedder → Sparse Embedder → Hybrid Retriever → Reranker → Prompt Builder → LLM Generator
```

## Requirements

- Python 3.9+
- Qdrant Cloud account
- Ollama installed and running locally
- At least 8GB RAM recommended

## Setup

### Option 1: Using Poetry (Recommended)

1. Install Poetry if you haven't already:
```bash
pip install poetry
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Option 2: Using pip

1. Install dependencies:
```bash
pip install -r requirements.txt
```
**With Poetry:**
```bash
poetry run index-docs
# or
poetry run python index_documents.py
```

**With pip:**

2. Configure environment variables in `.env`:
```
QDRANT_ENDPOINT=your_qdrant_url
QDRANT_API_KEY=your_api_key
OLLAMA_URL=http://127.0.0.1:11435
OLLAMA_MODEL=llama3.1:8b
```

3. Start Ollama:
```bash
ollama serve
ollama pull llama3.1:8b
```

## Usage

### 1. Index Documents

**With Poetry:**
```bash
poetry run python scripts/index_documents.py
```

**Direct:**
```bash
python scripts/index_documents.py
```

This will:
- Load documents from `data/documents_ro/`
- Extract metadata and entities
- Generate summaries for each chunk
- Create both dense and sparse embeddings
- Store everything in Qdrant

### 2. Query Documents

**With Poetry:**
```bash
poetry run python query_app.py
# or provide query directly
poetry run python query_app.py "facturi sub 1000 lei"
```

**Direct:**
```bash
python query_app.py
```

Interactive query interface with:
- Hybrid search (configurable weights)
- Context from summaries
- Metadata filtering
- Source citation

### 3. Run Evaluation

**With Poetry:**
```bash
poetry run python evaluation/run_evaluation.py
```

This will:
- Run predefined test queries
- Score metadata extraction, filter usage, and retrieval quality
- Track results in Langfuse
- Save results to `evaluation/results/`

### 4. Manage Qdrant Indexes

**Create payload indexes:**
```bash
poetry run python scripts/create_payload_indexes.py
```

**Recreate entire collection:**
```bash
poetry run python scripts/recreate_collection.py
```

**Check indexed data:**
```bash
poetry run python scripts/test_check_data.py
```

## Configuration

Edit `config.py` to customize:
- Embedding models
- Chunk sizes
- Retrieval parameters
- Summarization settings
- Metadata extraction prompts

## Metadata Enrichment

Automatically extracted metadata includes:
- **Entities**: Companies, people, locations, dates
- **Topics**: Main subjects and themes
- **Document Type**: Contracts, reports, emails, etc.
- **Language**: Detected language
- **Keywords**: Important terms

## Summarization

Each document chunk gets:
- **Brief Summary**: 1-2 sentences
- **Key Points**: Bullet points of main ideas
- **Context**: Surrounding document context

## Hybrid Search Explained

Combines two retrieval methods:
1. **Sparse Embeddings**: Keyword/term matching (like BM25)
2. **Dense Embeddings**: Semantic similarity

You can adjust the balance between them for your use case:
- More sparse weight → Better for exact matches
- More dense weight → Better for conceptual queries

## Project Structure

```
app_v2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Poetry configuration
├── query_app.py                # Interactive query interface (entry point)
│
├── src/                        # Core application code
│   ├── __init__.py
│   ├── app.py                  # Main RAG application class
│   ├── config.py               # Configuration settings
│   ├── langfuse_tracker.py     # Langfuse observability
│   └── utils.py                # Utility functions
│
├── components/                 # Custom Haystack components
│   ├── README.md
│   ├── semantic_chunker.py     # Semantic document chunking
│   ├── metadata_enricher.py    # Metadata extraction
│   ├── document_type_detector.py  # Document type detection
│   ├── query_metadata_extractor.py  # Query filter extraction
│   ├── boilerplate_filter.py   # Boilerplate detection
│   └── summarizer.py           # Document summarization
│
├── scripts/                    # Utility scripts
│   ├── README.md               # Scripts documentation
│   ├── index_documents.py      # Index documents to Qdrant
│   ├── create_payload_indexes.py  # Create metadata indexes
│   ├── recreate_collection.py  # Recreate Qdrant collection
│   ├── cleanup_old_indexes.py  # Clean up old indexes
│   ├── debug_section_type.py   # Debug section_type filter
│   └── test_check_data.py      # Quick data check
│
├── evaluation/                 # Evaluation system
│   ├── README.md               # Evaluation documentation
│   ├── evaluation_dataset.py   # Test queries and expected outputs
│   ├── run_evaluation.py       # Automated evaluation runner
│   └── results/                # Evaluation results
│
├── verify_check/               # Testing and debugging scripts
│   ├── test_structured_qa.py
│   ├── test_hybrid_retrieval.py
│   └── ... (other test scripts)
│
├── docs/                       # Documentation
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── LANGFUSE_SETUP.md
│   └── ... (other docs)
│
├── prompts/                    # LLM prompts
│   ├── rag_system.txt
│   ├── metadata_extraction.txt
│   ├── query_extraction.txt
│   └── ... (other prompts)
│
├── data/                       # Data directory
│   ├── documents_ro/           # Documents to index
│   └── documents_ro_v1/
│
├── logs/                       # Application logs
└── notebooks/                  # Jupyter notebooks
    ├── hybrid_retrieval_bm42.ipynb
    └── ... (other notebooks)
```

## Performance Tips

1. **GPU Acceleration**: Set `device='cuda'` for embedders if you have a GPU
2. **Batch Processing**: Increase batch sizes for faster indexing
3. **Ollama Performance**: Use quantized models (e.g., `llama3.1:8b-q4_0`) for faster inference
4. **Qdrant Optimization**: Enable compression in Qdrant for storage efficiency

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Qdrant Connection Issues
- Verify credentials in `.env`
- Check network connectivity
- Confirm collection exists

### Out of Memory
- Reduce `batch_size` in config
- Use smaller embedding models
- Process documents in smaller batches

## License

MIT License - See main repository for details
