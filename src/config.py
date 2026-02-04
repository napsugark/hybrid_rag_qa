"""
Configuration file for Advanced Hybrid RAG Application
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent  # Project root (one level up from src/)
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents_ro"
LOGS_DIR = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"

# ============================================================================
# QDRANT CONFIGURATION
# ============================================================================
# Qdrant Cloud is required - set these in your .env file
QDRANT_URL = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "hybrid_rag_documents_multilingual_large_dense_gpt4omini_v_11"
# QDRANT_COLLECTION = "hybrid_rag_documents_multilingual_large_dense_v7"

# QDRANT_EMBEDDING_DIM = 1024  # Snowflake arctic-embed-l dimension
QDRANT_EMBEDDING_DIM = 1024  # Snowflake arctic-embed-l-v2.0 dimension
# QDRANT_EMBEDDING_DIM = 768  # Snowflake arctic-embed-m-v2.0 dimension for the smaller/faster model

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11435")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = 600  # seconds
OLLAMA_KEEP_ALIVE = "30m"

# ============================================================================
# AZURE OPENAI CONFIGURATION
# ============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"
)  # Set your deployment name
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-06-01"
) 
GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 300,
}

# ============================================================================
# MODEL SELECTION - Switch between Ollama and Azure OpenAI
# ============================================================================
# Change this to switch models:
#   - "OLLAMA" for local Llama models via Ollama
#   - "AZURE_OPENAI" for Azure OpenAI (GPT-4o-mini, etc.)
MODEL_TO_USE = os.getenv("MODEL_TO_USE", "AZURE_OPENAI")  # or "OLLAMA"

if MODEL_TO_USE == "OLLAMA":
    LLM_TYPE = "OLLAMA"
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LLM_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11435")
    LLM_API_KEY = None
    LLM_API_VERSION = None
elif MODEL_TO_USE == "AZURE_OPENAI":
    LLM_TYPE = "AZURE_OPENAI"
    LLM_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    LLM_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
    LLM_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    LLM_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
else:
    raise ValueError(f"Unknown MODEL_TO_USE: {MODEL_TO_USE}. Use 'OLLAMA' or 'AZURE_OPENAI'")






# ============================================================================
# EMBEDDING MODELS
# ============================================================================
# Dense embedding model (semantic similarity) multilingual model
# DENSE_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l"
DENSE_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0" # larger/better
# DENSE_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-m-v2.0" - smaller/faster


DENSE_EMBEDDING_PREFIX = "Represent this sentence for searching relevant passages: "

# Sparse embedding model (keyword matching)
# English SPLADE model (not good for Romanian)
# SPARSE_EMBEDDING_MODEL = "prithvida/Splade_PP_en_v1"
# BM25 - language-agnostic, supports Romanian stopwords
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"


# Device for embeddings ('cpu', 'cuda', 'mps')
EMBEDDING_DEVICE = "cpu"

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
# Chunking strategy
CHUNK_SPLIT_BY = "word"  # 'word', 'sentence', 'page'
CHUNK_SIZE = 300  # words per chunk
CHUNK_OVERLAP = 50  # words overlap between chunks

# File types to process
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".md"]

# Production-grade document processing
USE_DOCUMENT_TYPE_DETECTION = True  # Detect invoice/contract/receipt types
USE_SEMANTIC_CHUNKING = True  # Use logical sections instead of fixed-size chunks
USE_BOILERPLATE_FILTER = True  # Remove legal/payment text before embedding

# Semantic chunking settings
SEMANTIC_CHUNK_MIN_SIZE = 100  # Minimum chunk size in characters
SEMANTIC_CHUNK_MAX_SIZE = 800  # Maximum chunk size in characters
SEMANTIC_CHUNK_OVERLAP = 50  # Overlap between chunks

# Boilerplate filtering settings
BOILERPLATE_MIN_SCORE = 3  # Minimum patterns to classify as boilerplate
SKIP_LEGAL_SECTIONS = True  # Skip legal disclaimers
SKIP_PAYMENT_SECTIONS = True  # Skip payment instructions

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
# Number of documents to retrieve
TOP_K = 5 

# Hybrid search weights (must sum to 1.0)
DENSE_WEIGHT = 0.4  # Weight for semantic search
SPARSE_WEIGHT = 0.6  # Weight for keyword search (higher for dates/invoices/structured data)

# Reranking
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 10  # Final number of documents after reranking

# ============================================================================
# METADATA ENRICHMENT
# ============================================================================
ENABLE_METADATA_EXTRACTION = True

# Metadata extraction strategy
# Options: "full_document", "first_page", "first_n_chars"
METADATA_EXTRACTION_STRATEGY = "full_document"  # Extract from entire document before chunking
METADATA_EXTRACTION_MAX_CHARS = 3000  # Only used if strategy = "first_n_chars"

# Metadata to extract
METADATA_FIELDS = [
    "company",  # ← FILTERABLE company name
    "client",  # ← FILTERABLE client name
    "year",  # ← FILTERABLE year (integer)
    "month",  # ← FILTERABLE month (integer)
    "day",  # ← FILTERABLE day (integer)
    "date",  # ← FILTERABLE full date
    "document_type",  # ← FILTERABLE doc type
    "invoice_number",  # ← FILTERABLE invoice ID
    "amount",  # ← FILTERABLE monetary value
    "currency",  # ← FILTERABLE currency
    "entities",  # General entities
    "topics",  # Topics
    "keywords",  # Keywords
    "language",  # Language
    "triples",  # Relationship triples
]

# ============================================================================
# SUMMARIZATION
# ============================================================================
ENABLE_SUMMARIZATION = True

# Smart summarization - only summarize valuable content
SUMMARIZE_ONLY_VALUABLE_CHUNKS = True  # Skip boilerplate chunks
SUMMARIZE_SECTION_TYPES = ["line_items", "scope", "terms", "unknown"]  # Which to summarize
SKIP_SUMMARY_SECTION_TYPES = ["legal", "payment", "header", "totals"]  # Which to skip

# Summary generation settings
SUMMARY_MAX_LENGTH = 150  # words
SUMMARY_STYLE = "concise"  # 'concise', 'detailed', 'bullet_points'

# ============================================================================
# PROMPTS (loaded from files)
# ============================================================================
PROMPTS_DIR = BASE_DIR / "prompts"
METADATA_EXTRACTION_PROMPT_FILE = PROMPTS_DIR / "metadata_extraction.txt"
QUERY_EXTRACTION_PROMPT_FILE = PROMPTS_DIR / "query_extraction.txt"
SUMMARIZATION_PROMPT_FILE = PROMPTS_DIR / "summarization.txt"
BOILERPLATE_DETECTION_PROMPT_FILE = PROMPTS_DIR / "boilerplate_detection.txt"
RAG_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "rag_system.txt"

# RAG User Prompt Variants - switch between different styles
RAG_USER_PROMPT_FILE = PROMPTS_DIR / "rag_user.txt"              # Default: comprehensive with structured reasoning
RAG_USER_PROMPT_CONCISE_FILE = PROMPTS_DIR / "rag_user_concise.txt"  # Concise: short 2-3 sentence answers
RAG_USER_PROMPT_STRUCTURED_FILE = PROMPTS_DIR / "rag_user_structured.txt"

# Active prompt to use (change this to switch styles)
ACTIVE_RAG_USER_PROMPT = RAG_USER_PROMPT_CONCISE_FILE  # Change to RAG_USER_PROMPT_CONCISE_FILE for short answers

# ============================================================================
# RAG PROMPT TEMPLATES (loaded from files)
# ============================================================================
# Note: RAG prompts are loaded from prompts/ directory
# To customize prompts, edit the files directly:
# - prompts/rag_system.txt - System instructions for the LLM
# - prompts/rag_user.txt - User prompt template (default)
# - prompts/rag_user_structured.txt - Alternative structured format

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# LANGFUSE CONFIGURATION
# ============================================================================
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

# ============================================================================
# PERFORMANCE
# ============================================================================
# Batch processing
INDEXING_BATCH_SIZE = 10
QUERY_BATCH_SIZE = 5

# Caching
ENABLE_EMBEDDING_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"

# Cache configuration
EMBEDDING_CACHE_SIZE = 1000  # Max number of cached query embeddings
RETRIEVAL_CACHE_SIZE = 500   # Max number of cached retrieval results
RESPONSE_CACHE_SIZE = 200    # Max number of cached LLM responses
EMBEDDING_CACHE_TTL = 3600   # 1 hour TTL for embeddings
RETRIEVAL_CACHE_TTL = 1800   # 30 minutes TTL for retrievals
RESPONSE_CACHE_TTL = 3600    # 1 hour TTL for responses

# ============================================================================
# RESILIENCE & ERROR HANDLING
# ============================================================================
# Circuit breaker thresholds
QDRANT_CIRCUIT_BREAKER_THRESHOLD = 5     # Open circuit after 5 failures
QDRANT_CIRCUIT_BREAKER_TIMEOUT = 60.0    # Recovery timeout in seconds
OLLAMA_CIRCUIT_BREAKER_THRESHOLD = 3     # Open circuit after 3 failures
OLLAMA_CIRCUIT_BREAKER_TIMEOUT = 30.0    # Recovery timeout in seconds

# Rate limiting
OLLAMA_MAX_CONCURRENT = 2      # Max concurrent Ollama requests (GPU protection)
OLLAMA_MAX_PER_MINUTE = 60     # Max Ollama requests per minute
QDRANT_MAX_CONCURRENT = 10     # Max concurrent Qdrant operations

# Retry configuration
MAX_RETRY_ATTEMPTS = 3         # Max retry attempts for failed operations
INITIAL_RETRY_DELAY = 1.0      # Initial delay before retry in seconds
MAX_RETRY_DELAY = 60.0         # Max retry delay in seconds

# ============================================================================
# DISPLAY OPTIONS
# ============================================================================
DISPLAY_CONFIG = {
    "show_scores": True,
    "show_metadata": True,
    "show_summaries": True,
    "show_sources": True,
    "max_content_preview": 800,  # characters
    "color_scheme": "default",  # for rich terminal output
}
