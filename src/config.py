"""
Configuration for Advanced Hybrid RAG Application

Uses Pydantic BaseSettings for typed, validated, env-aware configuration.
All existing ``config.SOME_CONSTANT`` access patterns are preserved via a
module-level ``__getattr__`` that proxies to the singleton ``settings`` instance.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# Path constants (derived from file location, not from env)
# ============================================================================
BASE_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DOCUMENTS_DIR: Path = DATA_DIR / "documents_ro"
LOGS_DIR: Path = BASE_DIR / "logs"
PROMPTS_DIR: Path = BASE_DIR / "prompts"
CACHE_DIR: Path = BASE_DIR / ".cache"

# ============================================================================
# Prompt file paths (numbered for pipeline-order clarity)
# ============================================================================
BOILERPLATE_DETECTION_PROMPT_FILE: Path = PROMPTS_DIR / "01_boilerplate_detection.txt"
METADATA_EXTRACTION_PROMPT_FILE: Path = PROMPTS_DIR / "02_metadata_extraction.txt"
SUMMARIZATION_PROMPT_FILE: Path = PROMPTS_DIR / "03_summarization.txt"
QUERY_EXTRACTION_PROMPT_FILE: Path = PROMPTS_DIR / "04_query_extraction.txt"
RAG_SYSTEM_PROMPT_FILE: Path = PROMPTS_DIR / "05_rag_system.txt"
RAG_USER_PROMPT_FILE: Path = PROMPTS_DIR / "06_rag_user.txt"
RAG_USER_PROMPT_CONCISE_FILE: Path = PROMPTS_DIR / "07_rag_user_concise.txt"
RAG_USER_PROMPT_STRUCTURED_FILE: Path = PROMPTS_DIR / "08_rag_user_structured.txt"

# Active prompt variant (change to switch answer style)
ACTIVE_RAG_USER_PROMPT: Path = RAG_USER_PROMPT_CONCISE_FILE


# ============================================================================
# Settings class
# ============================================================================
class Settings(BaseSettings):
    """Typed, validated configuration loaded from .env + environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Qdrant ----------------------------------------------------------
    QDRANT_URL: Optional[str] = Field(
        default=None,
        validation_alias="QDRANT_ENDPOINT",
        description="Qdrant Cloud endpoint URL",
    )
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "rag_hybrid_arcticl2_v11"
    QDRANT_EMBEDDING_DIM: int = 1024  # Snowflake arctic-embed-l-v2.0

    # ---- Ollama ----------------------------------------------------------
    OLLAMA_URL: str = "http://127.0.0.1:11435"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT: int = 600
    OLLAMA_KEEP_ALIVE: str = "30m"

    # ---- Azure OpenAI ----------------------------------------------------
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-06-01"

    # ---- Generation ------------------------------------------------------
    GENERATION_CONFIG: Dict[str, Any] = {"temperature": 0.1, "max_tokens": 300}

    # ---- Model selection -------------------------------------------------
    MODEL_TO_USE: str = "AZURE_OPENAI"  # "OLLAMA" or "AZURE_OPENAI"

    # Derived — resolved by validator
    LLM_TYPE: str = ""
    LLM_MODEL: str = ""
    LLM_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_API_VERSION: Optional[str] = None

    @model_validator(mode="after")
    def _resolve_llm_settings(self) -> "Settings":
        if self.MODEL_TO_USE == "OLLAMA":
            self.LLM_TYPE = "OLLAMA"
            self.LLM_MODEL = self.OLLAMA_MODEL
            self.LLM_URL = self.OLLAMA_URL
            self.LLM_API_KEY = None
            self.LLM_API_VERSION = None
        elif self.MODEL_TO_USE == "AZURE_OPENAI":
            self.LLM_TYPE = "AZURE_OPENAI"
            self.LLM_MODEL = self.AZURE_OPENAI_DEPLOYMENT
            self.LLM_URL = self.AZURE_OPENAI_ENDPOINT
            self.LLM_API_KEY = self.AZURE_OPENAI_API_KEY
            self.LLM_API_VERSION = self.AZURE_OPENAI_API_VERSION
        else:
            raise ValueError(
                f"Unknown MODEL_TO_USE: {self.MODEL_TO_USE}. "
                "Use 'OLLAMA' or 'AZURE_OPENAI'"
            )
        return self

    # ---- Embedding models ------------------------------------------------
    DENSE_EMBEDDING_MODEL: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    DENSE_EMBEDDING_PREFIX: str = (
        "Represent this sentence for searching relevant passages: "
    )
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    EMBEDDING_DEVICE: str = "cpu"  # 'cpu', 'cuda', 'mps'

    # ---- Document processing ---------------------------------------------
    CHUNK_SPLIT_BY: str = "word"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".txt", ".pdf", ".md"]

    USE_DOCUMENT_TYPE_DETECTION: bool = True
    USE_SEMANTIC_CHUNKING: bool = True
    USE_BOILERPLATE_FILTER: bool = True

    SEMANTIC_CHUNK_MIN_SIZE: int = 100
    SEMANTIC_CHUNK_MAX_SIZE: int = 800
    SEMANTIC_CHUNK_OVERLAP: int = 50

    BOILERPLATE_MIN_SCORE: int = 3
    SKIP_LEGAL_SECTIONS: bool = True
    SKIP_PAYMENT_SECTIONS: bool = True

    # ---- Retrieval -------------------------------------------------------
    TOP_K: int = 5
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANKER_TOP_K: int = 10

    # Minimum reranker score to include a document in Mayan /query results.
    # Cross-encoder scores are roughly calibrated 0-1; 0.3 is a reasonable
    # threshold to drop clearly irrelevant retrievals.
    MAYAN_RESULTS_MIN_SCORE: float = 0.3

    # ---- Metadata enrichment ---------------------------------------------
    ENABLE_METADATA_EXTRACTION: bool = True
    METADATA_EXTRACTION_STRATEGY: str = "full_document"
    METADATA_EXTRACTION_MAX_CHARS: int = 3000
    METADATA_FIELDS: List[str] = [
        "company",
        "client",
        "year",
        "month",
        "day",
        "date",
        "document_type",
        "invoice_number",
        "amount",
        "currency",
        "entities",
        "topics",
        "keywords",
        "language",
        "triples",
    ]

    # ---- Summarization ---------------------------------------------------
    ENABLE_SUMMARIZATION: bool = True
    SUMMARIZE_ONLY_VALUABLE_CHUNKS: bool = True
    SUMMARIZE_SECTION_TYPES: List[str] = [
        "line_items", "scope", "terms", "unknown",
    ]
    SKIP_SUMMARY_SECTION_TYPES: List[str] = [
        "legal", "payment", "header", "totals",
    ]
    SUMMARY_MAX_LENGTH: int = 150
    SUMMARY_STYLE: str = "concise"

    # ---- Logging ---------------------------------------------------------
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # ---- Langfuse --------------------------------------------------------
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    @property
    def LANGFUSE_ENABLED(self) -> bool:  # noqa: N802
        return bool(self.LANGFUSE_PUBLIC_KEY)

    # ---- Mayan EDMS ------------------------------------------------------
    MAYAN_URL: str = ""
    MAYAN_API_TOKEN: str = ""

    # ---- Performance / caching -------------------------------------------
    INDEXING_BATCH_SIZE: int = 10
    QUERY_BATCH_SIZE: int = 5
    ENABLE_EMBEDDING_CACHE: bool = True

    EMBEDDING_CACHE_SIZE: int = 1000
    RETRIEVAL_CACHE_SIZE: int = 500
    RESPONSE_CACHE_SIZE: int = 200
    EMBEDDING_CACHE_TTL: int = 3600
    RETRIEVAL_CACHE_TTL: int = 1800
    RESPONSE_CACHE_TTL: int = 3600

    # ---- Resilience ------------------------------------------------------
    QDRANT_CIRCUIT_BREAKER_THRESHOLD: int = 5
    QDRANT_CIRCUIT_BREAKER_TIMEOUT: float = 60.0
    OLLAMA_CIRCUIT_BREAKER_THRESHOLD: int = 3
    OLLAMA_CIRCUIT_BREAKER_TIMEOUT: float = 30.0

    OLLAMA_MAX_CONCURRENT: int = 2
    OLLAMA_MAX_PER_MINUTE: int = 60
    QDRANT_MAX_CONCURRENT: int = 10

    MAX_RETRY_ATTEMPTS: int = 3
    INITIAL_RETRY_DELAY: float = 1.0
    MAX_RETRY_DELAY: float = 60.0

    # ---- Display ---------------------------------------------------------
    DISPLAY_CONFIG: Dict[str, Any] = {
        "show_scores": True,
        "show_metadata": True,
        "show_summaries": True,
        "show_sources": True,
        "max_content_preview": 800,
        "color_scheme": "default",
    }


# ============================================================================
# Singleton instance
# ============================================================================
settings = Settings()


# ============================================================================
# Backward-compatible module-level access:  config.QDRANT_URL  etc.
# ============================================================================
def __getattr__(name: str):
    """Proxy attribute lookups to the settings singleton."""
    try:
        return getattr(settings, name)
    except AttributeError:
        raise AttributeError(f"module 'config' has no attribute {name}") from None
