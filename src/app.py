"""
Advanced Hybrid RAG Application
Combines sparse + dense embeddings, metadata enrichment, summarization, and local LLM
"""

import logging
from pathlib import Path
from datetime import datetime
import time
from typing import List, Optional, Dict, Any

from haystack import Pipeline, Document
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
    FastembedSparseDocumentEmbedder,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret, ComponentDevice
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from .utils import format_time
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)

from .components.metadata_enricher import MetadataEnricher
from .components.summarizer import DocumentSummarizer
from .components.query_metadata_extractor import RomanianQueryMetadataExtractor
from .langfuse_tracker import setup_langfuse, get_observe_decorator
from . import config

# Import resilience and caching modules
from .resilience import (
    RetryConfig,
    retry_with_backoff,
    CircuitBreaker,
    RateLimiter,
)
from .cache import CacheManager

# Initialize Langfuse client FIRST (sets up OTel tracer for @observe)
langfuse_client = setup_langfuse()
# Then get the @observe decorator
observe = get_observe_decorator()

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    config.LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"hybrid_rag_{timestamp}.log"

    logger = logging.getLogger("HybridRAG")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class HybridRAGApplication:

    """Advanced Hybrid RAG Application"""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.logger = setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("Initializing Advanced Hybrid RAG Application")
        self.logger.info("=" * 80)

        self.qdrant_url = qdrant_url or config.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or config.QDRANT_API_KEY
        self.collection_name = collection_name or config.QDRANT_COLLECTION

        if not self.qdrant_url:
            raise ValueError(
                "Qdrant URL not found. Set QDRANT_ENDPOINT "
                "in .env or pass qdrant_url as argument."
            )
        if not self.qdrant_api_key:
            self.logger.info("No Qdrant API key provided — connecting without authentication (local mode)")

        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Qdrant URL: {self.qdrant_url}")
        self.logger.info(f"  - Collection: {self.collection_name}")
        self.logger.info(f"  - LLM URL: {config.LLM_URL}")
        self.logger.info(f"  - LLM Model: {config.LLM_MODEL}")
        self.logger.info(f"  - Dense Model: {config.DENSE_EMBEDDING_MODEL}")
        self.logger.info(f"  - Sparse Model: {config.SPARSE_EMBEDDING_MODEL}")
        self.logger.info(
            f"  - Metadata Enrichment: {config.ENABLE_METADATA_EXTRACTION}"
        )
        self.logger.info(f"  - Summarization: {config.ENABLE_SUMMARIZATION}")
        self.logger.info(f"  - Reranking: {config.USE_RERANKER}")

        self.langfuse = setup_langfuse()

        # Initialize caching layer
        self.cache_manager = CacheManager(
            embedding_cache_size=getattr(config, "EMBEDDING_CACHE_SIZE", 1000),
            retrieval_cache_size=getattr(config, "RETRIEVAL_CACHE_SIZE", 500),
            response_cache_size=getattr(config, "RESPONSE_CACHE_SIZE", 200),
            embedding_ttl=getattr(config, "EMBEDDING_CACHE_TTL", 3600),
            retrieval_ttl=getattr(config, "RETRIEVAL_CACHE_TTL", 1800),
            response_ttl=getattr(config, "RESPONSE_CACHE_TTL", 3600),
        )
        self.logger.info("Cache manager initialized")

        # Initialize circuit breakers for external services
        self.qdrant_circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, "QDRANT_CIRCUIT_BREAKER_THRESHOLD", 5),
            recovery_timeout=getattr(config, "QDRANT_CIRCUIT_BREAKER_TIMEOUT", 60.0),
            expected_exception=Exception,
            name="Qdrant",
        )
        self.ollama_circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, "OLLAMA_CIRCUIT_BREAKER_THRESHOLD", 3),
            recovery_timeout=getattr(config, "OLLAMA_CIRCUIT_BREAKER_TIMEOUT", 30.0),
            expected_exception=Exception,
            name="Ollama",
        )
        self.logger.info("Circuit breakers initialized")

        # Initialize rate limiters
        self.ollama_rate_limiter = RateLimiter(
            max_concurrent=getattr(config, "OLLAMA_MAX_CONCURRENT", 2),
            max_per_minute=getattr(config, "OLLAMA_MAX_PER_MINUTE", 60),
            name="Ollama",
        )
        self.qdrant_rate_limiter = RateLimiter(
            max_concurrent=getattr(config, "QDRANT_MAX_CONCURRENT", 10),
            max_per_minute=None,
            name="Qdrant",
        )
        self.logger.info("Rate limiters initialized")

        # Initialize query metadata extractor
        self.query_metadata_extractor = RomanianQueryMetadataExtractor(
            llm_type=getattr(config, "LLM_TYPE", "OLLAMA"),
            llm_model=getattr(config, "LLM_MODEL", "llama3.1:8b"),
            llm_url=getattr(config, "LLM_URL", "http://127.0.0.1:11435"),
            llm_api_key=getattr(config, "LLM_API_KEY", None),
            llm_api_version=getattr(config, "LLM_API_VERSION", None),
        )
        self.logger.info("Query metadata extractor initialized")

        # Initialize components
        self.document_store = None
        self.indexing_pipeline = None
        self.retrieval_pipeline = None
        self.generation_pipeline = None

        self._initialize_document_store()
        self._build_indexing_pipeline()
        self._build_query_pipelines()

        # Warm up pipelines — download and load all models eagerly
        # so the first request isn't penalized by model downloads
        self.logger.info("Warming up pipelines (downloading/loading models)...")
        try:
            self.retrieval_pipeline.warm_up()
            self.logger.info("  [OK] Retrieval pipeline warmed up")
        except Exception as e:
            self.logger.warning(f"  [WARN] Retrieval pipeline warm_up failed: {e}")
        try:
            self.generation_pipeline.warm_up()
            self.logger.info("  [OK] Generation pipeline warmed up")
        except Exception as e:
            self.logger.warning(f"  [WARN] Generation pipeline warm_up failed: {e}")

        self.logger.info("Hybrid RAG Application initialized successfully")
        self.logger.info("=" * 80)

    # ===============================
    # DOCUMENT STORE
    # ===============================
    def _initialize_document_store(self):
        """Initialize Qdrant document store with hybrid search"""
        self.logger.info("Connecting to Qdrant...")
        
        # Retry configuration for Qdrant connection
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=10.0,
        )
        
        @retry_with_backoff(
            config=retry_config,
            exceptions=(Exception,),
            on_retry=lambda e, attempt: self.logger.warning(
                f"Qdrant connection attempt {attempt} failed: {e}"
            ),
        )
        def connect_to_qdrant():
            qdrant_kwargs = dict(
                url=self.qdrant_url,
                index=self.collection_name,
                embedding_dim=config.QDRANT_EMBEDDING_DIM,
                use_sparse_embeddings=True,
                recreate_index=False,
                return_embedding=False,
                wait_result_from_api=True,
            )
            if self.qdrant_api_key:
                qdrant_kwargs["api_key"] = Secret.from_token(self.qdrant_api_key)
            return self.qdrant_circuit_breaker.call(
                lambda: QdrantDocumentStore(**qdrant_kwargs)
            )
        
        try:
            self.document_store = connect_to_qdrant()
            doc_count = self.document_store.count_documents()
            self.logger.info(f"Connected to Qdrant collection: {self.collection_name}")
            self.logger.info(f"  - Existing documents: {doc_count}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant after retries: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not establish connection to Qdrant. "
                f"Please check if Qdrant is accessible at {self.qdrant_url} "
                f"and verify your API key."
            ) from e

    # ===============================
    # INDEXING PIPELINE
    # ===============================
    def _build_indexing_pipeline(self):
        """Build PRODUCTION-GRADE document indexing pipeline"""
        self.logger.info("Building production-grade indexing pipeline...")
        try:
            self.indexing_pipeline = Pipeline()
            
            # 1. DOCUMENT TYPE DETECTION (invoice/contract/receipt)
            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                from .components.document_type_detector import DocumentTypeDetector
                self.indexing_pipeline.add_component(
                    "type_detector",
                    DocumentTypeDetector()
                )
                self.logger.info("  [OK] Document type detection enabled")
            
            # 2. METADATA EXTRACTION - Extract from FULL DOCUMENT before chunking
            # This ensures all chunks inherit the same metadata (company, date, invoice_number, etc.)
            if config.ENABLE_METADATA_EXTRACTION:
                # Load prompt from file
                metadata_prompt = None
                prompt_file = getattr(config, "METADATA_EXTRACTION_PROMPT_FILE", None)
                if prompt_file and prompt_file.exists():
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        metadata_prompt = f.read()
                        self.logger.info(f"  [OK] Loaded metadata extraction prompt from {prompt_file.name}")
                else:
                    self.logger.warning(f"  [WARN] Metadata prompt file not found: {prompt_file}")
                    metadata_prompt = None  # Will use MetadataEnricher's default
                
                self.indexing_pipeline.add_component(
                    "metadata_enricher",
                    MetadataEnricher(
                        llm_type=config.LLM_TYPE,
                        llm_model=config.LLM_MODEL,
                        llm_url=config.LLM_URL,
                        llm_api_key=config.LLM_API_KEY,
                        llm_api_version=config.LLM_API_VERSION,
                        metadata_fields=config.METADATA_FIELDS,
                        append_metadata_to_content=False,
                        prompt_template=metadata_prompt,
                    ),
                )
                self.logger.info("  [OK] Metadata extraction enabled")
            
            # 3. SEMANTIC CHUNKING or Fixed-Size Chunking
            if getattr(config, "USE_SEMANTIC_CHUNKING", True):
                from .components.semantic_chunker import SemanticDocumentChunker
                self.indexing_pipeline.add_component(
                    "semantic_chunker",
                    SemanticDocumentChunker(
                        min_chunk_size=getattr(config, "SEMANTIC_CHUNK_MIN_SIZE", 100),
                        max_chunk_size=getattr(config, "SEMANTIC_CHUNK_MAX_SIZE", 800),
                        overlap_size=getattr(config, "SEMANTIC_CHUNK_OVERLAP", 50),
                    ),
                )
                self.logger.info("  [OK] Semantic chunking enabled (logical sections)")
            else:
                # Fall back to old fixed-size chunking
                self.indexing_pipeline.add_component(
                    "splitter",
                    DocumentSplitter(
                        split_by=config.CHUNK_SPLIT_BY,
                        split_length=config.CHUNK_SIZE,
                        split_overlap=config.CHUNK_OVERLAP,
                    ),
                )
                self.logger.info("  [WARN] Using legacy fixed-size chunking")
            
            # 4. BOILERPLATE FILTERING - Remove legal/payment text before embedding
            if getattr(config, "USE_BOILERPLATE_FILTER", True):
                from .components.boilerplate_filter import BoilerplateFilter
                self.indexing_pipeline.add_component(
                    "boilerplate_filter",
                    BoilerplateFilter(
                        min_boilerplate_score=getattr(config, "BOILERPLATE_MIN_SCORE", 3),
                        skip_legal_sections=getattr(config, "SKIP_LEGAL_SECTIONS", True),
                        skip_payment_sections=getattr(config, "SKIP_PAYMENT_SECTIONS", True),
                    ),
                )
                self.logger.info("  [OK] Boilerplate filtering enabled")

            # 5. SUMMARIZATION - Optional per-chunk summaries
            if config.ENABLE_SUMMARIZATION:
                self.indexing_pipeline.add_component(
                    "summarizer",
                    DocumentSummarizer(
                        llm_type=config.LLM_TYPE,
                        llm_model=config.LLM_MODEL,
                        llm_url=config.LLM_URL,
                        llm_api_key=config.LLM_API_KEY,
                        llm_api_version=config.LLM_API_VERSION,
                        max_summary_length=config.SUMMARY_MAX_LENGTH,
                        summary_style=config.SUMMARY_STYLE,
                    ),
                )
                self.logger.info("  [OK] Summarization enabled")

            # 6. EMBEDDERS - Dense + Sparse
            self.indexing_pipeline.add_component(
                "dense_embedder",
                SentenceTransformersDocumentEmbedder(
                    model=config.DENSE_EMBEDDING_MODEL,
                    device=ComponentDevice.from_str(config.EMBEDDING_DEVICE),
                ),
            )
            self.indexing_pipeline.add_component(
                "sparse_embedder",
                FastembedSparseDocumentEmbedder(model=config.SPARSE_EMBEDDING_MODEL),
            )
            self.indexing_pipeline.add_component(
                "writer", DocumentWriter(document_store=self.document_store)
            )
            self.logger.info("  [OK] Dense + Sparse embedders configured")

            # CONNECT COMPONENTS IN PRODUCTION PIPELINE
            # Flow: type_detector → metadata_enricher → semantic_chunker → boilerplate_filter → summarizer → embedders → writer
            current_component = None
            
            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                current_component = "type_detector"
            
            if config.ENABLE_METADATA_EXTRACTION:
                if current_component:
                    self.indexing_pipeline.connect(current_component, "metadata_enricher")
                current_component = "metadata_enricher"
            
            if getattr(config, "USE_SEMANTIC_CHUNKING", True):
                if current_component:
                    self.indexing_pipeline.connect(current_component, "semantic_chunker")
                current_component = "semantic_chunker"
            else:
                if current_component:
                    self.indexing_pipeline.connect(current_component, "splitter")
                current_component = "splitter"
            
            if getattr(config, "USE_BOILERPLATE_FILTER", True):
                if current_component:
                    self.indexing_pipeline.connect(current_component, "boilerplate_filter")
                current_component = "boilerplate_filter"
            
            if config.ENABLE_SUMMARIZATION:
                if current_component:
                    self.indexing_pipeline.connect(current_component, "summarizer")
                current_component = "summarizer"
            
            # Connect to embedders
            if current_component:
                self.indexing_pipeline.connect(current_component, "dense_embedder")
            self.indexing_pipeline.connect("dense_embedder", "sparse_embedder")
            self.indexing_pipeline.connect("sparse_embedder", "writer")

            self.logger.info("=" * 80)
            self.logger.info("PRODUCTION-GRADE INDEXING PIPELINE READY:")
            self.logger.info("  1. Document type detection (invoice/contract/receipt)")
            self.logger.info("  2. Metadata extraction (from full document)")
            self.logger.info("  3. Semantic chunking (logical sections)")
            self.logger.info("  4. Boilerplate filtering (remove legal/payment text)")
            self.logger.info("  5. Smart summarization")
            self.logger.info("  6. Hybrid embeddings (dense + sparse)")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to build indexing pipeline: {e}", exc_info=True)
            raise

    # ===============================
    # QUERY PIPELINES
    # ===============================
    def _build_query_pipelines(self):
        """Build retrieval and generation pipelines separately
        
        Architecture matches deepset/haystack-rag-app:
        - Sparse retriever (BM25-like) with FULL original query
        - Dense retriever (embedding) with FULL original query  
        - DocumentJoiner to merge results (like deepset)
        - Optional reranker
        """
        self.logger.info("Building retrieval and generation pipelines (deepset-style)...")

        try:
            # ---- RETRIEVAL PIPELINE (DEEPSET ARCHITECTURE) ----
            self.retrieval_pipeline = Pipeline()
            
            # Store dense embedder as instance variable for reuse
            self.dense_embedder = SentenceTransformersTextEmbedder(
                model=config.DENSE_EMBEDDING_MODEL,
                device=ComponentDevice.from_str(config.EMBEDDING_DEVICE),
                prefix=config.DENSE_EMBEDDING_PREFIX,
            )
            self.retrieval_pipeline.add_component("dense_embedder", self.dense_embedder)
            
            # Sparse embedder for BM25-like retrieval
            self.retrieval_pipeline.add_component(
                "sparse_embedder",
                FastembedSparseTextEmbedder(model=config.SPARSE_EMBEDDING_MODEL),
            )
            
            # SEPARATE RETRIEVERS (like deepset's BM25 + Embedding retrievers)
            retriever_top_k = config.TOP_K * 2 if config.USE_RERANKER else config.TOP_K
            
            # Sparse retriever (BM25-like) - uses sparse embeddings
            self.retrieval_pipeline.add_component(
                "sparse_retriever",
                QdrantSparseEmbeddingRetriever(
                    document_store=self.document_store,
                    top_k=retriever_top_k,
                ),
            )
            
            # Dense retriever (embedding-based)
            self.retrieval_pipeline.add_component(
                "dense_retriever",
                QdrantEmbeddingRetriever(
                    document_store=self.document_store,
                    top_k=retriever_top_k,
                ),
            )
            
            # DocumentJoiner - merges results from both retrievers (like deepset)
            self.retrieval_pipeline.add_component(
                "document_joiner",
                DocumentJoiner(join_mode="concatenate"),
            )
            
            if config.USE_RERANKER:
                self.retrieval_pipeline.add_component(
                    "reranker",
                    SentenceTransformersSimilarityRanker(
                        model=config.RERANKER_MODEL, top_k=config.RERANKER_TOP_K
                    ),
                )
            
            # Connect components (matching deepset architecture)
            self.retrieval_pipeline.connect(
                "sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding"
            )
            self.retrieval_pipeline.connect(
                "dense_embedder.embedding", "dense_retriever.query_embedding"
            )
            self.retrieval_pipeline.connect(
                "sparse_retriever.documents", "document_joiner.documents"
            )
            self.retrieval_pipeline.connect(
                "dense_retriever.documents", "document_joiner.documents"
            )
            if config.USE_RERANKER:
                self.retrieval_pipeline.connect(
                    "document_joiner.documents", "reranker.documents"
                )
            
            self.logger.info("  [OK] Sparse retriever (BM25-like) configured")
            self.logger.info("  [OK] Dense retriever (embedding) configured")
            self.logger.info("  [OK] DocumentJoiner (concatenate mode) configured")
            if config.USE_RERANKER:
                self.logger.info(f"  [OK] Reranker configured: {config.RERANKER_MODEL}")

            # ---- GENERATION PIPELINE ----
            # Load RAG prompts from files
            rag_system_prompt = config.RAG_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8") if config.RAG_SYSTEM_PROMPT_FILE.exists() else "You are a helpful assistant."
            rag_user_prompt = config.ACTIVE_RAG_USER_PROMPT.read_text(encoding="utf-8") if config.ACTIVE_RAG_USER_PROMPT.exists() else "Answer: {{ query }}"
            
            template = [
                ChatMessage.from_system(rag_system_prompt),
                ChatMessage.from_user(rag_user_prompt),
            ]
            self.generation_pipeline = Pipeline()
            self.generation_pipeline.add_component(
                "prompt_builder", ChatPromptBuilder(template=template, required_variables=["query", "documents"])
            )
            
            # Use correct generator based on config.LLM_TYPE
            if config.LLM_TYPE == "OLLAMA":
                self.generation_pipeline.add_component(
                    "generator",
                    OllamaChatGenerator(
                        model=config.LLM_MODEL,
                        url=config.LLM_URL,
                        generation_kwargs=config.GENERATION_CONFIG,
                        timeout=config.OLLAMA_TIMEOUT,
                        keep_alive=config.OLLAMA_KEEP_ALIVE,
                    ),
                )
                self.logger.info(f"Generation pipeline using Ollama: {config.LLM_MODEL}")
            elif config.LLM_TYPE == "AZURE_OPENAI":
                self.generation_pipeline.add_component(
                    "generator",
                    AzureOpenAIChatGenerator(
                        azure_deployment=config.LLM_MODEL,
                        azure_endpoint=config.LLM_URL,
                        api_key=Secret.from_token(config.LLM_API_KEY),
                        api_version=config.LLM_API_VERSION,
                        generation_kwargs=config.GENERATION_CONFIG,
                    ),
                )
                self.logger.info(f"Generation pipeline using Azure OpenAI: {config.LLM_MODEL}")
            else:
                raise ValueError(f"Unsupported LLM_TYPE: {config.LLM_TYPE}")
            
            self.generation_pipeline.connect(
                "prompt_builder.prompt", "generator.messages"
            )
            self.logger.info("Retrieval and generation pipelines built successfully")

        except Exception as e:
            self.logger.error("Failed to build query pipelines", exc_info=True)
            raise

    # ===============================
    # DOCUMENT LOADING
    # ===============================
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        self.logger.info(f"Loading documents from: {folder_path}")
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        raw_docs = []

        # TXT
        txt_files = list(folder.glob("*.txt"))
        if txt_files:
            txt_converter = TextFileToDocument()
            txt_docs = txt_converter.run(sources=txt_files)
            raw_docs.extend(txt_docs["documents"])

        # PDF
        pdf_files = list(folder.glob("*.pdf"))
        if pdf_files:
            pdf_converter = PyPDFToDocument()
            pdf_docs = pdf_converter.run(sources=pdf_files)
            raw_docs.extend(pdf_docs["documents"])

        # MD
        md_files = list(folder.glob("*.md"))
        if md_files:
            md_converter = TextFileToDocument()
            md_docs = md_converter.run(sources=md_files)
            raw_docs.extend(md_docs["documents"])

        self.logger.info(f"Total documents loaded: {len(raw_docs)}")
        return raw_docs

    # ===============================
    # DUPLICATE FILTERING
    # ===============================
    def get_indexed_sources(self) -> set:
        try:
            all_docs = list(self.document_store.filter_documents())
            sources = {
                doc.meta.get("file_path", doc.meta.get("source", ""))
                for doc in all_docs
            }
            sources.discard("")
            return sources
        except Exception as e:
            self.logger.warning(f"Could not retrieve indexed sources: {e}")
            return set()

    def filter_duplicate_documents(self, documents: List[Document]) -> List[Document]:
        indexed_sources = self.get_indexed_sources()
        if not indexed_sources:
            return documents
        new_docs = []
        skipped_sources = set()
        for doc in documents:
            source = doc.meta.get("file_path", doc.meta.get("source", ""))
            if source and source in indexed_sources:
                skipped_sources.add(source)
            else:
                new_docs.append(doc)
        if skipped_sources:
            self.logger.info(f"Skipping {len(skipped_sources)} already indexed sources")
        return new_docs

    # ===============================
    # INDEX DOCUMENTS
    # ===============================
    def index_documents(
        self, documents: List[Document], skip_duplicates: bool = True
    ) -> int:
        if skip_duplicates:
            documents = self.filter_duplicate_documents(documents)
            if not documents:
                self.logger.warning("No new documents to index")
                return self.document_store.count_documents()

        self.logger.info(f"Starting indexing for {len(documents)} documents")
        
        # Run pipeline with rate limiting and error handling
        try:
            # Determine the first component based on configuration
            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                # Pipeline starts with type_detector
                with self.qdrant_rate_limiter:
                    self.indexing_pipeline.run({"type_detector": {"documents": documents}})
            elif config.ENABLE_METADATA_EXTRACTION:
                # Pipeline starts with metadata_enricher
                with self.qdrant_rate_limiter:
                    self.indexing_pipeline.run({"metadata_enricher": {"documents": documents}})
            elif getattr(config, "USE_SEMANTIC_CHUNKING", True):
                # Pipeline starts with semantic_chunker
                with self.qdrant_rate_limiter:
                    self.indexing_pipeline.run({"semantic_chunker": {"documents": documents}})
            else:
                # Fallback to splitter
                with self.qdrant_rate_limiter:
                    self.indexing_pipeline.run({"splitter": {"documents": documents}})
            
            final_count = self.document_store.count_documents()
            self.logger.info(f"Indexing complete. Total documents: {final_count}")
            
            # Invalidate retrieval cache after indexing new documents
            self.cache_manager.retrieval_cache.invalidate_all()
            self.logger.info("Retrieval cache invalidated after indexing")
            
            return final_count
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to index documents. {len(documents)} documents were not indexed. "
                f"Error: {str(e)}"
            ) from e

    # ===============================
    # QUERY PIPELINE HELPERS
    # ===============================
    
    @observe(name="metadata_extraction")
    def _extract_metadata(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query"""
        extraction_start = time.time()
        query_metadata = self.query_metadata_extractor.run(query=query)
        
        extracted_filters = query_metadata.get("filters")
        search_query = query_metadata.get("search_query", query)
        extracted_metadata = query_metadata.get("metadata", {})
        extraction_time = time.time() - extraction_start
        
        if extracted_filters:
            self.logger.info(f"Extracted filters: {extracted_filters}")
            self.logger.info(f"Extracted metadata: {extracted_metadata}")
        else:
            self.logger.info("No filters extracted, using full semantic search")
        
        # Normalize numeric filters
        if extracted_filters:
            for condition in extracted_filters.get("conditions", []):
                value = condition.get("value")
                if isinstance(value, str) and value.isdigit():
                    condition["value"] = int(value)
        
        return {
            "filters": extracted_filters,
            "metadata": extracted_metadata,
            "search_query": search_query,
            "extraction_time": extraction_time,
        }
    
    def _filter_by_score(
        self, retrieved_docs: List[Document], query: str, extracted_filters: Optional[Dict]
    ) -> List[Document]:
        """Apply minimal filtering - deepset-style (no aggressive score filtering)
        
        Deepset approach: Pass all documents, let reranker/LLM handle relevance.
        We only deduplicate and limit count.
        """
        MAX_DOCS = config.RERANKER_TOP_K if config.USE_RERANKER else config.TOP_K
        
        # Deduplicate by document ID (important when joining sparse+dense results)
        seen_ids = set()
        unique_docs = []
        for d in retrieved_docs:
            doc_id = d.id if d.id else hash(d.content[:100])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(d)
        
        self.logger.info(f"Documents after deduplication: {len(unique_docs)} (from {len(retrieved_docs)})")
        
        # Sort by score and limit (no score threshold - deepset-style)
        sorted_docs = sorted(
            unique_docs,
            key=lambda d: getattr(d, "score", 0),
            reverse=True
        )[:MAX_DOCS]
        
        return sorted_docs
    
    @observe(name="document_retrieval")
    def _retrieve_documents(
        self, query: str, metadata: Dict[str, Any]
    ) -> tuple[List[Document], float]:
        """Retrieve documents using deepset-style architecture
        
        Key difference from before: 
        - ALWAYS use ORIGINAL query for embeddings (not modified search_query)
        - Filters are optional enhancement, not primary retrieval method
        """
        retrieval_start = time.time()
        
        # IMPORTANT: Use ORIGINAL query for embeddings (deepset-style)
        # Metadata filters are just for narrowing, not for query modification
        original_query = query  # Always use original, not metadata["search_query"]
        extracted_filters = metadata.get("filters")
        
        self.logger.info(f"Retrieving with ORIGINAL query: '{original_query}'")
        if extracted_filters:
            self.logger.info(f"Additional filters: {extracted_filters}")
        
        # Check retrieval cache (use original query as key)
        cached_docs = self.cache_manager.retrieval_cache.get(
            query=original_query,
            filters=extracted_filters,
            top_k=config.TOP_K
        )
        if cached_docs is not None:
            self.logger.info(f"Retrieved {len(cached_docs)} documents from cache")
            return cached_docs, time.time() - retrieval_start
        
        # Build retrieval inputs for new deepset-style pipeline
        retrieval_inputs = {
            "dense_embedder": {"text": original_query},
            "sparse_embedder": {"text": original_query},
        }
        
        # Add filters to BOTH retrievers if present (optional enhancement)
        if extracted_filters:
            retrieval_inputs["sparse_retriever"] = {"filters": extracted_filters}
            retrieval_inputs["dense_retriever"] = {"filters": extracted_filters}
        
        if config.USE_RERANKER:
            retrieval_inputs["reranker"] = {"query": original_query}
        
        # Run retrieval with rate limiting and circuit breaker
        try:
            with self.qdrant_rate_limiter:
                retrieval_result = self.qdrant_circuit_breaker.call(
                    self.retrieval_pipeline.run,
                    retrieval_inputs
                )
        except Exception as e:
            self.logger.warning(f"Retrieval with filters failed: {e}")
            
            # FALLBACK: Retry WITHOUT filters (deepset-style - filters are optional)
            if extracted_filters:
                self.logger.info("Retrying retrieval WITHOUT filters...")
                retrieval_inputs_no_filters = {
                    "dense_embedder": {"text": original_query},
                    "sparse_embedder": {"text": original_query},
                }
                if config.USE_RERANKER:
                    retrieval_inputs_no_filters["reranker"] = {"query": original_query}
                
                try:
                    with self.qdrant_rate_limiter:
                        retrieval_result = self.qdrant_circuit_breaker.call(
                            self.retrieval_pipeline.run,
                            retrieval_inputs_no_filters
                        )
                    self.logger.info("Retrieval succeeded without filters")
                except Exception as e2:
                    self.logger.error(f"Retrieval failed even without filters: {e2}", exc_info=True)
                    return [], time.time() - retrieval_start
            else:
                self.logger.error(f"Retrieval failed: {e}", exc_info=True)
                return [], time.time() - retrieval_start
        
        # Cache embeddings with original query as key
        if "dense_embedder" in retrieval_result:
            dense_emb = retrieval_result["dense_embedder"].get("embedding")
            if dense_emb:
                self.cache_manager.embedding_cache.put_dense(original_query, dense_emb)
        if "sparse_embedder" in retrieval_result:
            sparse_emb = retrieval_result["sparse_embedder"].get("sparse_embedding")
            if sparse_emb:
                self.cache_manager.embedding_cache.put_sparse(original_query, sparse_emb)
        
        # Get results from reranker (if enabled) or document_joiner (deepset-style)
        if config.USE_RERANKER:
            retrieved_docs = retrieval_result.get("reranker", {}).get("documents", [])
        else:
            retrieved_docs = retrieval_result.get("document_joiner", {}).get("documents", [])
        
        retrieval_time = time.time() - retrieval_start
        self.logger.info(f"Retrieved {len(retrieved_docs)} raw documents in {retrieval_time:.2f}s")
        
        # Apply deduplication and limit (no aggressive score filtering - deepset-style)
        filtered_docs = self._filter_by_score(retrieved_docs, query, extracted_filters)
        
        self.logger.info(f"Final documents after processing: {len(filtered_docs)}")
        
        # FALLBACK: If filters produced 0 results, retry WITHOUT filters (deepset-style)
        if not filtered_docs and extracted_filters:
            self.logger.warning(
                f"Filters returned 0 results. Retrying WITHOUT filters for broader search. "
                f"Original filters: {extracted_filters}"
            )
            retrieval_inputs_no_filters = {
                "dense_embedder": {"text": original_query},
                "sparse_embedder": {"text": original_query},
            }
            if config.USE_RERANKER:
                retrieval_inputs_no_filters["reranker"] = {"query": original_query}
            
            try:
                with self.qdrant_rate_limiter:
                    retrieval_result_fallback = self.qdrant_circuit_breaker.call(
                        self.retrieval_pipeline.run,
                        retrieval_inputs_no_filters
                    )
                
                if config.USE_RERANKER:
                    fallback_docs = retrieval_result_fallback.get("reranker", {}).get("documents", [])
                else:
                    fallback_docs = retrieval_result_fallback.get("document_joiner", {}).get("documents", [])
                
                filtered_docs = self._filter_by_score(fallback_docs, query, None)
                self.logger.info(
                    f"Fallback retrieval (no filters) returned {len(filtered_docs)} documents"
                )
            except Exception as e_fallback:
                self.logger.error(f"Fallback retrieval also failed: {e_fallback}", exc_info=True)
        
        # Cache retrieval results (use original query as key)
        if filtered_docs:
            self.cache_manager.retrieval_cache.put(
                query=original_query,
                documents=filtered_docs,
                filters=extracted_filters,
                top_k=config.TOP_K
            )
        
        return filtered_docs, retrieval_time
    
    @observe(name="answer_generation")
    def _generate_answer(
        self, query: str, retrieved_docs: List[Document]
    ) -> tuple[List[ChatMessage], float]:
        """Generate answer from retrieved documents with caching and rate limiting"""
        generation_start = time.time()
        
        # Check response cache
        cached_response = self.cache_manager.response_cache.get(query, retrieved_docs)
        if cached_response is not None:
            self.logger.info("Using cached LLM response")
            return cached_response, time.time() - generation_start
        
        generation_inputs = {
            "prompt_builder": {"query": query, "documents": retrieved_docs}
        }
        
        # Apply rate limiting and circuit breaker for LLM calls
        try:
            # Only apply rate limiter for Ollama (not for Azure OpenAI)
            if config.LLM_TYPE == "OLLAMA":
                with self.ollama_rate_limiter:
                    generation_result = self.ollama_circuit_breaker.call(
                        self.generation_pipeline.run,
                        generation_inputs
                    )
            else:
                # Azure OpenAI has its own rate limiting
                generation_result = self.generation_pipeline.run(generation_inputs)
                
            replies = generation_result.get("generator", {}).get("replies", [])
            
            # Cache the response
            if replies:
                self.cache_manager.response_cache.put(query, retrieved_docs, replies)
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            # Return graceful error message
            replies = [
                ChatMessage.from_assistant(
                    "I apologize, but I'm having trouble generating a response right now. "
                    "Please try again in a moment."
                )
            ]
        
        generation_time = time.time() - generation_start
        return replies, generation_time
    
    def _empty_response(self, metadata: Dict[str, Any], total_start: float) -> Dict[str, Any]:
        """Return empty response when no documents found"""
        self.logger.warning("No documents retrieved. Skipping generation.")
        return {
            "retriever": {"documents": []},
            "generator": {
                "replies": [
                    ChatMessage.from_assistant(
                        "I don't have enough information to answer this question."
                    )
                ]
            },
            "metadata": {
                "extraction_time": metadata.get("extraction_time", 0),
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": time.time() - total_start,
                "documents_retrieved": 0,
                "extracted_filters": metadata.get("filters"),
                "extracted_metadata": metadata.get("metadata", {}),
            },
        }
    
    def _build_response(
        self,
        metadata: Dict[str, Any],
        retrieved_docs: List[Document],
        replies: List[ChatMessage],
        total_start: float,
        retrieval_time: float,
        generation_time: float,
    ) -> Dict[str, Any]:
        """Build final response structure"""
        total_time = time.time() - total_start
        
        self.logger.info(
            f"Timing - Retrieval: {format_time(retrieval_time)} | "
            f"Generation: {format_time(generation_time)} | Total: {format_time(total_time)}"
        )
        
        return {
            "retriever": {"documents": retrieved_docs},
            "generator": {"replies": replies},
            "metadata": {
                "extraction_time": metadata.get("extraction_time", 0),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "documents_retrieved": len(retrieved_docs),
                "extracted_filters": metadata.get("filters"),
                "extracted_metadata": metadata.get("metadata", {}),
            },
        }
    
    @observe(name="hybrid_rag_query", capture_input=True, capture_output=True)
    def query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main query orchestrator"""
        self.logger.info(f"Processing query: {query}")
        total_start = time.time()
        
        # Get Langfuse trace ID using the official SDK method
        trace_id = None
        if langfuse_client and config.LANGFUSE_ENABLED:
            try:
                trace_id = langfuse_client.get_current_trace_id()
            except Exception as e:
                self.logger.debug(f"Could not get trace ID: {e}")
        
        try:
            metadata = self._extract_metadata(query)
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            # Return error response with trace_id
            error_resp = {
                "retriever": {"documents": []},
                "generator": {
                    "replies": [
                        ChatMessage.from_assistant(
                            "I encountered an error processing your question."
                        )
                    ]
                },
                "metadata": {
                    "extraction_time": 0,
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "total_time": time.time() - total_start,
                    "documents_retrieved": 0,
                    "extracted_filters": None,
                    "extracted_metadata": {},
                    "error": str(e),
                },
            }
            if trace_id:
                error_resp["_internal"] = {"langfuse_trace_id": trace_id}
            return error_resp
        
        retrieved_docs, retrieval_time = self._retrieve_documents(query, metadata)
        
        if not retrieved_docs:
            empty_resp = self._empty_response(metadata, total_start)
            # Add trace ID even for empty responses
            if trace_id:
                empty_resp["_internal"] = {"langfuse_trace_id": trace_id}
            return empty_resp
        
        replies, generation_time = self._generate_answer(query, retrieved_docs)
        
        response = self._build_response(
                metadata,
                retrieved_docs,
                replies,
                total_start,
                retrieval_time,
                generation_time,
            )

        # Add trace ID to response for evaluation tracking
        if trace_id:
            response["_internal"] = {"langfuse_trace_id": trace_id}
        
        return response
    

    # ===============================
    # OTHER UTILITIES
    # ===============================
    def get_document_count(self) -> int:
        return self.document_store.count_documents()

    def _ensure_collection_exists(self):
        """Check if the Qdrant collection exists; recreate it if missing.
        
        Handles the case where the collection was deleted externally
        (e.g. manual cleanup) while the app was running.
        """
        try:
            self.document_store.count_documents()
        except Exception as e:
            error_msg = str(e)
            if "doesn't exist" in error_msg or "Not Found" in error_msg:
                self.logger.warning(
                    f"Collection '{self.collection_name}' not found — recreating."
                )
                self.recreate_collection()
            else:
                raise

    def recreate_collection(self):
        self.logger.warning(
            f"Recreating collection '{self.collection_name}' - this will delete all data!"
        )
        qdrant_kwargs = dict(
            url=self.qdrant_url,
            index=self.collection_name,
            embedding_dim=config.QDRANT_EMBEDDING_DIM,
            use_sparse_embeddings=True,
            recreate_index=True,
            return_embedding=False,
            wait_result_from_api=True,
        )
        if self.qdrant_api_key:
            qdrant_kwargs["api_key"] = Secret.from_token(self.qdrant_api_key)
        self.document_store = QdrantDocumentStore(**qdrant_kwargs)
        self._build_indexing_pipeline()
        self._build_query_pipelines()

    def get_statistics(self) -> Dict[str, Any]:
        docs = self.document_store.filter_documents()
        return {
            "total_documents": len(docs),
            "with_summaries": sum(1 for d in docs if d.meta.get("summary")),
            "with_metadata": sum(1 for d in docs if d.meta.get("entities")),
            "sources": set(d.meta.get("file_path", "Unknown") for d in docs),
            "cache_stats": self.get_cache_stats(),
            "resilience_stats": self.get_resilience_stats(),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers"""
        return self.cache_manager.get_all_stats()
    
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get circuit breaker and rate limiter status"""
        return {
            "qdrant_circuit_breaker": {
                "state": self.qdrant_circuit_breaker.state,
                "failure_count": self.qdrant_circuit_breaker._failure_count,
            },
            "ollama_circuit_breaker": {
                "state": self.ollama_circuit_breaker.state,
                "failure_count": self.ollama_circuit_breaker._failure_count,
            },
        }

    # ===============================
    # MAYAN EDMS INTEGRATION
    # ===============================
    
    def delete_all_versions_of_document(self, document_id: int) -> int:
        """
        Delete ALL chunks for a document_id (all versions).
        Ensures only the latest version is stored after re-indexing.
        
        Args:
            document_id: The Mayan document ID
            
        Returns:
            Number of chunks deleted
        """
        self.logger.info(f"Deleting all versions for document_id={document_id}")
        
        try:
            filters = {
                "field": "meta.document_id",
                "operator": "==",
                "value": document_id,
            }
            
            docs_to_delete = self.document_store.filter_documents(filters=filters)
            
            if not docs_to_delete:
                self.logger.info(f"No existing chunks found for document_id={document_id}")
                return 0
            
            # Log which versions are being removed
            old_versions = set(d.meta.get("document_version_id") for d in docs_to_delete)
            self.logger.info(
                f"Removing {len(docs_to_delete)} chunks from "
                f"{len(old_versions)} previous version(s): {old_versions}"
            )
            
            doc_ids = [doc.id for doc in docs_to_delete if doc.id]
            if doc_ids:
                self.document_store.delete_documents(document_ids=doc_ids)
                self.cache_manager.retrieval_cache.invalidate_all()
                
            return len(doc_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}", exc_info=True)
            raise

    def delete_chunks_by_content_hash(self, content_hash: str, exclude_document_id: int) -> int:
        """
        Delete chunks that have the same content_hash but a DIFFERENT document_id.
        
        Handles the case where a file is deleted from Mayan and re-uploaded,
        getting a new document_id. The old chunks (from the old document_id)
        would otherwise remain as orphans.
        
        Args:
            content_hash: SHA-256 hash of the raw file bytes
            exclude_document_id: The current document_id (don't delete these)
            
        Returns:
            Number of chunks deleted
        """
        try:
            filters = {
                "field": "meta.content_hash",
                "operator": "==",
                "value": content_hash,
            }
            
            matching_docs = self.document_store.filter_documents(filters=filters)
            
            # Only delete chunks from OTHER document_ids
            to_delete = [
                doc for doc in matching_docs
                if doc.meta.get("document_id") != exclude_document_id and doc.id
            ]
            
            if not to_delete:
                return 0
            
            old_doc_ids = set(d.meta.get("document_id") for d in to_delete)
            self.logger.info(
                f"Content hash {content_hash[:12]}... found in {len(to_delete)} chunks "
                f"from old document_id(s) {old_doc_ids} — deleting orphans"
            )
            
            self.document_store.delete_documents(document_ids=[d.id for d in to_delete])
            self.cache_manager.retrieval_cache.invalidate_all()
            return len(to_delete)
            
        except Exception as e:
            self.logger.warning(f"Content hash dedup check failed: {e}")
            return 0
    
    def is_document_version_indexed(self, document_id: int, document_version_id: int) -> bool:
        """
        Check if a (document_id, document_version_id) pair already has chunks in the store.
        
        Args:
            document_id: Mayan document ID
            document_version_id: Mayan document version ID
            
        Returns:
            True if chunks already exist for this exact document version
        """
        try:
            filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.document_id", "operator": "==", "value": document_id},
                    {"field": "meta.document_version_id", "operator": "==", "value": document_version_id},
                ],
            }
            existing = self.document_store.filter_documents(filters=filters)
            return len(existing) > 0
        except Exception as e:
            self.logger.warning(f"Could not check existing version: {e}")
            return False

    def index_mayan_document(
        self,
        document: Document,
        document_id: int,
        document_version_id: int,
        allowed_users: List[int],
    ) -> Dict[str, Any]:
        """
        Index a document from Mayan EDMS — latest version only.
        
        Strategy: Only the most recent version of each document is stored.
        
        Logic:
        1. If this exact (document_id, document_version_id) is already indexed → skip
        2. Delete ALL previous versions of this document_id
        3. Index the new version with Mayan metadata
        
        Args:
            document: Haystack Document with content
            document_id: Mayan document ID
            document_version_id: Mayan document version ID
            allowed_users: List of user IDs with access permission
            
        Returns:
            Dict with 'status' ('indexed' or 'skipped') and 'document_count'
        """
        self.logger.info(
            f"Indexing Mayan document: document_id={document_id}, "
            f"document_version_id={document_version_id}, "
            f"allowed_users={allowed_users}"
        )
        
        # Step 0: Ensure collection exists (handles external deletion)
        self._ensure_collection_exists()
        
        # Step 1: Skip if this exact version is already indexed
        if self.is_document_version_indexed(document_id, document_version_id):
            self.logger.info(
                f"document_id={document_id}, document_version_id={document_version_id} "
                f"already indexed, skipping"
            )
            return {"status": "skipped", "document_count": 0}
        
        # Step 2: Delete ALL previous versions of this document
        deleted_count = self.delete_all_versions_of_document(document_id)
        if deleted_count > 0:
            self.logger.info(
                f"Replaced {deleted_count} chunks from previous version(s) "
                f"of document_id={document_id}"
            )
        
        # Step 2b: Delete orphaned chunks from same file under old document_id
        content_hash = document.meta.get("content_hash")
        if content_hash:
            orphan_count = self.delete_chunks_by_content_hash(content_hash, document_id)
            if orphan_count > 0:
                self.logger.info(
                    f"Cleaned up {orphan_count} orphaned chunks from same file "
                    f"under previous document_id(s)"
                )
        
        # Step 3: Add Mayan metadata to the document
        document.meta["document_id"] = document_id
        document.meta["document_version_id"] = document_version_id
        document.meta["allowed_users"] = allowed_users
        
        # Step 4: Run through existing pipeline (includes MetadataEnricher)
        doc_count = self.index_documents([document], skip_duplicates=False)
        return {"status": "indexed", "document_count": doc_count}
    
    def query_with_permissions(
        self, 
        query: str, 
        user_id: int, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with permission filtering.
        
        Only returns documents where user_id is in allowed_users.
        
        Args:
            query: The search query
            user_id: The requesting user's ID for permission filtering
            session_id: Optional session ID for tracking
            
        Returns:
            Dictionary with 'answer' and 'results' (unique document versions)
        """
        self.logger.info(f"Processing query with permissions: user_id={user_id}, query='{query}'")
        
        # Run ONLY retrieval (skip generation — we'll generate once after permission filtering)
        try:
            metadata = self._extract_metadata(query)
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {"answer": "", "results": []}
        
        retrieved_docs, retrieval_time = self._retrieve_documents(query, metadata)
        
        if not retrieved_docs:
            self.logger.warning("No documents retrieved")
            return {"answer": "", "results": []}
        
        # Step 1: Filter by permissions - only keep docs where user_id in allowed_users
        # Documents without allowed_users are EXCLUDED (strict mode - no public fallback)
        permitted_docs = []
        for doc in retrieved_docs:
            allowed_users = doc.meta.get("allowed_users", [])
            
            if not allowed_users:
                # No permissions set — skip (do not treat as public)
                self.logger.debug(
                    f"Filtered out doc {doc.id}: no allowed_users set"
                )
            elif user_id in allowed_users:
                permitted_docs.append(doc)
            else:
                self.logger.debug(
                    f"Filtered out doc {doc.id}: user {user_id} not in allowed_users {allowed_users}"
                )
        
        self.logger.info(
            f"Permission filtering: {len(retrieved_docs)} → {len(permitted_docs)} documents"
        )
        
        # Step 2: Deduplicate by (document_id, document_version_id)
        seen_versions = set()
        unique_results = []
        unique_docs_for_generation = []
        
        for doc in permitted_docs:
            doc_id = doc.meta.get("document_id")
            version_id = doc.meta.get("document_version_id")
            
            # Handle legacy documents without Mayan metadata
            if doc_id is None or version_id is None:
                # Use file_path as fallback for legacy documents
                fallback_key = doc.meta.get("file_path", doc.id)
                if fallback_key not in seen_versions:
                    seen_versions.add(fallback_key)
                    unique_docs_for_generation.append(doc)
                continue
            
            version_key = (doc_id, version_id)
            if version_key not in seen_versions:
                seen_versions.add(version_key)
                unique_results.append({
                    "document_id": doc_id,
                    "document_version_id": version_id,
                })
                unique_docs_for_generation.append(doc)
        
        self.logger.info(f"After deduplication: {len(unique_results)} unique document versions")
        
        # Step 3: Regenerate answer using ALL permitted chunks (not deduplicated)
        # unique_docs_for_generation is deduplicated by version — but the LLM needs
        # all chunks to have enough context for a good answer.
        if permitted_docs:
            replies, _ = self._generate_answer(query, permitted_docs)
            answer = ""
            if replies:
                reply = replies[0]
                if hasattr(reply, 'text'):
                    answer = reply.text
                elif hasattr(reply, 'content'):
                    answer = reply.content
                else:
                    answer = str(reply)
        else:
            answer = ""
            self.logger.warning("No permitted documents found for query")
        
        return {
            "answer": answer,
            "results": unique_results,
        }
