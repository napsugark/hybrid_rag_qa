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
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from .utils import format_time
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)

from components.metadata_enricher import MetadataEnricher
from components.summarizer import DocumentSummarizer
from components.query_metadata_extractor import RomanianQueryMetadataExtractor
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

# Get observe decorator and Langfuse client
observe = get_observe_decorator()
langfuse_client = setup_langfuse()

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

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "Qdrant credentials not found. Set QDRANT_URL and QDRANT_API_KEY "
                "in .env or pass them as arguments."
            )

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
            return self.qdrant_circuit_breaker.call(
                lambda: QdrantDocumentStore(
                    url=self.qdrant_url,
                    api_key=Secret.from_token(self.qdrant_api_key),
                    index=self.collection_name,
                    embedding_dim=config.QDRANT_EMBEDDING_DIM,
                    use_sparse_embeddings=True,
                    recreate_index=False,
                    return_embedding=False,
                    wait_result_from_api=True,
                )
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
                from components.document_type_detector import DocumentTypeDetector
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
                from components.semantic_chunker import SemanticDocumentChunker
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
                from components.boilerplate_filter import BoilerplateFilter
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
                    TransformersSimilarityRanker(
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

    def recreate_collection(self):
        self.logger.warning(
            f"Recreating collection '{self.collection_name}' - this will delete all data!"
        )
        self.document_store = QdrantDocumentStore(
            url=self.qdrant_url,
            api_key=Secret.from_token(self.qdrant_api_key),
            index=self.collection_name,
            embedding_dim=config.QDRANT_EMBEDDING_DIM,
            use_sparse_embeddings=True,
            recreate_index=True,
            return_embedding=False,
            wait_result_from_api=True,
        )
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
