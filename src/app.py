"""
Advanced Hybrid RAG Application — Facade

Thin orchestrator that composes:
  - IndexingPipeline  (src.pipelines.indexing)
  - RetrievalPipeline (src.pipelines.retrieval)

Plus Mayan EDMS integration helpers and cross-cutting concerns
(caching, circuit breakers, rate limiters, logging).
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from haystack import Document
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from .components.query_metadata_extractor import RomanianQueryMetadataExtractor
from .langfuse_tracker import setup_langfuse
from . import config

from .resilience import (
    RetryConfig,
    retry_with_backoff,
    CircuitBreaker,
    RateLimiter,
)
from .cache import CacheManager
from .pipelines.indexing import IndexingPipeline
from .pipelines.retrieval import RetrievalPipeline

logger = logging.getLogger("HybridRAG")


# ======================================================================
# Logging setup
# ======================================================================

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    config.LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"hybrid_rag_{timestamp}.log"

    _logger = logging.getLogger("HybridRAG")
    _logger.setLevel(getattr(logging, config.LOG_LEVEL))
    _logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    _logger.info(f"Logging initialized. Log file: {log_file}")
    return _logger


# ======================================================================
# Main application
# ======================================================================

class HybridRAGApplication:
    """Advanced Hybrid RAG Application — facade over indexing and retrieval pipelines."""

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
            self.logger.info(
                "No Qdrant API key provided — connecting without authentication (local mode)"
            )

        self.logger.info("Configuration:")
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

        # ---- Cross-cutting concerns ----
        self.cache_manager = CacheManager(
            embedding_cache_size=getattr(config, "EMBEDDING_CACHE_SIZE", 1000),
            retrieval_cache_size=getattr(config, "RETRIEVAL_CACHE_SIZE", 500),
            response_cache_size=getattr(config, "RESPONSE_CACHE_SIZE", 200),
            embedding_ttl=getattr(config, "EMBEDDING_CACHE_TTL", 3600),
            retrieval_ttl=getattr(config, "RETRIEVAL_CACHE_TTL", 1800),
            response_ttl=getattr(config, "RESPONSE_CACHE_TTL", 3600),
        )
        self.logger.info("Cache manager initialized")

        self.qdrant_circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(
                config, "QDRANT_CIRCUIT_BREAKER_THRESHOLD", 5
            ),
            recovery_timeout=getattr(
                config, "QDRANT_CIRCUIT_BREAKER_TIMEOUT", 60.0
            ),
            expected_exception=Exception,
            name="Qdrant",
        )
        self.ollama_circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(
                config, "OLLAMA_CIRCUIT_BREAKER_THRESHOLD", 3
            ),
            recovery_timeout=getattr(
                config, "OLLAMA_CIRCUIT_BREAKER_TIMEOUT", 30.0
            ),
            expected_exception=Exception,
            name="Ollama",
        )
        self.logger.info("Circuit breakers initialized")

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

        # ---- Query metadata extractor ----
        self.query_metadata_extractor = RomanianQueryMetadataExtractor(
            llm_type=getattr(config, "LLM_TYPE", "OLLAMA"),
            llm_model=getattr(config, "LLM_MODEL", "llama3.1:8b"),
            llm_url=getattr(config, "LLM_URL", "http://127.0.0.1:11435"),
            llm_api_key=getattr(config, "LLM_API_KEY", None),
            llm_api_version=getattr(config, "LLM_API_VERSION", None),
        )
        self.logger.info("Query metadata extractor initialized")

        # ---- Document store ----
        self.document_store: Optional[QdrantDocumentStore] = None
        self._initialize_document_store()

        # ---- Pipelines ----
        self.indexing = IndexingPipeline(
            document_store=self.document_store,
            qdrant_rate_limiter=self.qdrant_rate_limiter,
            cache_manager=self.cache_manager,
        )

        self.retrieval = RetrievalPipeline(
            document_store=self.document_store,
            qdrant_circuit_breaker=self.qdrant_circuit_breaker,
            ollama_circuit_breaker=self.ollama_circuit_breaker,
            qdrant_rate_limiter=self.qdrant_rate_limiter,
            ollama_rate_limiter=self.ollama_rate_limiter,
            cache_manager=self.cache_manager,
            query_metadata_extractor=self.query_metadata_extractor,
        )

        # Warm up retrieval models
        self.retrieval.warm_up()

        self.logger.info("Hybrid RAG Application initialized successfully")
        self.logger.info("=" * 80)

    # ==================================================================
    # Document store
    # ==================================================================
    def _initialize_document_store(self):
        """Connect to Qdrant with retries."""
        self.logger.info("Connecting to Qdrant...")

        retry_cfg = RetryConfig(
            max_attempts=3, initial_delay=2.0, max_delay=10.0
        )

        @retry_with_backoff(
            config=retry_cfg,
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
            self.logger.info(
                f"Connected to Qdrant collection: {self.collection_name}"
            )
            self.logger.info(f"  - Existing documents: {doc_count}")
        except Exception as e:
            self.logger.error(
                f"Failed to connect to Qdrant after retries: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Could not establish connection to Qdrant. "
                f"Please check if Qdrant is accessible at {self.qdrant_url} "
                f"and verify your API key."
            ) from e

    # ==================================================================
    # Delegate to IndexingPipeline
    # ==================================================================
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        return self.indexing.load_documents_from_folder(folder_path)

    def get_indexed_sources(self) -> set:
        return self.indexing.get_indexed_sources()

    def filter_duplicate_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        return self.indexing.filter_duplicate_documents(documents)

    def index_documents(
        self, documents: List[Document], skip_duplicates: bool = True
    ) -> int:
        return self.indexing.index_documents(documents, skip_duplicates)

    # ==================================================================
    # Delegate to RetrievalPipeline
    # ==================================================================
    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.retrieval.query(query, session_id)

    def query_with_permissions(
        self,
        query: str,
        user_id: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.retrieval.query_with_permissions(query, user_id, session_id)

    # ==================================================================
    # Utilities
    # ==================================================================
    def get_document_count(self) -> int:
        return self.document_store.count_documents()

    def _ensure_collection_exists(self):
        """Recreate collection if it was deleted externally."""
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
            f"Recreating collection '{self.collection_name}' — this will delete all data!"
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

        # Rebuild pipelines with new document store
        self.indexing = IndexingPipeline(
            document_store=self.document_store,
            qdrant_rate_limiter=self.qdrant_rate_limiter,
            cache_manager=self.cache_manager,
        )
        self.retrieval = RetrievalPipeline(
            document_store=self.document_store,
            qdrant_circuit_breaker=self.qdrant_circuit_breaker,
            ollama_circuit_breaker=self.ollama_circuit_breaker,
            qdrant_rate_limiter=self.qdrant_rate_limiter,
            ollama_rate_limiter=self.ollama_rate_limiter,
            cache_manager=self.cache_manager,
            query_metadata_extractor=self.query_metadata_extractor,
        )

    def get_statistics(self) -> Dict[str, Any]:
        docs = self.document_store.filter_documents()
        return {
            "total_documents": len(docs),
            "with_summaries": sum(1 for d in docs if d.meta.get("summary")),
            "with_metadata": sum(1 for d in docs if d.meta.get("entities")),
            "sources": set(
                d.meta.get("file_path", "Unknown") for d in docs
            ),
            "cache_stats": self.get_cache_stats(),
            "resilience_stats": self.get_resilience_stats(),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache_manager.get_all_stats()

    def get_resilience_stats(self) -> Dict[str, Any]:
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

    # ==================================================================
    # Mayan EDMS integration
    # ==================================================================
    def delete_all_versions_of_document(self, document_id: int) -> int:
        """Delete ALL chunks for a document_id (all versions)."""
        self.logger.info(f"Deleting all versions for document_id={document_id}")

        try:
            filters = {
                "field": "meta.document_id",
                "operator": "==",
                "value": document_id,
            }
            docs_to_delete = self.document_store.filter_documents(
                filters=filters
            )

            if not docs_to_delete:
                self.logger.info(
                    f"No existing chunks found for document_id={document_id}"
                )
                return 0

            old_versions = set(
                d.meta.get("document_version_id") for d in docs_to_delete
            )
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
            self.logger.error(
                f"Failed to delete documents: {e}", exc_info=True
            )
            raise

    def delete_chunks_by_content_hash(
        self, content_hash: str, exclude_document_id: int
    ) -> int:
        """Delete orphaned chunks with same content_hash but different document_id."""
        try:
            filters = {
                "field": "meta.content_hash",
                "operator": "==",
                "value": content_hash,
            }
            matching_docs = self.document_store.filter_documents(
                filters=filters
            )

            to_delete = [
                doc
                for doc in matching_docs
                if doc.meta.get("document_id") != exclude_document_id
                and doc.id
            ]

            if not to_delete:
                return 0

            old_doc_ids = set(d.meta.get("document_id") for d in to_delete)
            self.logger.info(
                f"Content hash {content_hash[:12]}... found in {len(to_delete)} "
                f"chunks from old document_id(s) {old_doc_ids} — deleting orphans"
            )

            self.document_store.delete_documents(
                document_ids=[d.id for d in to_delete]
            )
            self.cache_manager.retrieval_cache.invalidate_all()
            return len(to_delete)

        except Exception as e:
            self.logger.warning(f"Content hash dedup check failed: {e}")
            return 0

    def is_document_version_indexed(
        self, document_id: int, document_version_id: int
    ) -> bool:
        """Check if a (document_id, document_version_id) pair already exists."""
        try:
            filters = {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "meta.document_id",
                        "operator": "==",
                        "value": document_id,
                    },
                    {
                        "field": "meta.document_version_id",
                        "operator": "==",
                        "value": document_version_id,
                    },
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
        """Index a document from Mayan EDMS — latest version only."""
        self.logger.info(
            f"Indexing Mayan document: document_id={document_id}, "
            f"document_version_id={document_version_id}, "
            f"allowed_users={allowed_users}"
        )

        self._ensure_collection_exists()

        if self.is_document_version_indexed(document_id, document_version_id):
            self.logger.info(
                f"document_id={document_id}, "
                f"document_version_id={document_version_id} "
                f"already indexed, skipping"
            )
            return {"status": "skipped", "document_count": 0}

        deleted_count = self.delete_all_versions_of_document(document_id)
        if deleted_count > 0:
            self.logger.info(
                f"Replaced {deleted_count} chunks from previous version(s) "
                f"of document_id={document_id}"
            )

        content_hash = document.meta.get("content_hash")
        if content_hash:
            orphan_count = self.delete_chunks_by_content_hash(
                content_hash, document_id
            )
            if orphan_count > 0:
                self.logger.info(
                    f"Cleaned up {orphan_count} orphaned chunks from same file "
                    f"under previous document_id(s)"
                )

        document.meta["document_id"] = document_id
        document.meta["document_version_id"] = document_version_id
        document.meta["allowed_users"] = allowed_users

        doc_count = self.index_documents([document], skip_duplicates=False)
        return {"status": "indexed", "document_count": doc_count}
