"""
Retrieval pipeline — query processing, document retrieval, and answer generation

Architecture (deepset-style):
  sparse_embedder + dense_embedder → sparse_retriever + dense_retriever → joiner → reranker → generator
"""

import logging
import time
from typing import List, Optional, Dict, Any

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret, ComponentDevice
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)

from ..components.query_metadata_extractor import RomanianQueryMetadataExtractor
from ..langfuse_tracker import setup_langfuse, get_observe_decorator
from ..utils import format_time
from .. import config
from ..resilience import CircuitBreaker, RateLimiter
from ..cache import CacheManager

logger = logging.getLogger("HybridRAG.Retrieval")

# Langfuse decorator
langfuse_client = setup_langfuse()
observe = get_observe_decorator()


class RetrievalPipeline:
    """Handles query metadata extraction, document retrieval, and answer generation."""

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        qdrant_circuit_breaker: CircuitBreaker,
        ollama_circuit_breaker: CircuitBreaker,
        qdrant_rate_limiter: RateLimiter,
        ollama_rate_limiter: RateLimiter,
        cache_manager: CacheManager,
        query_metadata_extractor: RomanianQueryMetadataExtractor,
    ):
        self.document_store = document_store
        self.qdrant_circuit_breaker = qdrant_circuit_breaker
        self.ollama_circuit_breaker = ollama_circuit_breaker
        self.qdrant_rate_limiter = qdrant_rate_limiter
        self.ollama_rate_limiter = ollama_rate_limiter
        self.cache_manager = cache_manager
        self.query_metadata_extractor = query_metadata_extractor

        self.retrieval_pipeline: Optional[Pipeline] = None
        self.generation_pipeline: Optional[Pipeline] = None

        self._build_pipelines()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def _build_pipelines(self):
        """Build retrieval and generation pipelines (deepset-style)."""
        logger.info("Building retrieval and generation pipelines (deepset-style)...")

        try:
            # ---- RETRIEVAL PIPELINE ----
            self.retrieval_pipeline = Pipeline()

            self.dense_embedder = SentenceTransformersTextEmbedder(
                model=config.DENSE_EMBEDDING_MODEL,
                device=ComponentDevice.from_str(config.EMBEDDING_DEVICE),
                prefix=config.DENSE_EMBEDDING_PREFIX,
            )
            self.retrieval_pipeline.add_component(
                "dense_embedder", self.dense_embedder
            )

            self.retrieval_pipeline.add_component(
                "sparse_embedder",
                FastembedSparseTextEmbedder(model=config.SPARSE_EMBEDDING_MODEL),
            )

            retriever_top_k = (
                config.TOP_K * 2 if config.USE_RERANKER else config.TOP_K
            )

            self.retrieval_pipeline.add_component(
                "sparse_retriever",
                QdrantSparseEmbeddingRetriever(
                    document_store=self.document_store, top_k=retriever_top_k
                ),
            )
            self.retrieval_pipeline.add_component(
                "dense_retriever",
                QdrantEmbeddingRetriever(
                    document_store=self.document_store, top_k=retriever_top_k
                ),
            )
            self.retrieval_pipeline.add_component(
                "document_joiner", DocumentJoiner(join_mode="concatenate")
            )

            if config.USE_RERANKER:
                self.retrieval_pipeline.add_component(
                    "reranker",
                    SentenceTransformersSimilarityRanker(
                        model=config.RERANKER_MODEL, top_k=config.RERANKER_TOP_K
                    ),
                )

            # Connect
            self.retrieval_pipeline.connect(
                "sparse_embedder.sparse_embedding",
                "sparse_retriever.query_sparse_embedding",
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

            logger.info("  [OK] Sparse retriever (BM25-like) configured")
            logger.info("  [OK] Dense retriever (embedding) configured")
            logger.info("  [OK] DocumentJoiner (concatenate mode) configured")
            if config.USE_RERANKER:
                logger.info(f"  [OK] Reranker configured: {config.RERANKER_MODEL}")

            # ---- GENERATION PIPELINE ----
            rag_system_prompt = (
                config.RAG_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
                if config.RAG_SYSTEM_PROMPT_FILE.exists()
                else "You are a helpful assistant."
            )
            rag_user_prompt = (
                config.ACTIVE_RAG_USER_PROMPT.read_text(encoding="utf-8")
                if config.ACTIVE_RAG_USER_PROMPT.exists()
                else "Answer: {{ query }}"
            )

            template = [
                ChatMessage.from_system(rag_system_prompt),
                ChatMessage.from_user(rag_user_prompt),
            ]
            self.generation_pipeline = Pipeline()
            self.generation_pipeline.add_component(
                "prompt_builder",
                ChatPromptBuilder(
                    template=template,
                    required_variables=["query", "documents"],
                ),
            )

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
                logger.info(
                    f"Generation pipeline using Ollama: {config.LLM_MODEL}"
                )
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
                logger.info(
                    f"Generation pipeline using Azure OpenAI: {config.LLM_MODEL}"
                )
            else:
                raise ValueError(f"Unsupported LLM_TYPE: {config.LLM_TYPE}")

            self.generation_pipeline.connect(
                "prompt_builder.prompt", "generator.messages"
            )
            logger.info("Retrieval and generation pipelines built successfully")

        except Exception:
            logger.error("Failed to build query pipelines", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    def warm_up(self):
        """Download / load models eagerly so first query is fast."""
        logger.info("Warming up pipelines (downloading/loading models)...")
        try:
            self.retrieval_pipeline.warm_up()
            logger.info("  [OK] Retrieval pipeline warmed up")
        except Exception as e:
            logger.warning(f"  [WARN] Retrieval pipeline warm_up failed: {e}")
        try:
            self.generation_pipeline.warm_up()
            logger.info("  [OK] Generation pipeline warmed up")
        except Exception as e:
            logger.warning(f"  [WARN] Generation pipeline warm_up failed: {e}")

    # ------------------------------------------------------------------
    # Helper — recursive numeric coercion for filter trees
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric_filter_values(
        conditions: list, numeric_fields: set
    ) -> None:
        """Convert string digits → numbers **only** for known numeric fields.

        Walks nested AND/OR condition trees so that identifier fields like
        ``invoice_number`` are never accidentally cast to int.
        """
        for cond in conditions:
            # Nested group (AND / OR)
            if "conditions" in cond:
                RetrievalPipeline._coerce_numeric_filter_values(
                    cond["conditions"], numeric_fields
                )
                continue
            field = cond.get("field", "")
            value = cond.get("value")
            if field in numeric_fields and isinstance(value, str):
                try:
                    cond["value"] = float(value) if "." in value else int(value)
                except ValueError:
                    pass  # leave as-is

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------
    @observe(name="metadata_extraction")
    def extract_metadata(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query."""
        extraction_start = time.time()
        query_metadata = self.query_metadata_extractor.run(query=query)

        extracted_filters = query_metadata.get("filters")
        search_query = query_metadata.get("search_query", query)
        extracted_metadata = query_metadata.get("metadata", {})
        extraction_time = time.time() - extraction_start

        if extracted_filters:
            logger.info(f"Extracted filters: {extracted_filters}")
            logger.info(f"Extracted metadata: {extracted_metadata}")
        else:
            logger.info("No filters extracted, using full semantic search")

        # Normalize numeric filters — only for fields that are truly numeric
        # in Qdrant (e.g. amount). Identifier fields like invoice_number are
        # stored as strings even when they look like numbers.
        _NUMERIC_FIELDS = {"meta.amount", "meta.year", "meta.month", "meta.day"}
        if extracted_filters:
            self._coerce_numeric_filter_values(
                extracted_filters.get("conditions", []), _NUMERIC_FIELDS
            )

        return {
            "filters": extracted_filters,
            "metadata": extracted_metadata,
            "search_query": search_query,
            "extraction_time": extraction_time,
        }

    # ------------------------------------------------------------------
    # Score filtering / deduplication
    # ------------------------------------------------------------------
    def _deduplicate_and_limit(
        self,
        retrieved_docs: List[Document],
        query: str,
        extracted_filters: Optional[Dict],
    ) -> List[Document]:
        """Deduplicate and limit results (deepset-style — no aggressive score filtering)."""
        MAX_DOCS = (
            config.RERANKER_TOP_K if config.USE_RERANKER else config.TOP_K
        )

        seen_ids: set = set()
        unique_docs: List[Document] = []
        for d in retrieved_docs:
            doc_id = d.id if d.id else hash(d.content[:100])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(d)

        logger.info(
            f"Documents after deduplication: {len(unique_docs)} "
            f"(from {len(retrieved_docs)})"
        )

        sorted_docs = sorted(
            unique_docs, key=lambda d: getattr(d, "score", 0), reverse=True
        )[:MAX_DOCS]

        return sorted_docs

    # ------------------------------------------------------------------
    # Document retrieval
    # ------------------------------------------------------------------
    @observe(name="document_retrieval")
    def retrieve_documents(
        self, query: str, metadata: Dict[str, Any]
    ) -> tuple[List[Document], float]:
        """Retrieve documents using deepset-style architecture."""
        retrieval_start = time.time()

        original_query = query
        extracted_filters = metadata.get("filters")

        logger.info(f"Retrieving with ORIGINAL query: '{original_query}'")
        if extracted_filters:
            logger.info(f"Additional filters: {extracted_filters}")

        # Check cache
        cached_docs = self.cache_manager.retrieval_cache.get(
            query=original_query, filters=extracted_filters, top_k=config.TOP_K
        )
        if cached_docs is not None:
            logger.info(f"Retrieved {len(cached_docs)} documents from cache")
            return cached_docs, time.time() - retrieval_start

        # Build retrieval inputs
        retrieval_inputs: Dict[str, Any] = {
            "dense_embedder": {"text": original_query},
            "sparse_embedder": {"text": original_query},
        }

        if extracted_filters:
            retrieval_inputs["sparse_retriever"] = {"filters": extracted_filters}
            retrieval_inputs["dense_retriever"] = {"filters": extracted_filters}

        if config.USE_RERANKER:
            retrieval_inputs["reranker"] = {"query": original_query}

        # Run retrieval with rate limiting and circuit breaker
        try:
            with self.qdrant_rate_limiter:
                retrieval_result = self.qdrant_circuit_breaker.call(
                    self.retrieval_pipeline.run, retrieval_inputs
                )
        except Exception as e:
            logger.warning(f"Retrieval with filters failed: {e}")

            # FALLBACK: Retry WITHOUT filters
            if extracted_filters:
                logger.info("Retrying retrieval WITHOUT filters...")
                retrieval_inputs_no_filters: Dict[str, Any] = {
                    "dense_embedder": {"text": original_query},
                    "sparse_embedder": {"text": original_query},
                }
                if config.USE_RERANKER:
                    retrieval_inputs_no_filters["reranker"] = {
                        "query": original_query
                    }

                try:
                    with self.qdrant_rate_limiter:
                        retrieval_result = self.qdrant_circuit_breaker.call(
                            self.retrieval_pipeline.run,
                            retrieval_inputs_no_filters,
                        )
                    logger.info("Retrieval succeeded without filters")
                except Exception as e2:
                    logger.error(
                        f"Retrieval failed even without filters: {e2}",
                        exc_info=True,
                    )
                    return [], time.time() - retrieval_start
            else:
                logger.error(f"Retrieval failed: {e}", exc_info=True)
                return [], time.time() - retrieval_start

        # Get results
        if config.USE_RERANKER:
            retrieved_docs = retrieval_result.get("reranker", {}).get(
                "documents", []
            )
        else:
            retrieved_docs = retrieval_result.get("document_joiner", {}).get(
                "documents", []
            )

        retrieval_time = time.time() - retrieval_start
        logger.info(
            f"Retrieved {len(retrieved_docs)} raw documents in {retrieval_time:.2f}s"
        )

        filtered_docs = self._deduplicate_and_limit(
            retrieved_docs, query, extracted_filters
        )
        logger.info(
            f"Final documents after processing: {len(filtered_docs)}"
        )

        # FALLBACK: If filters produced 0 results, retry WITHOUT filters
        if not filtered_docs and extracted_filters:
            logger.warning(
                f"Filters returned 0 results. Retrying WITHOUT filters. "
                f"Original filters: {extracted_filters}"
            )
            retrieval_inputs_no_filters = {
                "dense_embedder": {"text": original_query},
                "sparse_embedder": {"text": original_query},
            }
            if config.USE_RERANKER:
                retrieval_inputs_no_filters["reranker"] = {
                    "query": original_query
                }

            try:
                with self.qdrant_rate_limiter:
                    retrieval_result_fallback = self.qdrant_circuit_breaker.call(
                        self.retrieval_pipeline.run,
                        retrieval_inputs_no_filters,
                    )

                if config.USE_RERANKER:
                    fallback_docs = retrieval_result_fallback.get(
                        "reranker", {}
                    ).get("documents", [])
                else:
                    fallback_docs = retrieval_result_fallback.get(
                        "document_joiner", {}
                    ).get("documents", [])

                filtered_docs = self._deduplicate_and_limit(
                    fallback_docs, query, None
                )
                logger.info(
                    f"Fallback retrieval (no filters) returned "
                    f"{len(filtered_docs)} documents"
                )
            except Exception as e_fallback:
                logger.error(
                    f"Fallback retrieval also failed: {e_fallback}",
                    exc_info=True,
                )

        # Cache results
        if filtered_docs:
            self.cache_manager.retrieval_cache.put(
                query=original_query,
                documents=filtered_docs,
                filters=extracted_filters,
                top_k=config.TOP_K,
            )

        return filtered_docs, retrieval_time

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------
    @observe(name="answer_generation")
    def generate_answer(
        self, query: str, retrieved_docs: List[Document]
    ) -> tuple[List[ChatMessage], float]:
        """Generate answer from retrieved documents."""
        generation_start = time.time()

        # Check cache
        cached_response = self.cache_manager.response_cache.get(
            query, retrieved_docs
        )
        if cached_response is not None:
            logger.info("Using cached LLM response")
            return cached_response, time.time() - generation_start

        generation_inputs = {
            "prompt_builder": {"query": query, "documents": retrieved_docs}
        }

        try:
            if config.LLM_TYPE == "OLLAMA":
                with self.ollama_rate_limiter:
                    generation_result = self.ollama_circuit_breaker.call(
                        self.generation_pipeline.run, generation_inputs
                    )
            else:
                generation_result = self.generation_pipeline.run(
                    generation_inputs
                )

            replies = generation_result.get("generator", {}).get("replies", [])

            if replies:
                self.cache_manager.response_cache.put(
                    query, retrieved_docs, replies
                )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            replies = [
                ChatMessage.from_assistant(
                    "I apologize, but I'm having trouble generating a response "
                    "right now. Please try again in a moment."
                )
            ]

        generation_time = time.time() - generation_start
        return replies, generation_time

    # ------------------------------------------------------------------
    # Query orchestration
    # ------------------------------------------------------------------
    def _empty_response(
        self, metadata: Dict[str, Any], total_start: float
    ) -> Dict[str, Any]:
        """Return empty response when no documents found."""
        logger.warning("No documents retrieved. Skipping generation.")
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
        """Build final response structure."""
        total_time = time.time() - total_start

        logger.info(
            f"Timing - Retrieval: {format_time(retrieval_time)} | "
            f"Generation: {format_time(generation_time)} | "
            f"Total: {format_time(total_time)}"
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
    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main query orchestrator."""
        logger.info(f"Processing query: {query}")
        total_start = time.time()

        # Get Langfuse trace ID
        trace_id = None
        if langfuse_client and config.LANGFUSE_ENABLED:
            try:
                trace_id = langfuse_client.get_current_trace_id()
            except Exception as e:
                logger.debug(f"Could not get trace ID: {e}")

        try:
            metadata = self.extract_metadata(query)
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            error_resp: Dict[str, Any] = {
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

        retrieved_docs, retrieval_time = self.retrieve_documents(query, metadata)

        if not retrieved_docs:
            empty_resp = self._empty_response(metadata, total_start)
            if trace_id:
                empty_resp["_internal"] = {"langfuse_trace_id": trace_id}
            return empty_resp

        replies, generation_time = self.generate_answer(query, retrieved_docs)

        response = self._build_response(
            metadata,
            retrieved_docs,
            replies,
            total_start,
            retrieval_time,
            generation_time,
        )

        if trace_id:
            response["_internal"] = {"langfuse_trace_id": trace_id}

        return response

    def query_with_permissions(
        self,
        query: str,
        user_id: int,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query with Mayan EDMS permission filtering.

        Only returns documents where *user_id* is in ``allowed_users``.
        The ``results`` list only includes documents above the reranker
        relevance threshold so that Mayan doesn't display irrelevant items.
        """
        logger.info(
            f"Processing query with permissions: user_id={user_id}, query='{query}'"
        )

        try:
            metadata = self.extract_metadata(query)
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {"answer": "", "results": []}

        retrieved_docs, _ = self.retrieve_documents(query, metadata)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return {"answer": "", "results": []}

        # Apply relevance score threshold — drop low-confidence documents
        min_score = getattr(config, "MAYAN_RESULTS_MIN_SCORE", 0.3)
        relevant_docs = [
            doc for doc in retrieved_docs
            if getattr(doc, "score", None) is None or doc.score >= min_score
        ]
        if relevant_docs:
            logger.info(
                f"Score threshold ({min_score}): {len(retrieved_docs)} → "
                f"{len(relevant_docs)} documents"
            )
        else:
            # If threshold is too aggressive, fall back to top-1
            relevant_docs = retrieved_docs[:1]
            logger.warning(
                f"All docs below score threshold {min_score}; "
                f"keeping top-1 (score={getattr(relevant_docs[0], 'score', 'N/A')})"
            )

        # Filter by permissions
        permitted_docs: List[Document] = []
        for doc in relevant_docs:
            allowed_users = doc.meta.get("allowed_users", [])
            if not allowed_users:
                logger.debug(
                    f"Filtered out doc {doc.id}: no allowed_users set"
                )
            elif user_id in allowed_users:
                permitted_docs.append(doc)
            else:
                logger.debug(
                    f"Filtered out doc {doc.id}: user {user_id} "
                    f"not in allowed_users {allowed_users}"
                )

        logger.info(
            f"Permission filtering: {len(relevant_docs)} → "
            f"{len(permitted_docs)} documents"
        )

        # Deduplicate by (document_id, document_version_id)
        seen_versions: set = set()
        unique_results: List[Dict[str, Any]] = []
        unique_docs_for_generation: List[Document] = []

        for doc in permitted_docs:
            doc_id = doc.meta.get("document_id")
            version_id = doc.meta.get("document_version_id")

            if doc_id is None or version_id is None:
                fallback_key = doc.meta.get("file_path", doc.id)
                if fallback_key not in seen_versions:
                    seen_versions.add(fallback_key)
                    unique_docs_for_generation.append(doc)
                continue

            version_key = (doc_id, version_id)
            if version_key not in seen_versions:
                seen_versions.add(version_key)
                unique_results.append(
                    {
                        "document_id": doc_id,
                        "document_version_id": version_id,
                    }
                )
                unique_docs_for_generation.append(doc)

        logger.info(
            f"After deduplication: {len(unique_results)} unique document versions"
        )

        # Generate answer from deduplicated permitted chunks
        if unique_docs_for_generation:
            replies, _ = self.generate_answer(query, unique_docs_for_generation)
            answer = ""
            if replies:
                reply = replies[0]
                if hasattr(reply, "text"):
                    answer = reply.text
                elif hasattr(reply, "content"):
                    answer = reply.content
                else:
                    answer = str(reply)
        else:
            answer = ""
            logger.warning("No permitted documents found for query")

        return {"answer": answer, "results": unique_results}
