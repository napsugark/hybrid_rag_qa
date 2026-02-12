"""
Indexing pipeline — document loading, processing, and storage

Builds the production-grade indexing pipeline:
  type_detector → metadata_enricher → semantic_chunker → boilerplate_filter → summarizer → embedders → writer
"""

import logging
from pathlib import Path
from typing import List, Optional

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
)
from haystack.utils import ComponentDevice
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from ..components.metadata_enricher import MetadataEnricher
from ..components.summarizer import DocumentSummarizer
from .. import config
from ..resilience import RateLimiter

logger = logging.getLogger("HybridRAG.Indexing")


class IndexingPipeline:
    """Handles document loading, processing, and indexing into Qdrant."""

    def __init__(
        self,
        document_store: QdrantDocumentStore,
        qdrant_rate_limiter: RateLimiter,
        cache_manager=None,
    ):
        self.document_store = document_store
        self.qdrant_rate_limiter = qdrant_rate_limiter
        self.cache_manager = cache_manager
        self.pipeline: Optional[Pipeline] = None

        self._build_pipeline()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def _build_pipeline(self):
        """Build production-grade document indexing pipeline."""
        logger.info("Building production-grade indexing pipeline...")
        try:
            self.pipeline = Pipeline()

            # 1. DOCUMENT TYPE DETECTION
            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                from ..components.document_type_detector import DocumentTypeDetector

                self.pipeline.add_component("type_detector", DocumentTypeDetector())
                logger.info("  [OK] Document type detection enabled")

            # 2. METADATA EXTRACTION
            if config.ENABLE_METADATA_EXTRACTION:
                metadata_prompt = None
                prompt_file = getattr(config, "METADATA_EXTRACTION_PROMPT_FILE", None)
                if prompt_file and prompt_file.exists():
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        metadata_prompt = f.read()
                    logger.info(
                        f"  [OK] Loaded metadata extraction prompt from {prompt_file.name}"
                    )
                else:
                    logger.warning(
                        f"  [WARN] Metadata prompt file not found: {prompt_file}"
                    )

                self.pipeline.add_component(
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
                logger.info("  [OK] Metadata extraction enabled")

            # 3. SEMANTIC CHUNKING or Fixed-Size Chunking
            if getattr(config, "USE_SEMANTIC_CHUNKING", True):
                from ..components.semantic_chunker import SemanticDocumentChunker

                self.pipeline.add_component(
                    "semantic_chunker",
                    SemanticDocumentChunker(
                        min_chunk_size=getattr(
                            config, "SEMANTIC_CHUNK_MIN_SIZE", 100
                        ),
                        max_chunk_size=getattr(
                            config, "SEMANTIC_CHUNK_MAX_SIZE", 800
                        ),
                        overlap_size=getattr(config, "SEMANTIC_CHUNK_OVERLAP", 50),
                    ),
                )
                logger.info("  [OK] Semantic chunking enabled (logical sections)")
            else:
                self.pipeline.add_component(
                    "splitter",
                    DocumentSplitter(
                        split_by=config.CHUNK_SPLIT_BY,
                        split_length=config.CHUNK_SIZE,
                        split_overlap=config.CHUNK_OVERLAP,
                    ),
                )
                logger.info("  [WARN] Using legacy fixed-size chunking")

            # 4. BOILERPLATE FILTERING
            if getattr(config, "USE_BOILERPLATE_FILTER", True):
                from ..components.boilerplate_filter import BoilerplateFilter

                self.pipeline.add_component(
                    "boilerplate_filter",
                    BoilerplateFilter(
                        min_boilerplate_score=getattr(
                            config, "BOILERPLATE_MIN_SCORE", 3
                        ),
                        skip_legal_sections=getattr(
                            config, "SKIP_LEGAL_SECTIONS", True
                        ),
                        skip_payment_sections=getattr(
                            config, "SKIP_PAYMENT_SECTIONS", True
                        ),
                    ),
                )
                logger.info("  [OK] Boilerplate filtering enabled")

            # 5. SUMMARIZATION
            if config.ENABLE_SUMMARIZATION:
                self.pipeline.add_component(
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
                logger.info("  [OK] Summarization enabled")

            # 6. EMBEDDERS + WRITER
            self.pipeline.add_component(
                "dense_embedder",
                SentenceTransformersDocumentEmbedder(
                    model=config.DENSE_EMBEDDING_MODEL,
                    device=ComponentDevice.from_str(config.EMBEDDING_DEVICE),
                ),
            )
            self.pipeline.add_component(
                "sparse_embedder",
                FastembedSparseDocumentEmbedder(model=config.SPARSE_EMBEDDING_MODEL),
            )
            self.pipeline.add_component(
                "writer", DocumentWriter(document_store=self.document_store)
            )
            logger.info("  [OK] Dense + Sparse embedders configured")

            # CONNECT COMPONENTS
            current_component = None

            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                current_component = "type_detector"

            if config.ENABLE_METADATA_EXTRACTION:
                if current_component:
                    self.pipeline.connect(current_component, "metadata_enricher")
                current_component = "metadata_enricher"

            if getattr(config, "USE_SEMANTIC_CHUNKING", True):
                if current_component:
                    self.pipeline.connect(current_component, "semantic_chunker")
                current_component = "semantic_chunker"
            else:
                if current_component:
                    self.pipeline.connect(current_component, "splitter")
                current_component = "splitter"

            if getattr(config, "USE_BOILERPLATE_FILTER", True):
                if current_component:
                    self.pipeline.connect(current_component, "boilerplate_filter")
                current_component = "boilerplate_filter"

            if config.ENABLE_SUMMARIZATION:
                if current_component:
                    self.pipeline.connect(current_component, "summarizer")
                current_component = "summarizer"

            if current_component:
                self.pipeline.connect(current_component, "dense_embedder")
            self.pipeline.connect("dense_embedder", "sparse_embedder")
            self.pipeline.connect("sparse_embedder", "writer")

            logger.info("=" * 80)
            logger.info("PRODUCTION-GRADE INDEXING PIPELINE READY:")
            logger.info("  1. Document type detection (invoice/contract/receipt)")
            logger.info("  2. Metadata extraction (from full document)")
            logger.info("  3. Semantic chunking (logical sections)")
            logger.info("  4. Boilerplate filtering (remove legal/payment text)")
            logger.info("  5. Smart summarization")
            logger.info("  6. Hybrid embeddings (dense + sparse)")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to build indexing pipeline: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Load documents (TXT, PDF, MD) from a folder."""
        logger.info(f"Loading documents from: {folder_path}")
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        raw_docs: List[Document] = []

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

        logger.info(f"Total documents loaded: {len(raw_docs)}")
        return raw_docs

    # ------------------------------------------------------------------
    # Duplicate filtering
    # ------------------------------------------------------------------
    def get_indexed_sources(self) -> set:
        """Return set of already-indexed source paths."""
        try:
            all_docs = list(self.document_store.filter_documents())
            sources = {
                doc.meta.get("file_path", doc.meta.get("source", ""))
                for doc in all_docs
            }
            sources.discard("")
            return sources
        except Exception as e:
            logger.warning(f"Could not retrieve indexed sources: {e}")
            return set()

    def filter_duplicate_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """Remove documents whose source is already indexed."""
        indexed_sources = self.get_indexed_sources()
        if not indexed_sources:
            return documents
        new_docs = []
        skipped_sources: set = set()
        for doc in documents:
            source = doc.meta.get("file_path", doc.meta.get("source", ""))
            if source and source in indexed_sources:
                skipped_sources.add(source)
            else:
                new_docs.append(doc)
        if skipped_sources:
            logger.info(
                f"Skipping {len(skipped_sources)} already indexed sources"
            )
        return new_docs

    # ------------------------------------------------------------------
    # Index documents
    # ------------------------------------------------------------------
    def index_documents(
        self, documents: List[Document], skip_duplicates: bool = True
    ) -> int:
        """Run documents through the indexing pipeline and store in Qdrant.

        Returns:
            Total document count after indexing.
        """
        if skip_duplicates:
            documents = self.filter_duplicate_documents(documents)
            if not documents:
                logger.warning("No new documents to index")
                return self.document_store.count_documents()

        logger.info(f"Starting indexing for {len(documents)} documents")

        try:
            # Determine the first component based on configuration
            if getattr(config, "USE_DOCUMENT_TYPE_DETECTION", True):
                input_data = {"type_detector": {"documents": documents}}
            elif config.ENABLE_METADATA_EXTRACTION:
                input_data = {"metadata_enricher": {"documents": documents}}
            elif getattr(config, "USE_SEMANTIC_CHUNKING", True):
                input_data = {"semantic_chunker": {"documents": documents}}
            else:
                input_data = {"splitter": {"documents": documents}}

            with self.qdrant_rate_limiter:
                self.pipeline.run(input_data)

            final_count = self.document_store.count_documents()
            logger.info(f"Indexing complete. Total documents: {final_count}")

            # Invalidate retrieval cache after indexing new documents
            if self.cache_manager:
                self.cache_manager.retrieval_cache.invalidate_all()
                logger.info("Retrieval cache invalidated after indexing")

            return final_count

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to index documents. {len(documents)} documents were not indexed. "
                f"Error: {str(e)}"
            ) from e
