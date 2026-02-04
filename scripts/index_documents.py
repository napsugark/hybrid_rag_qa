#!/usr/bin/env python3
"""
Index documents to Qdrant with hybrid search support
Includes metadata enrichment and summarization
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import HybridRAGApplication
from src import config


def main():
    """Index documents from the documents_ro folder"""
    logger = logging.getLogger("HybridRAG")
    
    logger.info("="*80)
    logger.info("ADVANCED HYBRID RAG - DOCUMENT INDEXING")
    logger.info("="*80)
    logger.info("")
    logger.info("Features enabled:")
    logger.info(f"  [ENABLED] Hybrid Search (Sparse + Dense Embeddings)")
    logger.info(f"  [{'ENABLED' if config.ENABLE_METADATA_EXTRACTION else 'DISABLED'}] Metadata Enrichment")
    logger.info(f"  [{'ENABLED' if config.ENABLE_SUMMARIZATION else 'DISABLED'}] Document Summarization")
    logger.info(f"  [{'ENABLED' if config.USE_RERANKER else 'DISABLED'}] Reranking")
    logger.info("")
    
    # Initialize application
    logger.info("Initializing application...")
    app = HybridRAGApplication()
    
    # Check existing documents
    existing_count = app.get_document_count()
    logger.info(f"Current documents in Qdrant: {existing_count}")
    
    if existing_count > 0:
        logger.info("Duplicate detection enabled - will skip already indexed sources")
    
    # Check documents path
    if not config.DOCUMENTS_DIR.exists():
        logger.error(f"Documents folder not found at {config.DOCUMENTS_DIR}")
        logger.error("Please create the folder and add your documents (TXT, PDF, MD)")
        return
    
    # Load documents
    logger.info(f"Loading documents from {config.DOCUMENTS_DIR}...")
    docs = app.load_documents_from_folder(str(config.DOCUMENTS_DIR))
    
    if not docs:
        logger.warning("No documents found to index!")
        return
    
    logger.info(f"Loaded {len(docs)} documents")
    if config.ENABLE_METADATA_EXTRACTION or config.ENABLE_SUMMARIZATION:
        logger.info("Metadata extraction and summarization enabled - this may take several minutes")
    
    # Index documents (with automatic duplicate detection)
    logger.info("")
    final_count = app.index_documents(docs, skip_duplicates=True)
    
    # Show statistics
    stats = app.get_statistics()
    
    logger.info("")
    logger.info("="*80)
    logger.info("INDEXING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total documents in Qdrant: {final_count}")
    logger.info(f"New documents added: {final_count - existing_count}")
    logger.info(f"Documents with summaries: {stats['with_summaries']}")
    logger.info(f"Documents with metadata: {stats['with_metadata']}")
    logger.info(f"Unique sources: {len(stats['sources'])}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
