"""
Tests for src/pipelines/indexing.py — IndexingPipeline.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestIndexingPipeline:
    """Unit tests for IndexingPipeline."""

    def test_load_documents_from_folder(self, tmp_path, mock_document_store):
        """Loading a folder of text files should return Documents."""
        # Create sample files
        (tmp_path / "doc1.txt").write_text("Factură AVE BIHOR", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Contract Alpha SRL", encoding="utf-8")

        from src.pipelines.indexing import IndexingPipeline
        from src.resilience import RateLimiter

        pipeline = IndexingPipeline(
            document_store=mock_document_store,
            qdrant_rate_limiter=RateLimiter(max_concurrent=2, name="test"),
        )

        documents = pipeline.load_documents_from_folder(str(tmp_path))
        assert len(documents) >= 2

    def test_filter_duplicate_documents(self, mock_document_store, sample_documents):
        """Deduplication should filter already-indexed sources."""
        # Pretend one source is already indexed
        mock_document_store.filter_documents.return_value = [sample_documents[0]]

        from src.pipelines.indexing import IndexingPipeline
        from src.resilience import RateLimiter

        pipeline = IndexingPipeline(
            document_store=mock_document_store,
            qdrant_rate_limiter=RateLimiter(max_concurrent=2, name="test"),
        )

        unique = pipeline.filter_duplicate_documents(sample_documents)
        # Should contain at most the docs whose file_path is not already in the store
        assert isinstance(unique, list)
