"""
Tests for src/pipelines/retrieval.py — RetrievalPipeline.
"""

from unittest.mock import MagicMock

import pytest


class TestRetrievalPipeline:
    """Unit tests for RetrievalPipeline."""

    def test_empty_response_structure(self):
        """_empty_response should return a well-formed dict."""
        from src.pipelines.retrieval import RetrievalPipeline
        from src.resilience import CircuitBreaker, RateLimiter
        from src.cache import CacheManager
        from src.components.query_metadata_extractor import RomanianQueryMetadataExtractor

        pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
        # Call the static-ish helper directly
        resp = pipeline._empty_response("test query", "no docs")
        assert resp["query"] == "test query"
        assert resp["answer"] == "no docs"
        assert resp["documents"] == []
        assert "timings" in resp
