"""
Tests for src/pipelines/retrieval.py — RetrievalPipeline.
"""

import time
from unittest.mock import MagicMock

import pytest


class TestRetrievalPipeline:
    """Unit tests for RetrievalPipeline."""

    def test_empty_response_structure(self):
        """_empty_response should return a well-formed dict."""
        from src.pipelines.retrieval import RetrievalPipeline

        pipeline = RetrievalPipeline.__new__(RetrievalPipeline)

        metadata = {
            "filters": None,
            "metadata": {},
            "search_query": "test query",
            "extraction_time": 0.01,
        }
        resp = pipeline._empty_response(metadata, time.time())

        assert resp["retriever"]["documents"] == []
        assert len(resp["generator"]["replies"]) == 1
        assert "metadata" in resp
        assert resp["metadata"]["documents_retrieved"] == 0
        assert resp["metadata"]["extracted_filters"] is None
