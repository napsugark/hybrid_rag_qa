"""
Tests for src/config.py — Pydantic BaseSettings configuration.
"""

import os
from unittest.mock import patch

import pytest


class TestSettings:
    """Verify settings load correctly from env and defaults."""

    def test_defaults_load(self):
        """Settings should instantiate with sane defaults."""
        from src.config import Settings

        s = Settings()
        assert s.QDRANT_COLLECTION == "rag_hybrid_arcticl2_v11"
        assert s.QDRANT_EMBEDDING_DIM == 1024
        assert s.CHUNK_SIZE == 300
        assert s.TOP_K == 5

    def test_model_selection_azure(self):
        """Azure OpenAI selection should populate LLM_* fields."""
        from src.config import Settings

        s = Settings(
            MODEL_TO_USE="AZURE_OPENAI",
            AZURE_OPENAI_ENDPOINT="https://test.openai.azure.com",
            AZURE_OPENAI_API_KEY="test-key",
        )
        assert s.LLM_TYPE == "AZURE_OPENAI"
        assert s.LLM_URL == "https://test.openai.azure.com"
        assert s.LLM_API_KEY == "test-key"

    def test_model_selection_ollama(self):
        """Ollama selection should populate LLM_* fields."""
        from src.config import Settings

        s = Settings(MODEL_TO_USE="OLLAMA", OLLAMA_URL="http://localhost:11434")
        assert s.LLM_TYPE == "OLLAMA"
        assert s.LLM_URL == "http://localhost:11434"
        assert s.LLM_API_KEY is None

    def test_invalid_model_selection_raises(self):
        """Unknown MODEL_TO_USE should raise ValueError."""
        from src.config import Settings

        with pytest.raises(ValueError, match="Unknown MODEL_TO_USE"):
            Settings(MODEL_TO_USE="UNKNOWN_PROVIDER")

    def test_langfuse_enabled_derived(self):
        """LANGFUSE_ENABLED should be True only when public key is set."""
        from src.config import Settings

        s = Settings(LANGFUSE_PUBLIC_KEY="")
        assert s.LANGFUSE_ENABLED is False

        s2 = Settings(LANGFUSE_PUBLIC_KEY="pk-lf-xxx")
        assert s2.LANGFUSE_ENABLED is True

    def test_backward_compat_module_getattr(self):
        """config.QDRANT_COLLECTION should still work at module level."""
        from src import config

        assert config.QDRANT_COLLECTION == "rag_hybrid_arcticl2_v11"
        assert isinstance(config.CHUNK_SIZE, int)
