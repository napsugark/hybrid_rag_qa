"""
Shared pytest fixtures for the hybrid RAG test suite.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_document_store():
    """A mock QdrantDocumentStore for unit tests that don't need Qdrant."""
    store = MagicMock()
    store.count_documents.return_value = 0
    store.filter_documents.return_value = []
    return store


@pytest.fixture
def sample_documents():
    """A handful of Haystack Document objects for testing."""
    from haystack import Document

    return [
        Document(
            content="Factură AVE BIHOR nr. 12345 din 15.01.2025, total 1500 RON.",
            meta={
                "file_path": "factura_ave.pdf",
                "company": "AVE BIHOR",
                "document_type": "factura",
                "year": 2025,
                "month": 1,
            },
        ),
        Document(
            content="Contract de prestări servicii între SC Alpha SRL și SC Beta SA.",
            meta={
                "file_path": "contract_alpha.pdf",
                "company": "Alpha SRL",
                "document_type": "contract",
                "year": 2024,
            },
        ),
    ]


@pytest.fixture
def prompts_dir(tmp_path: Path):
    """Create a temporary prompts directory with minimal prompt files."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "05_rag_system.txt").write_text("You are a helpful assistant.", encoding="utf-8")
    (prompts / "07_rag_user_concise.txt").write_text(
        "Answer the question based on the documents.\nDocuments: {{ documents }}\nQuestion: {{ query }}",
        encoding="utf-8",
    )
    return prompts
