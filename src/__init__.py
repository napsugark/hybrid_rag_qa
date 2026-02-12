"""
Core application package

Modules:
- app: Main RAG application facade
- config: Configuration settings
- pipelines: Indexing and retrieval pipeline classes
- integrations: External system clients (Mayan EDMS)
- langfuse_tracker: Langfuse observability integration
"""

from .app import HybridRAGApplication

__all__ = ['HybridRAGApplication']
