"""
Custom component initialization
"""

from .metadata_enricher import MetadataEnricher
from .summarizer import DocumentSummarizer
from .query_metadata_extractor import RomanianQueryMetadataExtractor
from .document_type_detector import DocumentTypeDetector
from .semantic_chunker import SemanticDocumentChunker
from .boilerplate_filter import BoilerplateFilter

__all__ = [
    "MetadataEnricher",
    "DocumentSummarizer",
    "RomanianQueryMetadataExtractor",
    "DocumentTypeDetector",
    "SemanticDocumentChunker",
    "BoilerplateFilter",
]
