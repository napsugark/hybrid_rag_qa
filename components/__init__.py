"""
Custom component initialization
"""

from .metadata_enricher import MetadataEnricher
from .summarizer import DocumentSummarizer
from .query_metadata_extractor import RomanianQueryMetadataExtractor

__all__ = ["MetadataEnricher", "DocumentSummarizer", "RomanianQueryMetadataExtractor"]
