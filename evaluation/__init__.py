"""
Evaluation module for RAG system

Components:
- evaluation_dataset.py: Test cases with expected outputs
- run_evaluation.py: Automated evaluation runner with Langfuse tracking
"""

from .evaluation_dataset import get_evaluation_dataset
from .run_evaluation import RAGEvaluator

__all__ = ['get_evaluation_dataset', 'RAGEvaluator']
