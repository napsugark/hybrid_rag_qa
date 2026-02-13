#!/usr/bin/env python3
"""
Evaluation script for RAG system with Langfuse tracking
Runs test queries and scores them automatically
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import HybridRAGApplication
from evaluation.evaluation_dataset import get_evaluation_dataset
from src import config

# Import Langfuse for evaluation tracking
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("[WARN] Langfuse not available. Install with: poetry add langfuse")


class RAGEvaluator:
    """Evaluate RAG system and track results in Langfuse"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.app = HybridRAGApplication()
        
        # Initialize Langfuse client for evaluation tracking
        if LANGFUSE_AVAILABLE and config.LANGFUSE_ENABLED:
            self.langfuse = Langfuse()
            print("[OK] Langfuse evaluation tracking enabled")
        else:
            self.langfuse = None
            print("[INFO] Running without Langfuse tracking")
        
        self.results = []
    
    def score_metadata_extraction(
        self, 
        extracted: Dict[str, Any], 
        expected: Dict[str, Any]
    ) -> float:
        """
        Score metadata extraction accuracy (0.0 to 1.0)
        
        Args:
            extracted: Metadata extracted by the system
            expected: Expected metadata from evaluation dataset
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not expected:
            return 1.0  # No expectations to check
        
        correct = 0
        total = len(expected)
        
        for key, expected_value in expected.items():
            extracted_value = extracted.get(key)
            
            # Handle None vs missing
            if expected_value is None:
                if extracted_value is None:
                    correct += 1
            elif extracted_value == expected_value:
                correct += 1
            elif key == "company" and isinstance(expected_value, str):
                # Allow partial company name matches (case-insensitive)
                if (extracted_value and 
                    expected_value.lower() in extracted_value.lower()):
                    correct += 0.8  # Partial credit
        
        return correct / total if total > 0 else 1.0
    
    def score_filter_usage(
        self, 
        filters: Dict[str, Any], 
        expected_fields: List[str]
    ) -> float:
        """
        Score whether the right filters were applied
        
        Args:
            filters: Filters applied by the system
            expected_fields: Expected filter fields (e.g., ['meta.year', 'meta.company'])
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not expected_fields:
            return 1.0
        
        if not filters:
            return 0.0
        
        # Extract all field names from nested filter structure
        used_fields = set()
        self._extract_fields(filters, used_fields)
        
        expected_set = set(expected_fields)
        matched = used_fields.intersection(expected_set)
        
        return len(matched) / len(expected_set) if expected_set else 1.0
    
    def _extract_fields(self, obj: Any, fields: set):
        """Recursively extract field names from nested filter structure"""
        if isinstance(obj, dict):
            if "field" in obj:
                fields.add(obj["field"])
            if "conditions" in obj:
                for condition in obj["conditions"]:
                    self._extract_fields(condition, fields)
    
    def score_retrieval_quality(
        self, 
        documents: List[Any],
        expected_criteria: List[str]
    ) -> float:
        """
        Score retrieval quality based on criteria
        
        Args:
            documents: Retrieved documents
            expected_criteria: List of expected retrieval criteria
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not documents:
            return 0.0
        
        # Basic scoring: did we retrieve relevant documents?
        # More sophisticated scoring would check document content
        base_score = min(len(documents) / 5.0, 1.0)  # Expect at least 5 docs
        
        return base_score
    
    def evaluate_query(self, eval_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            eval_item: Evaluation dataset item
            
        Returns:
            Evaluation result with scores
        """
        query = eval_item["query"]
        eval_id = eval_item["id"]
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {eval_id}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Run query through RAG system
            result = self.app.query(query)
            trace_id = result.get("_internal", {}).get("langfuse_trace_id")
            
            # Extract components from result structure
            # Result structure: {"retriever": {"documents": [...]}, "generator": {"replies": [...]}, "metadata": {...}}
            documents = result.get("retriever", {}).get("documents", [])
            replies = result.get("generator", {}).get("replies", [])
            result_metadata = result.get("metadata", {})
            
            # Extract answer text from ChatMessage replies
            if replies:
                # ChatMessage objects have content property
                answer_text = replies[0].content if hasattr(replies[0], 'content') else str(replies[0])
            else:
                answer_text = ""
            
            # Extract from metadata
            filters = result_metadata.get("extracted_filters")
            extracted_metadata = result_metadata.get("extracted_metadata", {})
            
            # Calculate scores
            metadata_score = self.score_metadata_extraction(
                extracted_metadata, 
                eval_item.get("expected_metadata", {})
            )
            
            filter_score = self.score_filter_usage(
                filters,
                eval_item.get("expected_filters", [])
            )
            
            retrieval_score = self.score_retrieval_quality(
                documents,
                eval_item.get("evaluation_criteria", [])
            )
            
            # Overall score (weighted average)
            overall_score = (
                metadata_score * 0.3 +
                filter_score * 0.3 +
                retrieval_score * 0.4
            )
            
            elapsed = time.time() - start_time
            
            eval_result = {
                "eval_id": eval_id,
                "query": query,
                "success": True,
                "scores": {
                    "metadata_extraction": metadata_score,
                    "filter_usage": filter_score,
                    "retrieval_quality": retrieval_score,
                    "overall": overall_score,
                },
                "metadata": {
                    "extracted": extracted_metadata,
                    "expected": eval_item.get("expected_metadata", {}),
                },
                "filters": filters,
                "num_documents": len(documents),
                "answer_length": len(answer_text),
                "answer_preview": answer_text[:200] if answer_text else "",
                "elapsed_time": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Track in Langfuse (note: trace_id not available in current implementation)
            # Scores will be logged separately
            if self.langfuse:
                # self._track_in_langfuse(eval_result)
                self._track_in_langfuse(eval_result, trace_id)
            # Print results
            self._print_result(eval_result)
            
            return eval_result
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                "eval_id": eval_id,
                "query": query,
                "success": False,
                "error": str(e),
                "scores": {
                    "metadata_extraction": 0.0,
                    "filter_usage": 0.0,
                    "retrieval_quality": 0.0,
                    "overall": 0.0,
                },
                "elapsed_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }
    
    def _track_in_langfuse(self, eval_result: Dict[str, Any], trace_id: str):
        """Track evaluation scores in Langfuse"""
        if not trace_id:
            print("[WARN] No Langfuse trace_id found, skipping score logging")
            return
        try:            
            # Log overall score
            self.langfuse.create_score(
                name="overall_score",
                value=eval_result["scores"]["overall"],
                trace_id=trace_id,
                comment=f"{eval_result['eval_id']}: {eval_result['query']}",
                data_type="NUMERIC",
            )
            
            # Log component scores
            self.langfuse.create_score(
                name="metadata_extraction",
                value=eval_result["scores"]["metadata_extraction"],
                trace_id=trace_id,
                comment=f"{eval_result['eval_id']}: metadata extraction accuracy",
                data_type="NUMERIC",
            )
            
            self.langfuse.create_score(
                name="filter_usage",
                value=eval_result["scores"]["filter_usage"],
                trace_id=trace_id,
                comment=f"{eval_result['eval_id']}: filter application correctness",
                data_type="NUMERIC",
            )
            
            self.langfuse.create_score(
                name="retrieval_quality",
                value=eval_result["scores"]["retrieval_quality"],
                trace_id=trace_id,
                comment=f"{eval_result['eval_id']}: document retrieval quality",
                data_type="NUMERIC",
            )
            
        except Exception as e:
            print(f"[WARN] Failed to track in Langfuse: {e}")
    
    def _print_result(self, result: Dict[str, Any]):
        """Print evaluation result"""
        scores = result["scores"]
        
        print(f"\n[SCORES]")
        print(f"  Metadata Extraction: {scores['metadata_extraction']:.2%}")
        print(f"  Filter Usage:        {scores['filter_usage']:.2%}")
        print(f"  Retrieval Quality:   {scores['retrieval_quality']:.2%}")
        print(f"  Overall:             {scores['overall']:.2%}")
        
        print(f"\n[METRICS]")
        print(f"  Documents Retrieved: {result.get('num_documents', 0)}")
        print(f"  Answer Length:       {result.get('answer_length', 0)} chars")
        print(f"  Time:                {result.get('elapsed_time', 0):.2f}s")
        
        if result.get("success"):
            status = "[PASS]" if scores["overall"] >= 0.7 else "[WARN]"
            print(f"\n{status} Result: {'PASS' if scores['overall'] >= 0.7 else 'NEEDS IMPROVEMENT'}")
        else:
            print(f"\n[FAILED] Result: FAILED - {result.get('error')}")
    
    def run_evaluation(self, dataset: List[Dict[str, Any]] = None):
        """
        Run evaluation on entire dataset
        
        Args:
            dataset: Evaluation dataset (uses default if None)
        """
        if dataset is None:
            dataset = get_evaluation_dataset()
        
        print("="*80)
        print(f"RAG SYSTEM EVALUATION")
        print(f"Dataset: {len(dataset)} queries")
        print("="*80)
        
        self.results = []
        
        for item in dataset:
            result = self.evaluate_query(item)
            self.results.append(result)
            time.sleep(1)  # Brief pause between queries
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results()
        
        # Flush Langfuse
        if self.langfuse:
            self.langfuse.flush()
            print("\n[OK] Evaluation results sent to Langfuse")
    
    def _print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]
        
        avg_scores = {
            "metadata_extraction": sum(r["scores"]["metadata_extraction"] for r in successful) / len(successful) if successful else 0,
            "filter_usage": sum(r["scores"]["filter_usage"] for r in successful) / len(successful) if successful else 0,
            "retrieval_quality": sum(r["scores"]["retrieval_quality"] for r in successful) / len(successful) if successful else 0,
            "overall": sum(r["scores"]["overall"] for r in successful) / len(successful) if successful else 0,
        }
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal Queries:  {len(self.results)}")
        print(f"Successful:     {len(successful)}")
        print(f"Failed:         {len(failed)}")
        
        print(f"\nAverage Scores:")
        print(f"  Metadata Extraction: {avg_scores['metadata_extraction']:.2%}")
        print(f"  Filter Usage:        {avg_scores['filter_usage']:.2%}")
        print(f"  Retrieval Quality:   {avg_scores['retrieval_quality']:.2%}")
        print(f"  Overall:             {avg_scores['overall']:.2%}")
        
        passing = sum(1 for r in successful if r["scores"]["overall"] >= 0.7)
        print(f"\nPassing Rate: {passing}/{len(successful)} ({passing/len(successful)*100:.1f}%)" if successful else "\nPassing Rate: N/A")
    
    def _save_results(self):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(self.results),
            "results": self.results,
        }
        
        # Determine logs directory from config (fallback to ./logs)
        logs_dir = getattr(config, "LOGS_DIR", None)
        if logs_dir is None:
            logs_dir = Path("logs")

        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        filepath = logs_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {filepath}")


def main():
    """Main evaluation runner"""
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
