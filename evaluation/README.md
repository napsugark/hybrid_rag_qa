# Evaluation Directory

Automated evaluation system for the RAG pipeline with Langfuse tracking.

## Components

### evaluation_dataset.py
Defines test cases with:
- Test queries in Romanian
- Expected metadata extraction
- Expected filters
- Reference answers
- Evaluation criteria

### run_evaluation.py
Evaluation runner that:
- Runs test queries through the RAG system
- Scores metadata extraction (0-1)
- Scores filter usage (0-1)
- Scores retrieval quality (0-1)
- Tracks all scores in Langfuse

## Usage

```bash
# Run full evaluation
poetry run python evaluation/run_evaluation.py

# Check results in Langfuse UI
# Visit: https://cloud.langfuse.com
```

## Scoring Dimensions

1. **Metadata Extraction Score** (0-1)
   - Measures accuracy of extracted metadata vs expected
   - Checks year, month, day, company, client, etc.

2. **Filter Usage Score** (0-1)
   - Checks if correct filters were applied
   - Validates field, operator, and value

3. **Retrieval Quality Score** (0-1)
   - Measures document retrieval effectiveness
   - Based on number of relevant documents found

## Adding Test Cases

Edit `evaluation_dataset.py`:

```python
{
    "id": "eval_009",
    "query": "your test query",
    "expected_metadata": {
        "year": 2025,
        "month": 1
    },
    "expected_filters": [
        {"field": "meta.year", "operator": "==", "value": 2025}
    ],
    "reference_answer": "Expected answer pattern",
    "evaluation_criteria": "What to check"
}
```

## Results

Results are stored in:
- `results/` directory (JSON files)
- Langfuse platform (with trace IDs)
