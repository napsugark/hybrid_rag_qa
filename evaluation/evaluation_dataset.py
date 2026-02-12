"""
Evaluation dataset for RAG system
Contains test questions with expected metadata and reference answers
"""

from typing import List, Dict, Any

# Evaluation dataset: questions with expected outputs for testing
EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "id": "eval_edge_1001",
        "query": "facturi AVE",
        "expected_metadata": {
            "company": "AVE",
            "document_type": "factura",
        },
        "expected_filters": ["meta.company", "meta.document_type"],
        "reference_answer": "Invoices from any AVE-related company (like AVE BIHOR, AVE SRL)",
        "evaluation_criteria": [
            "Handles partial company name",
            "Does not include unrelated companies",
            "Retrieved only relevant invoices",
        ],
    },
    # 2. Wide amount range
    {
        "id": "eval_edge_1002",
        "query": "facturi între 500 și 5000 lei",
        "expected_metadata": {
            "document_type": "factura",
            "amount_min": 500,
            "amount_max": 5000,
        },
        "expected_filters": ["meta.document_type", "meta.amount"],
        "reference_answer": "Invoices between 500 and 5000 lei",
        "evaluation_criteria": [
            "Handles wide amount ranges correctly",
            "Includes invoices exactly at boundaries",
        ],
    },
    # 3. Relative date expression
    {
        "id": "eval_edge_1003",
        "query": "facturi AVE BIHOR luna trecută",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            # Year and month resolved dynamically
        },
        "expected_filters": [
            "meta.company",
            "meta.document_type",
            "meta.year",
            "meta.month",
        ],
        "reference_answer": "Invoices from AVE BIHOR for last month",
        "evaluation_criteria": [
            "Correctly parses 'luna trecută'",
            "Filters invoices for correct month",
        ],
    },
    # 4. Mixed document types
    {
        "id": "eval_edge_1004",
        "query": "documente Cubus Arts din 2024",
        "expected_metadata": {
            "company": "Cubus Arts",
        },
        "expected_filters": ["meta.company", "meta.year"],
        "reference_answer": "All documents related to Cubus Arts from 2024",
        "evaluation_criteria": [
            "Does not force document_type",
            "Retrieved mixed document types",
        ],
    },
    # 5. Numeric words
    {
        "id": "eval_edge_1005",
        "query": "facturi Cubus Arts peste două mii lei",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_min": 2000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Cubus Arts invoices over 2000 lei",
        "evaluation_criteria": [
            "Correctly parses numeric words",
            "Filters by amount_min = 2000",
        ],
    },
    # 6. Nonexistent filter
    {
        "id": "eval_edge_006",
        "query": "facturi AVE BIHOR cu cod fiscal 123456789",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
        },
        "expected_filters": ["meta.company", "meta.document_type"],
        "reference_answer": "Invoices from AVE BIHOR (cod fiscal filter ignored)",
        "evaluation_criteria": [
            "Ignores unsupported filters gracefully",
            "Retrieves correct invoices",
        ],
    },
    # 7. Multiple companies
    {
        "id": "eval_edge_007",
        "query": "facturi AVE BIHOR și Cubus Arts",
        "expected_metadata": {
            "company": ["AVE BIHOR", "Cubus Arts"],
            "document_type": "factura",
        },
        "expected_filters": ["meta.company", "meta.document_type"],
        "reference_answer": "Invoices from AVE BIHOR or Cubus Arts",
        "evaluation_criteria": [
            "Handles multiple companies correctly",
            "Retrieved invoices for both companies",
        ],
    },
    # 8. Typo in company name
    {
        "id": "eval_edge_008",
        "query": "facturi AVE BIHRO peste 2000 lei",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "amount_min": 2000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "High-value invoices from AVE BIHOR over 2000 lei",
        "evaluation_criteria": [
            "Handles minor typos",
            "Filters by amount_min correctly",
        ],
    },
    # 9. Date range ambiguity
    {
        "id": "eval_edge_009",
        "query": "facturi ianuarie-martie 2025",
        "expected_metadata": {
            "document_type": "factura",
            "year": 2025,
            "month_min": 1,
            "month_max": 3,
        },
        "expected_filters": ["meta.document_type", "meta.year", "meta.month"],
        "reference_answer": "Invoices from January to March 2025",
        "evaluation_criteria": [
            "Correctly parses date ranges",
            "Filters by month_min and month_max",
        ],
    },
    # 10. Superlative expressions
    {
        "id": "eval_edge_010",
        "query": "cele mai mari facturi AVE BIHOR",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "amount_min": 1000,  # inferred as "largest"
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Largest invoices from AVE BIHOR",
        "evaluation_criteria": [
            "Correctly interprets 'cele mai mari'",
            "Filters by amount_min appropriately",
        ],
    },
    # 11. Negative numbers
    {
        "id": "eval_edge_011",
        "query": "facturi Cubus Arts sub 0 lei",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_max": 0,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Invoices with negative or zero amounts",
        "evaluation_criteria": [
            "Correctly handles amount_max = 0",
            "Does not crash on negative values",
        ],
    },
    # 12. Missing company
    {
        "id": "eval_edge_012",
        "query": "toate facturile din 2025",
        "expected_metadata": {
            "document_type": "factura",
            "year": 2025,
        },
        "expected_filters": ["meta.document_type", "meta.year"],
        "reference_answer": "All invoices from 2025 regardless of company",
        "evaluation_criteria": [
            "Correctly handles missing company",
            "Retrieves invoices only by year and type",
        ],
    },
    # 13. Multi-document types with amount
    {
        "id": "eval_edge_013",
        "query": "documente peste 1000 lei AVE BIHOR",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "amount_min": 1000,
        },
        "expected_filters": ["meta.company", "meta.amount"],
        "reference_answer": "All documents (contracts, invoices) over 1000 lei from AVE BIHOR",
        "evaluation_criteria": [
            "Applies amount filter without forcing document type",
        ],
    },
    # 14. Ambiguous month names
    {
        "id": "eval_edge_014",
        "query": "facturi Cubus Arts în martie",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "month": 3,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.month"],
        "reference_answer": "Invoices from Cubus Arts in March (year unspecified)",
        "evaluation_criteria": [
            "Correctly identifies month 'martie'",
            "Handles missing year gracefully",
        ],
    },
    # 15. Compound queries with OR and AND
    {
        "id": "eval_edge_015",
        "query": "facturi AVE BIHOR sau Cubus Arts în 2025",
        "expected_metadata": {
            "company": ["AVE BIHOR", "Cubus Arts"],
            "document_type": "factura",
            "year": 2025,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.year"],
        "reference_answer": "Invoices from AVE BIHOR or Cubus Arts in 2025",
        "evaluation_criteria": [
            "Correctly handles 'sau' as OR",
            "Filters by year",
        ],
    },
    # 16. Nonexistent year
    {
        "id": "eval_edge_016",
        "query": "facturi AVE BIHOR din 2030",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "year": 2030,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.year"],
        "reference_answer": "No invoices expected for 2030",
        "evaluation_criteria": [
            "Handles non-existent years gracefully",
            "Returns empty results without errors",
        ],
    },
    # 17. Misordered query elements
    {
        "id": "eval_edge_017",
        "query": "peste 2000 lei facturi Cubus Arts",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_min": 2000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Invoices from Cubus Arts over 2000 lei",
        "evaluation_criteria": [
            "Correctly extracts amount_min even if query order is unusual",
        ],
    },
    # 18. Ambiguous language: 'toate facturile mari'
    {
        "id": "eval_edge_018",
        "query": "toate facturile mari",
        "expected_metadata": {
            "document_type": "factura",
            "amount_min": 1000,  # 'mari' interpreted as high-value
        },
        "expected_filters": ["meta.document_type", "meta.amount"],
        "reference_answer": "All high-value invoices",
        "evaluation_criteria": [
            "Infers 'mari' as amount_min",
            "Handles missing company filter",
        ],
    },
    # 19. Nested entities: company and branch
    {
        "id": "eval_edge_019",
        "query": "facturi AVE BIHOR Sediul Central",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "branch": "Sediul Central",  # extra metadata
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.branch"],
        "reference_answer": "Invoices from AVE BIHOR Central Office",
        "evaluation_criteria": [
            "Correctly parses branch-level entity",
            "Filters invoices for that branch only",
        ],
    },
    # 20. Complex amounts: ranges with different units
    {
        "id": "eval_edge_020",
        "query": "facturi Cubus Arts între 1k și 3k lei",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_min": 1000,
            "amount_max": 3000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Invoices from Cubus Arts between 1000 and 3000 lei",
        "evaluation_criteria": [
            "Parses shorthand amounts (1k = 1000)",
            "Applies min and max filters correctly",
        ],
    },
]

def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """Return the full evaluation dataset"""
    return EVALUATION_DATASET


def get_evaluation_by_id(eval_id: str) -> Dict[str, Any]:
    """Get a specific evaluation case by ID"""
    for item in EVALUATION_DATASET:
        if item["id"] == eval_id:
            return item
    raise ValueError(f"Evaluation ID {eval_id} not found")