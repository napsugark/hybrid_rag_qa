"""
Evaluation dataset for RAG system
Contains test questions with expected metadata and reference answers
"""

from typing import List, Dict, Any

# Evaluation dataset: questions with expected outputs for testing
EVALUATION_DATASET: List[Dict[str, Any]] = [
    # {
    #     "id": "eval_001",
    #     "query": "facturi sub 1000 lei",
    #     "expected_metadata": {
    #         "document_type": "factura",
    #         "amount_max": 1000,
    #     },
    #     "expected_filters": ["meta.document_type", "meta.amount"],
    #     "reference_answer": "Invoices under 1000 lei should include Digi invoices around 50-60 lei",
    #     "evaluation_criteria": [
    #         "Correctly filtered by document_type=factura",
    #         "Correctly filtered by amount <= 1000",
    #         "Retrieved relevant invoice chunks",
    #         "Answer mentions specific amounts under 1000 lei",
    #     ],
    # },
    # {
    #     "id": "eval_002",
    #     "query": "facturi Electrica din 2025",
    #     "expected_metadata": {
    #         "company": "Electrica",
    #         "document_type": "factura",
    #         "year": 2025,
    #     },
    #     "expected_filters": ["meta.company", "meta.document_type", "meta.year"],
    #     "reference_answer": "Electrica invoices from 2025",
    #     "evaluation_criteria": [
    #         "Correctly identified company Electrica",
    #         "Correctly filtered by year 2025",
    #         "Retrieved Electrica invoices only",
    #     ],
    # },
    # {
    #     "id": "eval_003",
    #     "query": "facturi din 18.01.2025",
    #     "expected_metadata": {
    #         "document_type": "factura",
    #         "year": 2025,
    #         "month": 1,
    #         "day": 18,
    #     },
    #     "expected_filters": ["meta.document_type", "meta.year", "meta.month", "meta.day"],
    #     "reference_answer": "Invoices from January 18, 2025",
    #     "evaluation_criteria": [
    #         "Correctly extracted exact date (18.01.2025)",
    #         "Filtered by year, month, and day",
    #         "Retrieved documents from that specific date",
    #     ],
    # },
    # {
    #     "id": "eval_004",
    #     "query": "facturi mari pentru energie",
    #     "expected_metadata": {
    #         "document_type": "factura",
    #         "amount_min": 1000,
    #     },
    #     "expected_filters": ["meta.document_type", "meta.amount"],
    #     "reference_answer": "Large energy invoices (over 1000 lei)",
    #     "evaluation_criteria": [
    #         "Correctly identified 'mari' as amount_min: 1000",
    #         "Filtered by large amounts",
    #         "Retrieved high-value energy invoices",
    #     ],
    # },
    # {
    #     "id": "eval_005",
    #     "query": "contracte Engie",
    #     "expected_metadata": {
    #         "company": "Engie",
    #         "document_type": "contract",
    #     },
    #     "expected_filters": ["meta.company", "meta.document_type"],
    #     "reference_answer": "Contracts with Engie",
    #     "evaluation_criteria": [
    #         "Correctly identified document type as contract",
    #         "Filtered by company Engie",
    #     ],
    # },
    # {
    #     "id": "eval_006",
    #     "query": "toate facturile din ianuarie 2025",
    #     "expected_metadata": {
    #         "document_type": "factura",
    #         "year": 2025,
    #         "month": 1,
    #     },
    #     "expected_filters": ["meta.document_type", "meta.year", "meta.month"],
    #     "reference_answer": "All invoices from January 2025",
    #     "evaluation_criteria": [
    #         "Correctly parsed Romanian month 'ianuarie'",
    #         "Filtered by month=1 and year=2025",
    #     ],
    # },
    # {
    #     "id": "eval_007",
    #     "query": "chitanțe OMV",
    #     "expected_metadata": {
    #         "company": "OMV",
    #         "document_type": "chitanta",
    #     },
    #     "expected_filters": ["meta.company", "meta.document_type"],
    #     "reference_answer": "OMV receipts",
    #     "evaluation_criteria": [
    #         "Correctly identified document type as chitanta (receipt)",
    #         "Filtered by company OMV",
    #     ],
    # },
    # {
    #     "id": "eval_008",
    #     "query": "facturi între 100 și 500 lei",
    #     "expected_metadata": {
    #         "document_type": "factura",
    #         "amount_min": 100,
    #         "amount_max": 500,
    #     },
    #     "expected_filters": ["meta.document_type", "meta.amount"],
    #     "reference_answer": "Invoices between 100 and 500 lei",
    #     "evaluation_criteria": [
    #         "Correctly extracted range (100-500)",
    #         "Applied both min and max amount filters",
    #     ],
    # },
    {
        "id": "eval_009",
        "query": "facturi AVE BIHOR",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
        },
        "expected_filters": ["meta.company", "meta.document_type"],
        "reference_answer": "Invoices issued by AVE BIHOR",
        "evaluation_criteria": [
            "Correctly identified company AVE BIHOR",
            "Correctly identified document type factura",
            "Retrieved only AVE BIHOR invoices",
        ],
    },
    {
        "id": "eval_010",
        "query": "facturi AVE BIHOR peste 2000 lei",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "amount_min": 2000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "High-value invoices from AVE BIHOR over 2000 lei",
        "evaluation_criteria": [
            "Correctly extracted amount_min = 2000",
            "Filtered by company AVE BIHOR",
            "Returned only large invoices",
        ],
    },
    {
        "id": "eval_011",
        "query": "facturi Cubus Arts din 2024",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "year": 2024,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.year"],
        "reference_answer": "Cubus Arts invoices from 2024",
        "evaluation_criteria": [
            "Correctly identified company Cubus Arts",
            "Correctly extracted year 2024",
            "Filtered invoices by year and company",
        ],
    },
    {
        "id": "eval_012",
        "query": "contract Cubus Arts",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "contract",
        },
        "expected_filters": ["meta.company", "meta.document_type"],
        "reference_answer": "Contracts signed with Cubus Arts",
        "evaluation_criteria": [
            "Correctly identified document type contract",
            "Filtered by company Cubus Arts",
        ],
    },
    {
        "id": "eval_013",
        "query": "facturi AVE BIHOR din martie 2025",
        "expected_metadata": {
            "company": "AVE BIHOR",
            "document_type": "factura",
            "year": 2025,
            "month": 3,
        },
        "expected_filters": [
            "meta.company",
            "meta.document_type",
            "meta.year",
            "meta.month",
        ],
        "reference_answer": "AVE BIHOR invoices from March 2025",
        "evaluation_criteria": [
            "Correctly parsed Romanian month 'martie'",
            "Filtered by year and month",
            "Returned only AVE BIHOR invoices",
        ],
    },
    {
        "id": "eval_014",
        "query": "facturi Cubus Arts sub 500 lei",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_max": 500,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Cubus Arts invoices under 500 lei",
        "evaluation_criteria": [
            "Correctly extracted amount_max = 500",
            "Filtered by company Cubus Arts",
            "Retrieved low-value invoices only",
        ],
    },
    {
        "id": "eval_015",
        "query": "toate documentele AVE BIHOR",
        "expected_metadata": {
            "company": "AVE BIHOR",
        },
        "expected_filters": ["meta.company"],
        "reference_answer": "All documents related to AVE BIHOR",
        "evaluation_criteria": [
            "Correctly identified company AVE BIHOR",
            "Did not incorrectly force document_type",
            "Retrieved mixed document types (facturi, contracte, etc.)",
        ],
    },
    {
        "id": "eval_016",
        "query": "facturi Cubus Arts între 1000 și 3000 lei",
        "expected_metadata": {
            "company": "Cubus Arts",
            "document_type": "factura",
            "amount_min": 1000,
            "amount_max": 3000,
        },
        "expected_filters": ["meta.company", "meta.document_type", "meta.amount"],
        "reference_answer": "Cubus Arts invoices between 1000 and 3000 lei",
        "evaluation_criteria": [
            "Correctly extracted amount range",
            "Applied both min and max filters",
            "Returned invoices within the range",
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
