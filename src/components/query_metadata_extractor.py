"""
Extract filterable metadata from Romanian queries for smart document retrieval
Example: "facturi Electrica din anul trecut" → filters: {company: "Electrica", year: 2025}
"""

import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

from .. import config

# Company name mappings: short name → list of full names in metadata
COMPANY_VARIATIONS = {
    "electrica": ["SOCIETATEA ELECTRICA FURNIZARE S.A.", "ELECTRICA FURNIZARE", "Electrică Furnizare", "ELECTRICA", "Electrica"],
    "digi": ["Digi Romania S.A.", "DIGI ROMANIA", "DIGI", "Digi"],
    "ave": ["AVE ROMANIA SRL", "AVE ROMANIA", "AVE"],
    "delivery": ["DELIVERY SOLUTIONS S.A.", "DELIVERY SOLUTIONS", "Delivery Solutions"],
    "cubus": ["S.C. Cubus Arts S.R.L.", "Cubus Arts", "CUBUS"],
}


@component
class RomanianQueryMetadataExtractor:
    """
    Extracts structured metadata filters from Romanian language queries

    This component analyzes natural language queries in Romanian and extracts
    filterable metadata that can be used to narrow down document search.

    Example transformations:
    - "facturi Electrica din anul trecut" → {company: "Electrica", year: 2025, document_type: "factura"}
    - "chitanțe OMV ianuarie 2024" → {company: "OMV", year: 2024, month: 1, document_type: "chitanta"}
    - "contracte Engie" → {company: "Engie", document_type: "contract"}

    The extracted metadata is converted into Qdrant-compatible filter structures
    that can be directly applied to retrieval operations.
    """

    def __init__(
        self,
        llm_type: str = None,
        llm_model: str = None,
        llm_url: str = None,
        llm_api_key: str = None,
        llm_api_version: str = None,
        timeout: int = 120,
    ):
        """
        Initialize the query metadata extractor

        Args:
            ollama_url: Ollama server URL (defaults to config.OLLAMA_URL)
            ollama_model: Ollama model name (defaults to config.OLLAMA_MODEL)
        """
        self.current_year = datetime.now().year
        self.llm_type = llm_type or getattr(config, "LLM_TYPE", "OLLAMA")
        self.llm_model = llm_model or getattr(config, "LLM_MODEL", "llama3.1:8b")
        self.llm_url = llm_url or getattr(config, "LLM_URL", "http://127.0.0.1:11435")
        self.llm_api_key = llm_api_key or getattr(config, "LLM_API_KEY", None)
        self.llm_api_version = llm_api_version or getattr(config, "LLM_API_VERSION", None)
        self.timeout = timeout

        # Load prompt template from file or use default
        prompt_file = getattr(config, "QUERY_EXTRACTION_PROMPT_FILE", None)
        if prompt_file and prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                query_extraction_prompt = f.read()
        else:
            # Fall back to default prompt
            query_extraction_prompt = """
You are a query analysis system for a Romanian document management system.

User query: "{{query}}"
Current year: {{current_year}}

Extract structured information for filtering:

RULES:
- "anul trecut" (last year) = {{current_year - 1}}
- "anul acesta" / "acest an" (this year) = {{current_year}}
- "facturi" / "factura" → document_type: "factura"
- "chitanțe" / "chitanta" → document_type: "chitanta"
- "contract" / "contracte" → document_type: "contract"
- "oferta" / "oferte" → document_type: "oferta"
- Company names: Electrica, Engie, OMV, Enel, Digi, Orange, Vodafone, etc.
- Months in Romanian: ianuarie=1, februarie=2, martie=3, aprilie=4, mai=5, iunie=6, iulie=7, august=8, septembrie=9, octombrie=10, noiembrie=11, decembrie=12
- CRITICAL: If you see an EXPLICIT date like "18.01.2025" or "DD.MM.YYYY", extract the EXACT YEAR from that date
- If "din" (from) is followed by a date, extract year and month from that EXPLICIT date
- COMPARATIVE AMOUNTS (Romanian):
  * "mari" / "mare" (large/big) → amount_min: 1000 (bills over 1000 RON)
  * "mici" / "mic" (small) → amount_max: 100 (bills under 100 RON)
  * "peste X" / "mai mari de X" (over X) → amount_min: X
  * "sub X" / "mai mici de X" (under X) → amount_max: X
  * "între X și Y" / "between X and Y" → amount_min: X, amount_max: Y
  * Common terms: "energie" (energy), "curent" (electricity), "gaz" (gas), "apă" (water), "telefon" (phone), "internet"

JSON format:
{
  "company": "company name or null if not mentioned",
  "year": year as integer or null,
  "month": month as integer 1-12 or null,
  "day": day as integer 1-31 or null,
  "document_type": "factura/chitanta/contract/oferta/etc or null",
  "invoice_number": "invoice number or null",
  "amount_min": minimum amount as number or null,
  "amount_max": maximum amount as number or null,
  "search_terms": ["important terms for semantic search"]
}

Examples:

Query: "Facturi mari pentru energie în 2025"
{
  "company": null,
  "year": 2025,
  "month": null,
  "day": null,
  "document_type": "factura",
  "invoice_number": null,
  "amount_min": 1000,
  "amount_max": null,
  "search_terms": ["facturi", "energie", "2025", "mari"]
}

Query: "facturi din 18.01.2025"
{
  "company": null,
  "year": 2025,
  "month": 1,
  "day": 18,
  "document_type": "factura",
  "invoice_number": null,
  "amount_min": null,
  "amount_max": null,
  "search_terms": ["facturi", "18.01.2025"]
}

Query: "facturi Electrica din anul trecut"
{
  "company": "Electrica",
  "year": {{current_year - 1}},
  "month": null,
  "day": null,
  "document_type": "factura",
  "invoice_number": null,
  "amount_min": null,
  "amount_max": null,
  "search_terms": ["facturi", "Electrica", "{{current_year - 1}}"]
}

Query: "chitanțe OMV ianuarie 2024"
{
  "company": "OMV",
  "year": 2024,
  "month": 1,
  "day": null,
  "document_type": "chitanta",
  "invoice_number": null,
  "amount_min": null,
  "amount_max": null,
  "search_terms": ["chitanțe", "OMV", "ianuarie", "2024"]
}

Query: "contracte Engie mai mari de 500 lei"
{
  "company": "Engie",
  "year": null,
  "month": null,
  "day": null,
  "document_type": "contract",
  "invoice_number": null,
  "amount_min": 500,
  "amount_max": null,
  "search_terms": ["contracte", "Engie", "peste 500 lei"]
}

Query: "facturi mici pentru telefon"
{
  "company": null,
  "year": null,
  "month": null,
  "day": null,
  "document_type": "factura",
  "invoice_number": null,
  "amount_min": null,
  "amount_max": 100,
  "search_terms": ["facturi", "telefon", "mici"]
}

Now analyze this query and respond ONLY with valid JSON:
"""

        # Build extraction pipeline
        self.pipeline = Pipeline()
        
        if self.llm_type == "OLLAMA":
            if OllamaGenerator is None:
                raise ImportError("OllamaGenerator not available")
            self.pipeline.add_component(
                "builder", PromptBuilder(template=query_extraction_prompt)
            )
            self.pipeline.add_component(
                "llm",
                OllamaGenerator(
                    model=self.llm_model,
                    url=self.llm_url,
                    timeout=120,
                    generation_kwargs={
                        "temperature": 0.1,
                        "num_predict": 300,
                    },
                ),
            )
            self.pipeline.connect("builder.prompt", "llm.prompt")
        elif self.llm_type == "AZURE_OPENAI":
            if AzureOpenAIChatGenerator is None:
                raise ImportError("AzureOpenAI not available")
            # For Azure OpenAI, wrap the template in ChatMessage format
            from haystack.components.builders import ChatPromptBuilder
            chat_template = [
                ChatMessage.from_user(query_extraction_prompt)
            ]
            self.pipeline.add_component(
                "builder", ChatPromptBuilder(template=chat_template)
            )
            self.pipeline.add_component(
                "llm",
                AzureOpenAIChatGenerator(
                    azure_deployment=self.llm_model,
                    azure_endpoint=self.llm_url,
                    api_key=Secret.from_token(self.llm_api_key),
                    api_version=self.llm_api_version,
                    timeout=self.timeout,
                    generation_kwargs={
                        "temperature": 0.1,
                        "max_tokens": 300,
                    },
                ),
            )
            self.pipeline.connect("builder.prompt", "llm.messages")
        else:
            raise ValueError(f"Unsupported LLM_TYPE: {self.llm_type}")

    @component.output_types(
        filters=Dict[str, Any], search_query=str, metadata=Dict[str, Any]
    )
    def run(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata from Romanian query and build Qdrant filters

        Args:
            query: Romanian language query (e.g., "facturi Electrica din anul trecut")

        Returns:
            Dictionary containing:
            - filters: Qdrant filter dictionary ready to use in retrieval
            - search_query: Cleaned search query for semantic search
            - metadata: Extracted metadata for debugging/logging
        """
        try:
            # Run extraction pipeline
            result = self.pipeline.run(
                {"builder": {"query": query, "current_year": self.current_year}}
            )

            # Parse LLM response
            if self.llm_type == "OLLAMA":
                llm_output = result["llm"]["replies"][0]
            else:  # AZURE_OPENAI
                reply = result["llm"]["replies"][0]
                # Extract text from ChatMessage._content[0].text structure
                if hasattr(reply, '_content') and len(reply._content) > 0:
                    llm_output = reply._content[0].text
                elif hasattr(reply, 'content'):
                    llm_output = reply.content
                else:
                    llm_output = str(reply)
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", llm_output, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = llm_output

            metadata = json.loads(json_str)

            # Build Qdrant filters from extracted metadata
            filters = self._build_qdrant_filters(metadata)
            search_terms = metadata.get("search_terms", [])
            if search_terms:
                search_query = " ".join(search_terms)
            else:
                search_query = query

            return {
                "filters": filters,
                "search_query": search_query,
                "metadata": metadata,
            }

        except json.JSONDecodeError as e:
            # JSON parsing failed - log and use fallback
            print(f"[WARN] JSON parsing failed: {e}")
            print(f"LLM output: {llm_output[:200]}")
            return self._fallback_extraction(query)

        except Exception as e:
            # General error - log and use fallback
            print(f"[WARN] Query metadata extraction failed: {e}")
            return self._fallback_extraction(query)

    def _build_qdrant_filters(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert extracted metadata into Qdrant filter structure
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Qdrant filter dictionary or None if no filters
        """
        conditions = []
        
        # Company filter (Haystack stores metadata in 'meta' object)
        # Expand company abbreviations to full names stored in metadata
        if metadata.get("company"):
            company_query = metadata["company"].lower()
            # Check if it's a known abbreviation
            if company_query in COMPANY_VARIATIONS:
                # Use nested OR with "==" instead of "in" operator because
                # Haystack's "in" maps to MatchText (requires text index),
                # but we have a keyword index. "==" uses MatchValue which
                # works correctly with keyword indexes.
                variations = COMPANY_VARIATIONS[company_query]
                if len(variations) == 1:
                    conditions.append({
                        "field": "meta.company",
                        "operator": "==",
                        "value": variations[0]
                    })
                else:
                    conditions.append({
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.company", "operator": "==", "value": v}
                            for v in variations
                        ]
                    })
            else:
                # Try exact match (case-sensitive)
                conditions.append({
                    "field": "meta.company",
                    "operator": "==",
                    "value": metadata["company"]
                })
        
        # Year filter
        if metadata.get("year"):
            conditions.append({
                "field": "meta.year",
                "operator": "==",
                "value": int(metadata["year"])
            })
        
        # Month filter
        if metadata.get("month"):
            conditions.append({
                "field": "meta.month",
                "operator": "==",
                "value": int(metadata["month"])
            })
        
        # Day filter
        if metadata.get("day"):
            conditions.append({
                "field": "meta.day",
                "operator": "==",
                "value": int(metadata["day"])
            })
        
        # Document type filter
        if metadata.get("document_type"):
            conditions.append({
                "field": "meta.document_type",
                "operator": "==",
                "value": metadata["document_type"]
            })
        
        # Invoice number filter
        if metadata.get("invoice_number"):
            conditions.append({
                "field": "meta.invoice_number",
                "operator": "==",
                "value": metadata["invoice_number"]
            })
        
        # Amount filters (comparative)
        if metadata.get("amount_min") is not None:
            conditions.append({
                "field": "meta.amount",
                "operator": ">=",
                "value": float(metadata["amount_min"])
            })
        
        if metadata.get("amount_max") is not None:
            conditions.append({
                "field": "meta.amount",
                "operator": "<=",
                "value": float(metadata["amount_max"])
            })
        
        # Section type filter: When querying amounts, prioritize totals/line_items sections
        # (header sections inherit metadata but don't contain the actual amount details)
        # NOTE: Using nested OR with "==" instead of "in" operator because Haystack's "in"
        # is converted to MatchText (full-text search) which requires a text index,
        # but we have a keyword index. Using "==" uses MatchValue which works with keyword.
        if metadata.get("amount_min") is not None or metadata.get("amount_max") is not None:
            conditions.append({
                "operator": "OR",
                "conditions": [
                    {"field": "meta.section_type", "operator": "==", "value": "totals"},
                    {"field": "meta.section_type", "operator": "==", "value": "line_items"},
                    {"field": "meta.section_type", "operator": "==", "value": "header"},
                    {"field": "meta.section_type", "operator": "==", "value": "simple_chunk"},
                ]
            })
        
        # Build final filter structure
        if conditions:
            return {"operator": "AND", "conditions": conditions}
        return None

    def _fallback_extraction(self, query: str) -> Dict[str, Any]:
        """
        Fallback extraction using simple pattern matching
        Used when LLM extraction fails

        Args:
            query: Original query string

        Returns:
            Dictionary with minimal filters and original query
        """
        conditions = []
        metadata = {}

        # Simple pattern matching for common cases
        query_lower = query.lower()

        # Detect document type
        if "factur" in query_lower:
            metadata["document_type"] = "factura"
            conditions.append(
                {"field": "meta.document_type", "operator": "==", "value": "factura"}
            )
        elif "chitanț" in query_lower:
            metadata["document_type"] = "chitanta"
            conditions.append(
                {"field": "meta.document_type", "operator": "==", "value": "chitanta"}
            )
        elif "contract" in query_lower:
            metadata["document_type"] = "contract"
            conditions.append(
                {"field": "meta.document_type", "operator": "==", "value": "contract"}
            )

        # Detect date patterns (DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY)
        date_match = re.search(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b", query)
        if date_match:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            year = int(date_match.group(3))

            metadata["day"] = day
            metadata["month"] = month
            metadata["year"] = year

            conditions.append({"field": "meta.day", "operator": "==", "value": day})
            conditions.append({"field": "meta.month", "operator": "==", "value": month})
            conditions.append({"field": "meta.year", "operator": "==", "value": year})
        else:
            # Detect year patterns only if no full date found
            year_match = re.search(r"\b(202[0-9])\b", query)
            if year_match:
                year = int(year_match.group(1))
                metadata["year"] = year
                conditions.append({"field": "meta.year", "operator": "==", "value": year})

        # Detect "anul trecut" (last year)
        if "anul trecut" in query_lower or "anul precedent" in query_lower:
            year = self.current_year - 1
            metadata["year"] = year
            # Replace year filter if already exists
            conditions = [c for c in conditions if c.get("field") != "meta.year"]
            conditions.append({"field": "meta.year", "operator": "==", "value": year})

        # Detect common companies (case-insensitive)
        companies = [
            "electrica",
            "engie",
            "omv",
            "enel",
            "digi",
            "orange",
            "vodafone",
            "rcs rds",
        ]
        for company in companies:
            if company in query_lower:
                company_title = company.title()
                metadata["company"] = company_title
                conditions.append(
                    {"field": "meta.company", "operator": "==", "value": company_title}
                )
                break

        # Build Haystack filter structure
        if conditions:
            filters = {"operator": "AND", "conditions": conditions}
        else:
            filters = None

        return {"filters": filters, "search_query": query, "metadata": metadata}


# Example usage and testing
if __name__ == "__main__":
    """
    Test the query metadata extractor with sample Romanian queries
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    # Initialize extractor
    extractor = RomanianQueryMetadataExtractor()

    # Test queries
    test_queries = [
        "facturi Electrica din anul trecut",
        "chitanțe OMV ianuarie 2024",
        "contracte Engie",
        "toate facturile de la Digi din 2025",
        "factura numarul 12345",
        "facturi mai mari de 500 lei",
    ]

    print("=" * 80)
    print("TESTING ROMANIAN QUERY METADATA EXTRACTOR")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        result = extractor.run(query=query)

        print(f"Extracted metadata: {result['metadata']}")
        print(f"Qdrant filters: {result['filters']}")
        print(f"Search query: {result['search_query']}")
