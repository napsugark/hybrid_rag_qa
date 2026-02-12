"""
Custom Haystack component for metadata enrichment using local LLM
"""

from typing import List, Dict, Any, Optional
import json
import logging

from haystack import component, Document
from haystack.components.builders import PromptBuilder
from .. import config
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage



logger = logging.getLogger(__name__)


@component
class MetadataEnricher:
    """
    Enriches documents with extracted metadata using a local LLM.
    
    Extracts:
    - Named entities (companies, people, locations, dates)
    - Topics and themes
    - Keywords
    - Document type classification
    - Language detection
    """

    def __init__(
        self,
        llm_type: str = None,
        llm_model: str = None,
        llm_url: str = None,
        llm_api_key: str = None,
        llm_api_version: str = None,
        metadata_fields: Optional[List[str]] = None,
        timeout: int = 600,
        append_metadata_to_content: bool = True,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize metadata enricher
        
        Args:
            ollama_url: URL of Ollama server
            ollama_model: Ollama model to use
            metadata_fields: List of metadata fields to extract
            timeout: Request timeout in seconds
            append_metadata_to_content: If True, appends extracted metadata to content for better retrieval
            prompt_template: Custom prompt template (uses default if not provided)
        """
        self.llm_type = llm_type or getattr(config, "LLM_TYPE", "OLLAMA")
        self.llm_model = llm_model or getattr(config, "LLM_MODEL", "llama3.1:8b")
        self.llm_url = llm_url or getattr(config, "LLM_URL", "http://127.0.0.1:11435")
        self.llm_api_key = llm_api_key or getattr(config, "LLM_API_KEY", None)
        self.llm_api_version = llm_api_version or getattr(config, "LLM_API_VERSION", None)
        self.timeout = timeout
        self.append_metadata_to_content = append_metadata_to_content
        
        if metadata_fields is None:
            self.metadata_fields = [
                "entities",
                "topics",
                "keywords",
                "document_type",
                "language",
                "triples"
            ]
        else:
            self.metadata_fields = metadata_fields
        
        # Use provided prompt or default
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            # Fallback default prompt (for backwards compatibility)
            self.prompt_template = """
Analyze the following text and extract structured metadata.

Text:
{{content}}
{{detected_type_hint}}

Respond with ONLY a valid JSON object in this exact format (no markdown, no comments, no extra text):
{
  "entities": ["entity1", "entity2"],
  "topics": ["topic1", "topic2"],
  "keywords": ["keyword1", "keyword2"],
  "document_type": "report",
  "language": "en",
  "triples": [
    {
      "subject": "Entity or concept",
      "relation": "action or relationship",
      "object": "Entity, value, or concept"
    }
  ]
}

Rules:
- entities: Max 5 important names (people, places, organizations, dates) - PRESERVE ORIGINAL LANGUAGE from text
- topics: Max 3 main themes - PRESERVE ORIGINAL LANGUAGE from text
- keywords: Max 5 important terms - PRESERVE ORIGINAL LANGUAGE from text
- document_type: One of: factura, chitanta, contract, oferta, raport, email, buget, technical_doc, other
- language: Two-letter code (en, ro, de, etc.)
- triples: ONLY extract 2-3 of the MOST IMPORTANT factual relationships
  * Extract ONLY if there are clear, meaningful business relationships (buyer-seller, payment, contract, delivery, ownership)
  * Subject and Object: PRESERVE ORIGINAL LANGUAGE (e.g., company names, amounts in original language)
  * Relation: Use English standardized verbs ("sold_to", "purchased_from", "paid_amount", "delivered_to", "contracted_with", "located_at")
  * DO NOT extract generic relations like "has", "uses", "operates"
  * DO NOT use entity attributes (addresses, IDs, dates) as subjects or objects unless they're the main focus
  * If no strong relationships exist, return empty array []
- DO NOT include comments, explanations, or markdown formatting
- Return ONLY the JSON object
"""
        
        # Initialize the correct generator based on config.LLM_TYPE
        if self.llm_type == "OLLAMA":
            if OllamaGenerator is None:
                raise ImportError("OllamaGenerator not available")
            self.generator = OllamaGenerator(
                model=self.llm_model,
                url=self.llm_url,
                timeout=self.timeout,
                generation_kwargs={
                    "temperature": 0.3,
                    "num_predict": 1200,
                }
            )
            logger.info(f"MetadataEnricher initialized with Ollama model {self.llm_model}")
        elif self.llm_type == "AZURE_OPENAI":
            if AzureOpenAIChatGenerator is None:
                raise ImportError("AzureOpenAIChatGenerator not available")
            self.generator = AzureOpenAIChatGenerator(
                azure_deployment=self.llm_model,
                azure_endpoint=self.llm_url,
                api_key=Secret.from_token(self.llm_api_key),
                api_version=self.llm_api_version,
                timeout=self.timeout,
                generation_kwargs={
                    "temperature": 0.3,
                    "max_tokens": 1200,
                },
            )
            logger.info(f"MetadataEnricher initialized with Azure OpenAI deployment {self.llm_model}")
        else:
            raise ValueError(f"Unsupported LLM_TYPE: {self.llm_type}")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Enrich documents with metadata
        
        Args:
            documents: List of documents to enrich
            
        Returns:
            Dictionary with enriched documents
        """
        if not documents:
            return {"documents": []}
        
        enriched_docs = []
        logger.info(f"Enriching {len(documents)} documents with metadata")
        
        for idx, doc in enumerate(documents):
            try:
                # Skip if content is too short
                if not doc.content or len(doc.content.strip()) < 50:
                    logger.debug(f"Document {idx} too short for metadata extraction")
                    enriched_docs.append(doc)
                    continue
                
                # Prepare content for extraction based on strategy
                strategy = getattr(config, 'METADATA_EXTRACTION_STRATEGY', 'full_document')
                
                if strategy == 'full_document':
                    content_for_extraction = doc.content
                elif strategy == 'first_page':
                    # Approximate first page as first 2500 chars (conservative estimate)
                    content_for_extraction = doc.content[:2500]
                elif strategy == 'first_n_chars':
                    max_chars = getattr(config, 'METADATA_EXTRACTION_MAX_CHARS', 3000)
                    content_for_extraction = doc.content[:max_chars]
                else:
                    logger.warning(f"Unknown METADATA_EXTRACTION_STRATEGY: {strategy}, using full document")
                    content_for_extraction = doc.content
                
                logger.debug(f"Extracting metadata using strategy='{strategy}' "
                           f"({len(content_for_extraction)} / {len(doc.content)} chars)")
                
                prompt = self.prompt_template.replace("{{content}}", content_for_extraction)
                
                # Inject detected_document_type hint if available
                detected_type = doc.meta.get("detected_document_type", "")
                if detected_type and detected_type != "unknown":
                    hint = (f"\nHINT: A preliminary analysis detected this document as "
                            f"type '{detected_type}'. Consider this when classifying.")
                else:
                    hint = ""
                prompt = prompt.replace("{{detected_type_hint}}", hint)

                # Send prompt to LLM and handle both Ollama and AzureOpenAI correctly
                chat_message = ChatMessage.from_user(prompt)

                # Conditional kwargs depending on LLM type
                kwargs = {}
                if self.llm_type == "OLLAMA":
                    kwargs = {"num_predict": 1200, "temperature": 0.3}
                else:  # AzureOpenAI
                    kwargs = {"max_tokens": 1200, "temperature": 0.3}

                # Run the generator
                result = self.generator.run(messages=[chat_message], generation_kwargs=kwargs)

                # Extract reply text safely
                if result and "replies" in result and result["replies"]:
                    reply = result["replies"][0]
                    if isinstance(reply, ChatMessage):
                        response_text = reply.text.strip()  # Use .text instead of .content
                    else:
                        response_text = str(reply).strip()
                else:
                    logger.error(f"No response from LLM for document {idx}. Result: {result}")
                    enriched_docs.append(doc)
                    continue

                # Parse JSON from the response
                metadata = self._parse_json_response(response_text)
                
                if metadata:
                    # Log agreement/disagreement for later evaluation
                    if "detected_document_type" in doc.meta and "document_type" in metadata:
                        detected = doc.meta["detected_document_type"]
                        llm_type = metadata["document_type"]
                        if detected and detected != "unknown":
                            if detected == llm_type:
                                logger.info(
                                    f"Doc {idx}: type AGREEMENT — "
                                    f"detected='{detected}', llm='{llm_type}'"
                                )
                            else:
                                logger.warning(
                                    f"Doc {idx}: type DISAGREEMENT — "
                                    f"detected='{detected}', llm='{llm_type}'"
                                )
                    
                    # Store LLM's classification as llm_document_type for measurement
                    if "document_type" in metadata:
                        metadata["llm_document_type"] = metadata["document_type"]
                    
                    # Add extracted metadata to document meta
                    doc.meta.update(metadata)
                    logger.info(f"Successfully extracted metadata for doc {idx}: {list(metadata.keys())}")
                    
                    # Append metadata to content for better retrieval
                    if self.append_metadata_to_content:
                        metadata_text = self._format_metadata_for_content(metadata)
                        if metadata_text:
                            doc.content = f"{doc.content}\n\n{metadata_text}"
                            logger.debug(f"Appended metadata to content for doc {idx}")
                else:
                    logger.warning(f"Failed to parse metadata for document {idx}.")
                    logger.warning(f"Full response: {response_text}")
                
                enriched_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error enriching document {idx}: {type(e).__name__}: {str(e)}")
                logger.exception(f"Full traceback for document {idx}:")
                enriched_docs.append(doc)  # Add document without enrichment
        
        logger.info(f"Enriched {len(enriched_docs)} documents")
        return {"documents": enriched_docs}
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling various formats
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed metadata dictionary or None
        """
        # Clean response - remove markdown code fences and extra text
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0].strip()
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0].strip()
        
        # Remove common prefixes
        for prefix in ['JSON:', 'Here is the JSON:', 'Response:']:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Find JSON object boundaries
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        
        # Auto-fix incomplete JSON (missing closing brace)
        if start == -1:
            logger.debug(f"No JSON object found in response")
            return None
            
        if end == -1 or end <= start:
            logger.debug(f"Incomplete JSON detected (missing closing brace), auto-fixing...")
            # Add missing closing brace
            cleaned = cleaned.rstrip() + '\n}'
            end = cleaned.rfind('}')
        
        json_str = cleaned[start:end+1]
        
        # Remove JavaScript-style comments (// ...)
        import re
        json_str = re.sub(r'//[^\n]*', '', json_str)
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON: {e}")
            logger.debug(f"Attempted to parse: {json_str[:200]}...")
            return None

    def _format_metadata_for_content(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata as searchable text to append to document content
        
        Args:
            metadata: Extracted metadata dictionary
            
        Returns:
            Formatted metadata text
        """
        parts = []
        
        # Add structured fields first (most important for filtering)
        structured = []
        if company := metadata.get("company"):
            structured.append(f"Company: {company}")
        if client := metadata.get("client"):
            structured.append(f"Client: {client}")
        if date := metadata.get("date"):
            structured.append(f"Date: {date}")
        if year := metadata.get("year"):
            structured.append(f"Year: {year}")
        if month := metadata.get("month"):
            structured.append(f"Month: {month}")
        if day := metadata.get("day"):
            structured.append(f"Day: {day}")
        if invoice_number := metadata.get("invoice_number"):
            structured.append(f"Invoice: {invoice_number}")
        if amount := metadata.get("amount"):
            currency = metadata.get("currency", "")
            structured.append(f"Amount: {amount} {currency}".strip())
        
        if structured:
            parts.append(f"[Metadata - {', '.join(structured)}]")
        
        # Add entities (dates, names, locations) - most important for search
        if entities := metadata.get("entities", []):
            parts.append(f"[Entities: {', '.join(str(e) for e in entities)}]")
        
        # Add keywords for better keyword matching
        if keywords := metadata.get("keywords", []):
            parts.append(f"[Keywords: {', '.join(str(k) for k in keywords)}]")
        
        # Add topics
        if topics := metadata.get("topics", []):
            parts.append(f"[Topics: {', '.join(str(t) for t in topics)}]")
        
        # Add document type and language
        extras = []
        if doc_type := metadata.get("document_type"):
            extras.append(f"Type: {doc_type}")
        if language := metadata.get("language"):
            extras.append(f"Language: {language}")
        
        if extras:
            parts.append(f"[{', '.join(extras)}]")
        
        # Add triples for relationship-based search
        if triples := metadata.get("triples"):
            triples_text = self._format_triples_for_content(triples)
            if triples_text:
                parts.append(triples_text)
        
        return "\n".join(parts)
    
    def _format_triples_for_content(self, triples: List[Dict[str, str]]) -> str:
        """
        Format triples as searchable text
        
        Args:
            triples: List of triple dictionaries with subject, relation, object
            
        Returns:
            Formatted triples text
        """
        lines = []
        for triple in triples:
            subject = triple.get("subject")
            relation = triple.get("relation")
            obj = triple.get("object")
            
            if subject and relation and obj:
                lines.append(f"[Triple] {subject} → {relation} → {obj}")
        
        return "\n".join(lines)
