"""
Semantic Chunker Component
Splits documents by logical sections instead of fixed size
Specialized for Romanian invoices, contracts, and official documents
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from haystack import component, Document

logger = logging.getLogger(__name__)


@component
class SemanticDocumentChunker:
    """
    Splits documents by semantic sections (header, line items, legal, etc.)
    instead of arbitrary character/word boundaries
    
    For invoices:
    - Header (company, client, dates) → metadata only, minimal embedding
    - Line items (services, products) → FULL embedding
    - Totals (amounts, tax) → metadata only
    - Payment info (IBAN, instructions) → SKIP
    - Legal terms → SKIP or minimal
    
    For contracts:
    - Parties (who) → metadata + minimal embedding
    - Scope (what) → FULL embedding
    - Terms (duration, amounts) → FULL embedding
    - Legal clauses → SKIP or minimal
    """
    
    # Section boundary patterns for Romanian documents
    INVOICE_SECTIONS = {
        "header": [
            r"(?i)^(?:factur[aă]|invoice)",
            r"(?i)dat[aă]\s+(?:emiterii|facturii)",
            r"(?i)furnizor|emitent|seller",
            r"(?i)client|cump[aă]r[aă]tor|buyer",
        ],
        "line_items": [
            r"(?i)(?:detalii|specifica[țt]ie|pozi[țt]ii)\s+factur",
            r"(?i)servicii\s+furnizate",
            r"(?i)descriere\s+(?:serviciu|produs)",
            r"(?i)produse\s+[șs]i\s+servicii",
            r"(?i)consum\s+energie",
        ],
        "totals": [
            r"(?i)total\s+(?:de\s+plat[aă]|general|factur[aă])",
            r"(?i)subtotal",
            r"(?i)(?:TVA|VAT|tax)",
        ],
        "payment": [
            r"(?i)modalitate\s+de\s+plat[aă]",
            r"(?i)(?:IBAN|cod\s+SWIFT)",
            r"(?i)cont\s+bancar",
            r"(?i)dat[aă]\s+scadent[aă]",
        ],
        "legal": [
            r"(?i)condi[țt]ii\s+generale",
            r"(?i)termeni\s+[șs]i\s+condi[țt]ii",
            r"(?i)(?:conform|potrivit)\s+(?:legii|legisla[țt]iei)",
            r"(?i)ANRE",
            r"(?i)solu[țt]ionarea\s+dispute",
            r"(?i)penalită[țt]i",
        ],
    }
    
    CONTRACT_SECTIONS = {
        "parties": [
            r"(?i)^(?:contract|conven[țt]ie|acord)",
            r"(?i)(?:p[aă]r[țt]ile|parts)\s+contractante",
            r"(?i)\u00eentre.*[șs]i",
        ],
        "scope": [
            r"(?i)obiectul\s+contractului",
            r"(?i)scopul\s+(?:contractului|prezen[ț]ei)",
            r"(?i)servicii\s+prestate",
        ],
        "terms": [
            r"(?i)durata\s+contractului",
            r"(?i)valoare\s+contract",
            r"(?i)pre[țt]|tarif|cost",
            r"(?i)obliga[țt]ii",
        ],
        "legal": [
            r"(?i)clauze\s+finale",
            r"(?i)solu[țt]ionarea\s+litigiilor",
            r"(?i)for[țt][aă]\s+major[aă]",
        ],
    }
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        overlap_size: int = 50,
    ):
        """
        Initialize semantic chunker
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size before forcing a split
            overlap_size: Overlap between chunks for context
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Chunk documents semantically
        
        Args:
            documents: Input documents (typically full pages or full documents)
            
        Returns:
            Dictionary with chunked documents
        """
        if not documents:
            return {"documents": []}
        
        chunked_docs = []
        for doc in documents:
            doc_type = doc.meta.get("detected_document_type", "other")
            
            if doc.meta.get("requires_semantic_chunking", False):
                # Semantic chunking for invoices/contracts
                chunks = self._semantic_chunk(doc, doc_type)
                logger.info(f"Semantically chunked {doc.meta.get('file_path', 'doc')} into {len(chunks)} sections")
            else:
                # Fall back to simple chunking for unknown types
                chunks = self._simple_chunk(doc)
                logger.info(f"Simple chunked {doc.meta.get('file_path', 'doc')} into {len(chunks)} chunks")
            
            chunked_docs.extend(chunks)
        
        return {"documents": chunked_docs}
    
    def _semantic_chunk(self, doc: Document, doc_type: str) -> List[Document]:
        """
        Chunk document by semantic sections
        
        Args:
            doc: Input document
            doc_type: Document type (factura, contract, etc.)
            
        Returns:
            List of chunked documents with section metadata
        """
        content = doc.content
        
        # Get section patterns for this document type
        if doc_type == "factura":
            section_patterns = self.INVOICE_SECTIONS
        elif doc_type == "contract":
            section_patterns = self.CONTRACT_SECTIONS
        else:
            # Fall back to simple chunking
            return self._simple_chunk(doc)
        
        # Detect section boundaries
        sections = self._detect_sections(content, section_patterns)
        
        # Create chunks from sections
        chunks = []
        for idx, (section_type, section_text, start_pos) in enumerate(sections):
            # Skip if too small
            if len(section_text.strip()) < self.min_chunk_size:
                continue
            
            # Split large sections
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_large_section(section_text, self.max_chunk_size, self.overlap_size)
            else:
                sub_chunks = [section_text]
            
            for sub_idx, chunk_text in enumerate(sub_chunks):
                chunk_doc = Document(
                    content=chunk_text,
                    meta={
                        **doc.meta,
                        "section_type": section_type,
                        "section_index": idx,
                        "sub_chunk_index": sub_idx,
                        "chunk_char_start": start_pos,
                        "chunk_char_count": len(chunk_text),
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks if chunks else [doc]  # Return original if no sections found
    
    def _detect_sections(
        self,
        content: str,
        section_patterns: Dict[str, List[str]]
    ) -> List[Tuple[str, str, int]]:
        """
        Detect section boundaries in document
        
        Args:
            content: Document text
            section_patterns: Dictionary of section patterns
            
        Returns:
            List of (section_type, section_text, start_position) tuples
        """
        # Find all section boundaries
        boundaries = []
        for section_type, patterns in section_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.MULTILINE)
                for match in pattern.finditer(content):
                    boundaries.append((match.start(), section_type))
        
        # Sort by position
        boundaries.sort()
        
        # Extract sections
        sections = []
        for i, (start_pos, section_type) in enumerate(boundaries):
            # Find end position (start of next section or end of document)
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(content)
            
            section_text = content[start_pos:end_pos]
            sections.append((section_type, section_text, start_pos))
        
        # If no sections detected, treat entire content as "unknown" section
        if not sections:
            sections.append(("unknown", content, 0))
        
        return sections
    
    def _split_large_section(
        self,
        text: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split a large section into smaller chunks with overlap
        
        Args:
            text: Section text
            max_size: Maximum chunk size
            overlap: Overlap size between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(end - 100, start)
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
                if sentence_ends:
                    end = search_start + sentence_ends[-1]
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _simple_chunk(self, doc: Document) -> List[Document]:
        """
        Fall back to simple word-based chunking
        
        Args:
            doc: Input document
            
        Returns:
            List of chunked documents
        """
        words = doc.content.split()
        chunk_size = 300  # words
        overlap = 50  # words
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_doc = Document(
                content=chunk_text,
                meta={
                    **doc.meta,
                    "section_type": "simple_chunk",
                    "chunk_index": i // (chunk_size - overlap),
                }
            )
            chunks.append(chunk_doc)
        
        return chunks if chunks else [doc]


if __name__ == "__main__":
    """Test semantic chunker"""
    logging.basicConfig(level=logging.INFO)
    
    chunker = SemanticDocumentChunker()
    
    # Test invoice
    invoice_content = """
    FACTURA FISCALA
    Nr. 12345
    Data emiterii: 18.01.2025
    
    Furnizor: S.C. Electrica S.A.
    Client: KELEMEN SANDOR
    
    DETALII FACTURA:
    - Energia electrică consumată: 1234 kWh
    - Perioada: 01.01.2025 - 31.01.2025
    - Cost energie: 2000 Lei
    
    TOTAL DE PLATĂ: 2641.91 Lei
    TVA: 19%
    
    Modalitate de plată:
    IBAN: RO49 AAAA 1B31 0075 9384 0000
    
    CONDIȚII GENERALE:
    Conform legislației în vigoare...
    ANRE dispune soluționarea disputelor...
    Penalități de întârziere...
    """
    
    invoice_doc = Document(
        content=invoice_content,
        meta={"detected_document_type": "factura", "requires_semantic_chunking": True}
    )
    
    result = chunker.run(documents=[invoice_doc])
    
    print(f"\n{'='*80}")
    print("SEMANTIC CHUNKING RESULTS")
    print(f"{'='*80}")
    
    for i, doc in enumerate(result["documents"]):
        print(f"\nChunk {i+1}:")
        print(f"Section type: {doc.meta.get('section_type', 'unknown')}")
        print(f"Length: {len(doc.content)} chars")
        print(f"Content preview: {doc.content[:100]}...")
