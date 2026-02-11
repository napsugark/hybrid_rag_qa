"""
Boilerplate Filter Component
Removes legal/payment boilerplate from documents before embedding
Specialized for Romanian invoices and official documents
"""

import re
import logging
from typing import List, Dict, Any
from haystack import component, Document

logger = logging.getLogger(__name__)


@component
class BoilerplateFilter:
    """
    Filters out boilerplate content (legal terms, payment instructions)
    before embedding to improve retrieval quality
    
    Removes sections containing:
    - Legal disclaimers
    - Payment terms and instructions
    - Standard contractual clauses
    - Environmental disclosures
    - Dispute resolution text
    
    Keeps sections containing:
    - Line items and services
    - Specific amounts and dates
    - Custom contract terms
    - Project/work descriptions
    """
    
    # Romanian invoice/contract boilerplate patterns
    BOILERPLATE_PATTERNS = [
        # Legal terms
        (r"(?i)condi[țt]ii\s+generale", 3),
        (r"(?i)termeni\s+[șs]i\s+condi[țt]ii", 3),
        (r"(?i)conform\s+(?:legii|legisla[țt]iei|prevederilor)", 2),
        (r"(?i)(?:în|potrivit)\s+conformitate\s+cu", 2),
        
        # Payment/banking boilerplate
        (r"(?i)plata\s+se\s+va\s+efectua", 2),
        (r"(?i)modalit[aăț]+i?\s+de\s+plat[aă]", 1),
        (r"(?i)date\s+bancare", 1),
        (r"(?i)cod\s+SWIFT", 1),
        
        # Penalties and legal consequences
        (r"(?i)penalită[țt]i\s+(?:de|pentru)\s+[îi]nt[âa]rziere", 2),
        (r"(?i)dob[âa]nzi\s+pentru\s+neplat[aă]", 2),
        (r"(?i)în\s+caz\s+de\s+neplat[aă]", 1),
        
        # ANRE and regulatory
        (r"(?i)\bANRE\b", 2),
        (r"(?i)autoritatea\s+na[țt]ional[aă]\s+de\s+reglementare", 2),
        (r"(?i)solu[țt]ionarea\s+dispute", 2),
        (r"(?i)drept\s+la\s+reclama[țt]ie", 1),
        
        # Environmental/energy disclosures
        (r"(?i)emisii\s+(?:de\s+)?CO2", 2),
        (r"(?i)de[șs]euri\s+radioactive", 2),
        (r"(?i)surse\s+primare\s+de\s+energie", 1),
        (r"(?i)eficien[țt][aă]\s+energetic[aă]", 1),
        
        # Standard clauses
        (r"(?i)for[țt][aă]\s+major[aă]", 2),
        (r"(?i)cesiunea\s+contractului", 1),
        (r"(?i)clauze\s+finale", 2),
        (r"(?i)legea\s+(?:nr|numarul)", 1),
    ]
    
    def __init__(
        self,
        min_boilerplate_score: int = 3,
        skip_legal_sections: bool = True,
        skip_payment_sections: bool = True,
    ):
        """
        Initialize boilerplate filter
        
        Args:
            min_boilerplate_score: Minimum score to classify as boilerplate
            skip_legal_sections: Remove sections marked as "legal"
            skip_payment_sections: Remove sections marked as "payment"
        """
        self.min_boilerplate_score = min_boilerplate_score
        self.skip_legal_sections = skip_legal_sections
        self.skip_payment_sections = skip_payment_sections
        
        # Compile patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), weight)
            for pattern, weight in self.BOILERPLATE_PATTERNS
        ]
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Filter boilerplate documents
        
        Args:
            documents: Input documents
            
        Returns:
            Dictionary with filtered documents (boilerplate removed)
        """
        if not documents:
            return {"documents": []}
        
        filtered_docs = []
        skipped_count = 0
        
        for doc in documents:
            # Check if document should be filtered
            if self._should_filter(doc):
                skipped_count += 1
                logger.debug(
                    f"Filtering boilerplate chunk: {doc.meta.get('section_type', 'unknown')} "
                    f"(score: {doc.meta.get('boilerplate_score', 0)})"
                )
            else:
                filtered_docs.append(doc)
        
        logger.info(
            f"Boilerplate filter: kept {len(filtered_docs)} chunks, "
            f"removed {skipped_count} boilerplate chunks"
        )
        
        return {"documents": filtered_docs}
    
    def _should_filter(self, doc: Document) -> bool:
        """
        Determine if document should be filtered as boilerplate
        
        Args:
            doc: Document to check
            
        Returns:
            True if document is boilerplate and should be removed
        """
        # Skip based on section type
        section_type = doc.meta.get("section_type", "")
        
        if self.skip_legal_sections and section_type == "legal":
            doc.meta["boilerplate_score"] = 999  # High score for legal sections
            doc.meta["boilerplate_reason"] = "legal_section"
            return True
        
        if self.skip_payment_sections and section_type == "payment":
            doc.meta["boilerplate_score"] = 999  # High score for payment sections
            doc.meta["boilerplate_reason"] = "payment_section"
            return True
        
        # Calculate boilerplate score based on content
        score = self._calculate_boilerplate_score(doc.content)
        doc.meta["boilerplate_score"] = score
        
        if score >= self.min_boilerplate_score:
            doc.meta["boilerplate_reason"] = "high_boilerplate_score"
            return True
        
        return False
    
    def _calculate_boilerplate_score(self, text: str) -> int:
        """
        Calculate boilerplate score for text
        
        Args:
            text: Text to analyze
            
        Returns:
            Boilerplate score (higher = more boilerplate)
        """
        score = 0
        
        for pattern, weight in self.compiled_patterns:
            if pattern.search(text):
                score += weight
        
        return score


if __name__ == "__main__":
    """Test boilerplate filter"""
    logging.basicConfig(level=logging.INFO)
    
    filter_component = BoilerplateFilter()
    
    # Test documents
    boilerplate_doc = Document(
        content="""
        CONDIȚII GENERALE
        Conform legislației în vigoare și prevederilor ANRE, 
        plata se va efectua în termen de 30 zile.
        Penalități de întârziere se vor aplica...
        """,
        meta={"section_type": "legal"}
    )
    
    valuable_doc = Document(
        content="""
        SERVICII FURNIZATE:
        - Energie electrică: 1234 kWh
        - Perioada: Ianuarie 2025
        - Cost unitar: 1.62 Lei/kWh
        - Total: 2000 Lei
        """,
        meta={"section_type": "line_items"}
    )
    
    result = filter_component.run(documents=[boilerplate_doc, valuable_doc])
    
    print(f"\nFiltered results: {len(result['documents'])} documents kept")
    for doc in result["documents"]:
        print(f"- Section type: {doc.meta.get('section_type')}")
        print(f"  Boilerplate score: {doc.meta.get('boilerplate_score', 0)}")
